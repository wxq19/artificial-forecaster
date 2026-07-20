"""Run the TAF agent for one station at issue time and persist it, paired with the human
TAF (the model-forecast collection cron; M4 step 2).

Leakage discipline (the whole point of collecting LIVE):
  - obs are fetched ONCE into the BENCHMARK DB (truth banking: successive cycles tile the
    timeline, so verification obs accumulate as a side effect of collection), then the
    pre-cutoff back-window is COPIED into a THROWAWAY per-run DB with the cutoff enforced
    in SQL (store.copy_obs), so the model's DB tools can never see an ob at or after the
    scheduled ISSUE time -- even if a forecaster posted a few minutes early;
  - get_current_taf is DROPPED from the toolset, so the model cannot fetch the official
    TAF it is being scored against; get_previous_taf serves only a PRIOR-cycle bulletin
    (issue-time buffer AND valid_from strictly before this run's valid_from);
  - the human TAF for this station is archived here too (frozen at issue time), so the
    pair is captured together; scoring happens LATER (score_taf.py --pending), once the
    window elapses and obs accumulate.

One invocation = ONE matrix cell (model x temperature x worksheet_mode); the cron/wrapper
fans out the matrix. All persistence runs under the single-writer lock.

  uv run python scripts/collect.py --station KWRI --model google/gemma-4-31B-it
  uv run python scripts/collect.py --station KBAB --issue-time 2026-07-16T2300Z --temperature 0
"""

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from forecaster import awc, modeldata, neighbors, stations, store, tafgen
from forecaster import worksheet as wksht
from forecaster.agent import AgentConfig, run_agent
from forecaster.config import settings
from forecaster.runlog import persist_run
from forecaster.tools import EMIT_TAF, GET_PREVIOUS_TAF, SUBMIT_WORKSHEET, TOOLS

TOOL_CAPS = {"get_map": 8, "get_sounding": 8, "get_fcst_sounding": 8, "get_point_forecast": 8}


def _floor_hour(t: datetime) -> datetime:
    """Military TAF valid-from = the issue HOUR (they issue at the top of the cycle hour,
    valid from that same hour). Floor the issue time to the hour. A scheduled cron run at
    the station's cycle time therefore lands valid-from exactly on the cycle boundary."""
    return t.replace(minute=0, second=0, microsecond=0)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def _system_prompt(max_steps: int, mode: str, taf_access: bool, model_data: bool = False) -> str:
    """The USAF-forecaster system prompt for a COLLECTION run -- the tool list omits
    get_current_taf on purpose (leakage guard); get_previous_taf appears only when this
    cell grants prior-TAF access; the get_model_* tools appear only when model_data is on
    (naming them in the prompt is what makes the model actually call them)."""
    gate = {
        "off": "Reason, then call emit_taf.",
        "advisory": "Fill and submit a worksheet (submit_taf_worksheet) BEFORE emit_taf. Its "
                    "findings are advisory -- address them, but you may emit once your reasoning is sound.",
        "required": "You MUST submit a worksheet (submit_taf_worksheet) that passes its completeness "
                    "check BEFORE emit_taf is accepted. If emit_taf is refused, fix the worksheet and "
                    "re-submit, then emit.",
    }[mode]
    prev = (" get_previous_taf (the prior official TAF, for continuity)," if taf_access else "")
    md = (" get_model_state (hourly NWP surface guidance -- T/Td/wind/MSLP/ceiling from "
          "GFS/HRRR/NBM), get_hazard_scan (icing + convective/turbulence environment, "
          "cross-model), get_model_verification (recent model-vs-obs bias), "
          "get_nearby_model_data (a field at upstream points)," if model_data else "")
    s = (
        "You are a USAF weather forecaster issuing terminal aerodrome forecasts under AFMAN "
        "15-124. Tools: query_obs/get_latest_obs (stored METARs), get_trend (meteogram), "
        "get_sounding/get_fcst_sounding (skew-Ts), get_map (synoptic charts), get_point_forecast "
        "(hourly model point forecast), get_climo (typical conditions), get_imagery (sat/radar), "
        "get_terrain (local terrain + coastline with the nearby airfields plotted on a relief "
        "map), get_nearby_obs (latest observations from those neighbors),"
        + md + prev + " check_taf (AFMAN dry-run), and emit_taf (submit the forecast). Each data-tool "
        "receipt begins with an [evidence_id: ev_NNN] you can cite. " + gate + " Think step by step, "
        "gather what you need, and base the forecast only on tool data. Place the field in its "
        "mesoscale setting: use get_terrain to see the surrounding terrain and airfields, then "
        "get_nearby_obs for the upwind/relevant neighbors to judge whether a restriction is "
        "regional or local, what may advect in, and any terrain-driven effect (upslope/downslope, "
        "sea breeze, valley cold-air pooling). "
        f"You have up to {max_steps} tool-calling turns -- take the time to reason thoroughly."
    )
    if model_data:
        s += (" Model guidance is available: anchor quantitative trends -- temperature extremes "
              "(TX/TN), the pressure/QNH trend, wind and gusts -- to get_model_state and "
              "get_point_forecast rather than extrapolating the last observation, use "
              "get_hazard_scan to judge convective and icing risk, and get_model_verification "
              "to weight the model your recent obs say is less biased.")
    if mode != "off":
        s += "\n\n" + wksht.worksheet_guide(settings.evidence_mode)
    return s + "\n\n" + tafgen.emit_taf_guide()


def _task_prompt(st: stations.Station, valid_from: datetime) -> str:
    proxy = st.bufkit_proxy
    note = (f"\nNOTE: {st.icao} has surface observations but NO model BUFKIT output -- use nearby "
            f"{proxy} for get_fcst_sounding and get_point_forecast." if proxy else "")
    return (
        f"Produce a {st.taf_hours}-hour Air Force TAF for {st.icao} ({st.name}), valid from "
        f"{valid_from:%d%H%M}Z ({valid_from:%Y-%m-%d %H:%MZ}).{note} "
        "Begin by checking current and recent conditions."
    )


def _load_metar_retry(icao: str, hours: float, bench_db, tries: int = 3):
    """awc.load_metar with a short retry: AWC intermittently returns an empty body that
    trips json.loads (the rc=1 crash). The transient clears in a second or two, so a brief
    backoff turns a per-cell crash into a reliable single fetch. Raises if all tries fail."""
    for i in range(tries):
        try:
            return awc.load_metar(icao, hours=hours, db_path=bench_db)
        except Exception:  # noqa: BLE001 -- retry the transient; re-raise on the last attempt
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def _ingest_obs(icao: str, model_icao: str, neighbor_icaos: list[str],
                hours: float, bench_db) -> dict:
    """Fetch+bank obs for the home station (+ proxy + neighbors) into the benchmark DB, ONCE.
    insert_obs is idempotent, so the model cells later COPY these banked obs (store.copy_obs)
    rather than each re-hitting AWC. Returns the obs-feed summary for the run record."""
    load = _load_metar_retry(icao, hours, bench_db)
    if model_icao != icao:                  # also bank the proxy's obs for the model tools
        load = {"base": load, "proxy": _load_metar_retry(model_icao, hours, bench_db)}
    for nb in neighbor_icaos:
        try:
            _load_metar_retry(nb, hours, bench_db)
        except Exception as e:  # noqa: BLE001 -- a dud neighbor is skipped, not fatal
            print(f"  neighbor {nb} ingest skipped ({type(e).__name__}: {e})")
    return load


def main() -> int:
    t0 = datetime.now(timezone.utc).replace(tzinfo=None)   # process start -> runs.duration_s
    ap = argparse.ArgumentParser(description="Collect one agent TAF run, paired with the human TAF.")
    ap.add_argument("--station", required=True, help="roster ICAO (see forecaster.stations)")
    ap.add_argument("--model", default=settings.llm_model, help="model id (default: settings.llm_model)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--mode", default=settings.worksheet_mode, choices=["off", "advisory", "required"])
    ap.add_argument("--taf-access", action=argparse.BooleanOptionalAction, default=True,
                    help="give the model the leakage-safe previous TAF via get_previous_taf")
    ap.add_argument("--issue-time", help="UTC issue time (e.g. 2026-07-16T2300Z); default: now")
    ap.add_argument("--ingest-hours", type=float, default=24.0,
                    help="pre-cutoff obs back-window (24 matches get_trend's default look-back)")
    ap.add_argument("--neighbors", action=argparse.BooleanOptionalAction, default=True,
                    help="also ingest nearby-station obs for get_nearby_obs (spatial awareness)")
    ap.add_argument("--model-data", action=argparse.BooleanOptionalAction,
                    default=settings.model_data_enabled,
                    help="GRIBStream model-data tier: prefetch the station's model neighborhood at "
                         "issue time and grant the get_model_* tools (BILLS credits; off by default)")
    ap.add_argument("--ingest-only", action="store_true",
                    help="fetch+bank obs (home+proxy+neighbors) into the benchmark DB and exit -- "
                         "the scheduler runs this ONCE per station so the matrix cells share one "
                         "AWC fetch instead of each repeating it")
    ap.add_argument("--ingest", action=argparse.BooleanOptionalAction, default=True,
                    help="--no-ingest skips the AWC obs fetch and reads obs already banked by a "
                         "prior --ingest-only pass (the scheduler uses this for the matrix cells)")
    ap.add_argument("--max-steps", type=int, default=24, help="tool-calling turns (ample by default)")
    ap.add_argument("--max-tokens", type=int, default=16000, help="per-turn completion budget")
    ap.add_argument("--db", default=None, help="benchmark DB path (default: settings.db_path)")
    args = ap.parse_args()

    icao = args.station.upper()
    if icao not in stations.BY_ICAO:
        ap.error(f"{icao} is not on the roster ({', '.join(stations.icaos())})")
    st = stations.BY_ICAO[icao]
    settings.worksheet_mode = args.mode         # keep the sink's config in step with the gate

    issue = (datetime.fromisoformat(args.issue_time.rstrip("Z")) if args.issue_time
             else datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0, second=0))
    valid_from = _floor_hour(issue)
    valid_to = valid_from + timedelta(hours=st.taf_hours)
    model_icao = stations.model_station(icao)
    bench_db = args.db or settings.db_path
    # Nearest-neighbor airfields for spatial awareness (get_nearby_obs); best-effort per
    # neighbor so one dud id (no live AWC METAR) never sinks the run. Banked into the
    # benchmark DB and copied under the SAME cutoff as the home station, so the tool is
    # leakage-safe by the same SQL guard.
    neighbor_icaos = ([n[0] for n in neighbors.neighbors_of(icao)]
                      if args.neighbors else [])

    # --ingest-only: the scheduler runs this ONCE per station so the matrix cells share a
    # SINGLE AWC fetch (and one frozen obs snapshot) instead of each cell re-hitting AWC.
    # Bank obs (home + proxy + neighbors) into the benchmark DB and exit; insert_obs is
    # idempotent, so the cells then COPY these under --no-ingest.
    if args.ingest_only:
        with store.write_lock(args.db):
            load = _ingest_obs(icao, model_icao, neighbor_icaos, args.ingest_hours, bench_db)
        print(f"[{datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ}] ingest-only {icao}: {json.dumps(load)}")
        # Prefetch the model neighborhood ONCE per station/cycle (like obs banking): as_of =
        # valid_from pins the run cutoff, so the archive is leakage-safe and the matrix cells
        # copy it for 0 credits. Gated: no credits unless --model-data is on.
        if args.model_data:
            md = modeldata.prefetch(icao, as_of=valid_from, db_path=bench_db)
            print(f"  model-data prefetch {icao}: inserted {md['rows_inserted']} rows, "
                  f"{md['credits_charged']} credits" + (f"; notes={md['notes']}" if md["notes"] else ""))
        return 0

    short = args.model.split("/")[-1]
    experiment_id = f"{icao}_{issue:%Y%m%dT%H%M}"           # the collection event (all cells share it)
    taf_tag = "taf" if args.taf_access else "notaf"
    run_id = f"{experiment_id}_{short}_{args.mode}_t{args.temperature}_{taf_tag}"

    # THROWAWAY per-run obs DB (removed in the finally: on Debian 13 /tmp is tmpfs, so a
    # leaked dir per cron cell would accumulate in RAM). Everything below happens inside
    # the try so the temp dir is cleaned even when a step raises.
    run_dir = tempfile.mkdtemp(prefix="collect_")
    run_db = str(Path(run_dir) / "obs.duckdb")
    cutoff = issue - timedelta(minutes=settings.previous_taf_buffer_min)
    prev_taf = None
    try:
        # Under the lock, serialized against the poller/scorer:
        #   1. bank obs for THIS cell into the BENCHMARK DB unless --no-ingest (the scheduler
        #      pre-banked them via a single --ingest-only pass, so the cells reuse the same
        #      frozen snapshot); NO cutoff -- truth banking, the model never reads this DB;
        #   2. copy the pre-cutoff back-window into the per-run DB (cutoff enforced in SQL by
        #      store.copy_obs) for the model's read tools;
        #   3. read the leakage-safe context: the CLIMO product and -- if this cell grants it --
        #      the latest PRIOR-CYCLE human TAF (issue-time buffer AND valid_from strictly before
        #      this run's valid_from, so the current cycle's bulletin can never qualify);
        #   4. stub the runs row, so a cell killed by the scheduler's timeout still leaves a
        #      record (persist_run replaces it by run_id on success).
        with store.write_lock(args.db):
            load = (_ingest_obs(icao, model_icao, neighbor_icaos, args.ingest_hours, bench_db)
                    if args.ingest else {"reused_bank": True, "station": icao})
            bcon = store.connect(bench_db)      # RW so a fresh benchmark DB gets its schema
            try:
                store.init_scoring_schema(bcon)
                store.init_climo_schema(bcon)
                store.init_runs_schema(bcon)
                if args.taf_access:
                    prev_taf = store.previous_human_taf(bcon, icao, before=cutoff,
                                                        valid_before=valid_from)
                store.insert_run(bcon, {
                    "run_id": run_id, "experiment_id": experiment_id, "station": icao,
                    "issue_time_utc": issue, "valid_from_utc": valid_from,
                    "valid_to_utc": valid_to, "producer_kind": "artificial",
                    "model": args.model, "temperature": args.temperature,
                    "max_tokens": args.max_tokens, "worksheet_mode": args.mode,
                    "stop_reason": "incomplete",
                    "fatal": "collection started; killed or timed out before final persist",
                    "created_at": datetime.now(timezone.utc).replace(tzinfo=None)})
            finally:
                bcon.close()
            rcon = store.connect(run_db)
            try:
                # Cutoff anchors on valid_from, NOT issue: a scheduled run has issue ==
                # valid_from, but a manual run keeps its wall-clock minutes -- cutting at
                # `issue` would copy obs from inside the scoring window (e.g. a 03:00 ob
                # into a run invoked at 03:07 forecasting from 03:00).
                n_obs = store.copy_obs(rcon, bench_db, icao,
                                       before=valid_from, hours=args.ingest_hours)
                if model_icao != icao:
                    n_obs += store.copy_obs(rcon, bench_db, model_icao,
                                            before=valid_from, hours=args.ingest_hours)
                for nb in neighbor_icaos:       # same cutoff -> leakage-safe by construction
                    n_obs += store.copy_obs(rcon, bench_db, nb,
                                            before=valid_from, hours=args.ingest_hours)
                # init_scoring_schema so get_previous_taf reads a real (possibly empty)
                # tafs table; copy_climo creates the climo schema itself, so get_climo
                # returns clean "not built" feedback rather than a SQL error.
                store.init_scoring_schema(rcon)
                n_climo = store.copy_climo(rcon, bench_db, icao)
                # Model-data archive: leakage-safe by construction (prefetched with
                # as_of=valid_from), so it copies with NO cutoff -- just the station's
                # coordinate neighborhood. copy_model_data creates its own schema, so the
                # get_model_* tools return clean "not pre-fetched" feedback on an empty archive.
                n_md = (store.copy_model_data(
                            rcon, bench_db,
                            coords=modeldata.station_coords(icao, as_of=valid_from, db_path=bench_db))
                        if args.model_data else store.init_model_data_schema(rcon) or 0)
                if prev_taf:
                    store.insert_taf(rcon, prev_taf)
            finally:
                rcon.close()
        return _run_and_persist(args, st, icao, issue, valid_from, valid_to,
                                run_db, prev_taf, n_climo, n_obs, n_md, load,
                                run_id, experiment_id, t0)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def _run_and_persist(args, st, icao, issue, valid_from, valid_to,
                     run_db, prev_taf, n_climo, n_obs, n_md, load,
                     run_id, experiment_id, started_at) -> int:
    """The agent run + final persist (split from main so the temp-dir finally wraps it)."""
    # Strip get_current_taf (leakage) always; strip the model-data tier unless enabled (a
    # switchable experiment axis -- off means the archive is empty and no credits were spent).
    _MODEL_DATA_TOOLS = {"get_model_state", "get_hazard_scan", "get_model_verification",
                         "get_nearby_model_data"}
    _drop = {"get_current_taf"} | (set() if args.model_data else _MODEL_DATA_TOOLS)
    toolset = [t for t in TOOLS if t["function"]["name"] not in _drop]
    if args.taf_access:
        toolset.append(GET_PREVIOUS_TAF)
    toolset += ([SUBMIT_WORKSHEET] if args.mode != "off" else []) + [EMIT_TAF]

    messages = [{"role": "system", "content": _system_prompt(args.max_steps, args.mode, args.taf_access,
                                                             args.model_data)},
                {"role": "user", "content": _task_prompt(st, valid_from)}]
    cfg = AgentConfig(
        model=args.model, toolset=toolset, max_steps=args.max_steps, max_tokens=args.max_tokens,
        temperature=args.temperature, tool_caps=TOOL_CAPS, worksheet_mode=args.mode,
        step_budget_nudge=True, db_path=run_db,
    )

    print(f"[{datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ}] collect {icao} valid {valid_from:%d%H%M}Z "
          f"| model={args.model} temp={args.temperature} mode={args.mode} seed={cfg.seed} "
          f"taf_access={args.taf_access} climo_months={n_climo} run_obs={n_obs} "
          f"model_data={n_md if args.model_data else 'off'}"
          + (f" (prev {prev_taf['bulletin_type']} {prev_taf['issue_time_utc']:%d%H%MZ})"
             if prev_taf else (" (no prior TAF on file)" if args.taf_access else "")))
    res = run_agent(messages, cfg)
    print(f"  agent: stop={res.stop_reason} convergence={res.convergence} steps={len(res.steps)} "
          f"tokens={res.prompt_tokens}/{res.completion_tokens} clean_taf={res.final_taf is not None}"
          + (f" served={res.served_models}" if res.served_models else ""))
    if res.fatal:
        print(f"  FATAL: {res.fatal}")

    # Single-writer: archive the paired human TAF + persist the run to the benchmark DB
    # (persist_run replaces the stub row inserted before the agent ran).
    with store.write_lock(args.db):
        human = awc.load_taf(icao, db_path=args.db)
        summary = persist_run(
            res, run_id=run_id, station=icao, issue_time=issue,
            valid_from=valid_from, valid_to=valid_to, worksheet_mode=args.mode,
            experiment_id=experiment_id, harness_git_sha=_git_sha(), model=args.model,
            evidence_mode=settings.evidence_mode, db_path=args.db, started_at=started_at)

    print(f"  human TAF: {'NEW ' + str(human['new']) if human['new'] else 'no new bulletin'}")
    print(f"  persisted: run_id={summary['run_id']} taf_id={summary['taf_id']} "
          f"evaluation={summary['evaluation_id']}")
    print(f"  transcript: {summary['transcript_path']}")
    print(f"  obs feed: {json.dumps(load)[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
