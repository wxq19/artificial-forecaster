"""Run the TAF agent for one station at issue time and persist it, paired with the human
TAF (the model-forecast collection cron; M4 step 2).

Leakage discipline (the whole point of collecting LIVE):
  - the cutoff is the scheduled ISSUE time; obs are ingested with awc.load_metar(before=
    cutoff) into a THROWAWAY per-run DB, so the model's DB tools can never see an ob at or
    after issue time -- even if a forecaster posted a few minutes early;
  - get_current_taf is DROPPED from the toolset, so the model cannot fetch the official
    TAF it is being scored against;
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
import subprocess
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from forecaster import awc, stations, store, tafgen
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


def _system_prompt(max_steps: int, mode: str, taf_access: bool) -> str:
    """The USAF-forecaster system prompt for a COLLECTION run -- the tool list omits
    get_current_taf on purpose (leakage guard); get_previous_taf appears only when this
    cell grants prior-TAF access."""
    gate = {
        "off": "Reason, then call emit_taf.",
        "advisory": "Fill and submit a worksheet (submit_taf_worksheet) BEFORE emit_taf. Its "
                    "findings are advisory -- address them, but you may emit once your reasoning is sound.",
        "required": "You MUST submit a worksheet (submit_taf_worksheet) that passes its completeness "
                    "check BEFORE emit_taf is accepted. If emit_taf is refused, fix the worksheet and "
                    "re-submit, then emit.",
    }[mode]
    prev = (" get_previous_taf (the prior official TAF, for continuity)," if taf_access else "")
    s = (
        "You are a USAF weather forecaster issuing terminal aerodrome forecasts under AFMAN "
        "15-124. Tools: query_obs/get_latest_obs (stored METARs), get_trend (meteogram), "
        "get_sounding/get_fcst_sounding (skew-Ts), get_map (synoptic charts), get_point_forecast "
        "(hourly model point forecast), get_climo (typical conditions), get_imagery (sat/radar),"
        + prev + " check_taf (AFMAN dry-run), and emit_taf (submit the forecast). Each data-tool "
        "receipt begins with an [evidence_id: ev_NNN] you can cite. " + gate + " Think step by step, "
        "gather what you need, and base the forecast only on tool data. "
        f"You have up to {max_steps} tool-calling turns -- take the time to reason thoroughly."
    )
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect one agent TAF run, paired with the human TAF.")
    ap.add_argument("--station", required=True, help="roster ICAO (see forecaster.stations)")
    ap.add_argument("--model", default=settings.llm_model, help="model id (default: settings.llm_model)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--mode", default=settings.worksheet_mode, choices=["off", "advisory", "required"])
    ap.add_argument("--taf-access", action=argparse.BooleanOptionalAction, default=True,
                    help="give the model the leakage-safe previous TAF via get_previous_taf")
    ap.add_argument("--issue-time", help="UTC issue time (e.g. 2026-07-16T2300Z); default: now")
    ap.add_argument("--ingest-hours", type=float, default=12.0, help="pre-cutoff obs back-window")
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

    # THROWAWAY per-run obs DB, cut off at issue time (leakage-proof). init_climo_schema so
    # get_climo returns clean "not built" feedback rather than a SQL error on the temp DB;
    # init_scoring_schema so get_previous_taf reads a real (possibly empty) tafs table.
    run_db = str(Path(tempfile.mkdtemp(prefix="collect_")) / "obs.duckdb")
    load = awc.load_metar(model_icao, hours=args.ingest_hours, db_path=run_db, before=issue)
    if model_icao != icao:                      # also feed the base's OWN obs for the DB tools
        load_base = awc.load_metar(icao, hours=args.ingest_hours, db_path=run_db, before=issue)
        load = {"proxy": load, "base": load_base}
    con = store.connect(run_db)
    try:
        store.init_climo_schema(con)
        store.init_scoring_schema(con)
    finally:
        con.close()

    # Read the leakage-safe context out of the benchmark archive (under one lock, serialized
    # against the poller): the station's CLIMO product, and -- if this cell grants it -- the
    # latest HUMAN TAF issued >= previous_taf_buffer_min before the cutoff. Both are copied
    # into the per-run DB so get_climo / get_previous_taf serve them without a rebuild and
    # can never reach past the cutoff.
    bench_db = args.db or settings.db_path
    cutoff = issue - timedelta(minutes=settings.previous_taf_buffer_min)
    prev_taf = None
    with store.write_lock(args.db):
        bcon = store.connect(bench_db)          # RW so a fresh benchmark DB gets its schema
        try:
            store.init_scoring_schema(bcon)
            store.init_climo_schema(bcon)
            if args.taf_access:
                prev_taf = store.previous_human_taf(bcon, icao, before=cutoff)
        finally:
            bcon.close()
        rcon = store.connect(run_db)
        try:
            n_climo = store.copy_climo(rcon, bench_db, icao)
            if prev_taf:
                store.insert_taf(rcon, prev_taf)
        finally:
            rcon.close()

    toolset = [t for t in TOOLS if t["function"]["name"] != "get_current_taf"]
    if args.taf_access:
        toolset.append(GET_PREVIOUS_TAF)
    toolset += ([SUBMIT_WORKSHEET] if args.mode != "off" else []) + [EMIT_TAF]

    messages = [{"role": "system", "content": _system_prompt(args.max_steps, args.mode, args.taf_access)},
                {"role": "user", "content": _task_prompt(st, valid_from)}]
    cfg = AgentConfig(
        model=args.model, toolset=toolset, max_steps=args.max_steps, max_tokens=args.max_tokens,
        temperature=args.temperature, tool_caps=TOOL_CAPS, worksheet_mode=args.mode,
        step_budget_nudge=True, db_path=run_db,
    )

    print(f"[{datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ}] collect {icao} valid {valid_from:%d%H%M}Z "
          f"| model={args.model} temp={args.temperature} mode={args.mode} seed={cfg.seed} "
          f"taf_access={args.taf_access} climo_months={n_climo}"
          + (f" (prev {prev_taf['bulletin_type']} {prev_taf['issue_time_utc']:%d%H%MZ})"
             if prev_taf else (" (no prior TAF on file)" if args.taf_access else "")))
    res = run_agent(messages, cfg)
    print(f"  agent: stop={res.stop_reason} convergence={res.convergence} steps={len(res.steps)} "
          f"tokens={res.prompt_tokens}/{res.completion_tokens} clean_taf={res.final_taf is not None}"
          + (f" served={res.served_models}" if res.served_models else ""))
    if res.fatal:
        print(f"  FATAL: {res.fatal}")

    short = args.model.split("/")[-1]
    experiment_id = f"{icao}_{issue:%Y%m%dT%H%M}"           # the collection event (all cells share it)
    taf_tag = "taf" if args.taf_access else "notaf"
    run_id = f"{experiment_id}_{short}_{args.mode}_t{args.temperature}_{taf_tag}"

    # Single-writer: archive the paired human TAF + persist the run to the benchmark DB.
    with store.write_lock(args.db):
        human = awc.load_taf(icao, db_path=args.db)
        summary = persist_run(
            res, run_id=run_id, station=icao, issue_time=issue,
            valid_from=valid_from, valid_to=valid_to, worksheet_mode=args.mode,
            experiment_id=experiment_id, harness_git_sha=_git_sha(), model=args.model,
            evidence_mode=settings.evidence_mode, db_path=args.db)

    print(f"  human TAF: {'NEW ' + str(human['new']) if human['new'] else 'no new bulletin'}")
    print(f"  persisted: run_id={summary['run_id']} taf_id={summary['taf_id']} "
          f"evaluation={summary['evaluation_id']}")
    print(f"  transcript: {summary['transcript_path']}")
    print(f"  obs feed: {json.dumps(load)[:200]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
