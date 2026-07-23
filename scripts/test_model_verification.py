"""Comprehension test: does the model UNDERSTAND get_model_verification?

The tool was rebuilt 2026-07-22 to show every field (T/Td/QNH/wind/gust) grouped by
model run, on the theory that a reader can then see for itself that the fresher run is
closer. It has never been in front of a live agent. This driver puts it there, alone
with two neighbours, and asks three questions whose answers are checkable off the same
table the model was shown.

WHAT THE LOG CONTAINS: the tool's rendered text VERBATIM (the exact bytes the model
read), a reference answer computed independently from the archive, and the model's
reply -- so the reference and the reply can be compared by eye without re-running
anything.

    uv run python scripts/test_model_verification.py --dry-run     # plan + credit estimate
    uv run python scripts/test_model_verification.py               # fetch, ask, log

Credits: surface-only, one model, a short time grid -- roughly (back_hours + hours) /
step_h * 9 variables. Hazard levels are OFF; verification never reads them.

This driver imports a few private helpers from tools.py on purpose: the reference
answer must be computed from the SAME rows by the SAME arithmetic the renderer used,
or a mismatch would say more about the checker than about the model.
"""

import argparse
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from forecaster import agent, awc, modeldata, store, tools
from forecaster.config import settings
from forecaster.llm import client

QUESTIONS = (
    "1. Which GFS run should I trust most for TEMPERATURE, and is that run running too "
    "warm or too cold? Give the size of the bias.\n"
    "2. What is the model's WIND GUST bias? If raw GFS guidance gives me a 30 kt gust "
    "for tomorrow afternoon, what gust would you put in the TAF, and why?\n"
    "3. Look at the wind DIRECTION line. It reports a mean error and a typical miss. "
    "Do those two numbers agree, and which one should I act on?"
)

SYSTEM = (
    "You are an Air Force weather forecaster preparing to write a TAF. You have tools "
    "for what the models are FORECASTING and for how those model runs have VERIFIED "
    "against observations. Answer only from what the tools return; do not invent "
    "numbers. State each answer once and stop -- do not re-derive it."
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", default="KWRI", help="ICAO (must be a BUFKIT/GRIBStream site)")
    ap.add_argument("--model", default="gfs", choices=list(modeldata.MODELS))
    ap.add_argument("--issue", help="ISO issue time (UTC), default: now floored to the hour")
    ap.add_argument("--back-hours", type=int, default=24,
                    help="hours of PRE-issue tail to verify against obs (default 24)")
    ap.add_argument("--n-runs", type=int, default=3, help="model runs to compare (default 3)")
    ap.add_argument("--step-h", type=int, default=1, help="valid-time step (default 1 = hourly)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan + estimate, no fetch")
    ap.add_argument("--keep-db", action="store_true", help="keep the scratch DB for inspection")
    ap.add_argument("--llm-model", default=settings.llm_model)
    return ap.parse_args()


def reference_answer(con, station: str, model: str) -> str:
    """The checkable facts, recomputed from the archive rather than read off the render."""
    lat, lon, _ = tools._resolve_md_location(con, station, None) or (None, None, None)
    if lat is None:
        return "(no archived model data for this station)"
    rows = store.model_data_series(con, model, lat, lon, start=tools._WIDE_START,
                                   end=tools._WIDE_END, variables=list(tools._VER_ALIASES))
    by_run = tools._pivot_by_run(rows)
    if not by_run:
        return "(no archived rows)"
    all_vt = [vt for r in by_run.values() for vt in r]
    obs = tools._obs_by_hour(con, station, min(all_vt), max(all_vt))
    lines, per_run = [], {}
    for run in sorted(by_run, reverse=True)[:tools._VER_MAX_RUNS]:
        hours = sorted(vt for vt in by_run[run] if vt in obs)
        if not hours:
            continue
        acc: dict = {}
        for vt in hours:
            vm, o = by_run[run][vt], obs[vt]
            df, sf = tools._fcst_wind(vm)
            pairs = {
                "T": (tools._k2c(vm.get("t2m")), o.get("temp_c")),
                "Td": (tools._k2c(vm.get("td2m")), o.get("dewpoint_c")),
                "QNH": (tools._pa2inhg(vm.get("mslp")), o.get("altimeter_inhg")),
                "spd": (tools._ms2kt(vm.get("wind")) if vm.get("wind") else sf, o.get("wind_speed")),
                "gust": (tools._ms2kt(vm.get("gust")), o.get("wind_gust") or 0),
            }
            for k, (f, ob) in pairs.items():
                if f is not None and ob is not None:
                    acc.setdefault(k, []).append(f - ob)
            d = tools._deg_err(df, o.get("wind_dir_deg"))
            if d is not None:
                acc.setdefault("dir", []).append(d)
        if not acc:
            continue
        mean = {k: sum(v) / len(v) for k, v in acc.items()}
        typ_dir = sum(abs(v) for v in acc["dir"]) / len(acc["dir"]) if acc.get("dir") else None
        per_run[run] = mean
        lines.append(
            f"- run {run:%Y-%m-%dT%HZ} ({len(hours)} verified hrs): "
            f"T {mean.get('T', float('nan')):+.1f}C, Td {mean.get('Td', float('nan')):+.1f}C, "
            f"gust {mean.get('gust', float('nan')):+.1f}kt, "
            f"dir mean {mean.get('dir', float('nan')):+.0f}deg / typical "
            f"{'--' if typ_dir is None else f'{typ_dir:.0f}'}deg")
    if not per_run:
        return "(no hours where forecast and observation overlap)"
    best = min(per_run, key=lambda r: abs(per_run[r].get("T", 99)))
    newest = max(per_run)
    t_sign = "too warm" if per_run[best].get("T", 0) > 0 else "too cold"
    g = per_run[newest].get("gust")
    lines += [
        "",
        f"- Q1 reference: smallest |mean T error| is run {best:%HZ} "
        f"({per_run[best].get('T', float('nan')):+.1f}C, i.e. {t_sign}). "
        f"Newest run is {newest:%HZ}.",
        f"- Q2 reference: newest-run gust bias "
        f"{'--' if g is None else f'{g:+.1f} kt'}. The bias is SUBTRACTED to correct it, so "
        f"a 30 kt raw gust implies roughly "
        f"{'--' if g is None else f'{30 - g:.0f} kt'}.",
        "- Q3 reference: mean direction error and typical miss differ whenever errors "
        "cancel in sign; the typical (absolute) miss is the one that describes accuracy.",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    station = args.station.upper()
    issue = (datetime.fromisoformat(args.issue.replace("Z", "")) if args.issue
             else datetime.utcnow()).replace(minute=0, second=0, microsecond=0)
    runs = modeldata.verification_runs(issue, n_runs=args.n_runs)
    n_vars = len(modeldata._verify_vars(args.model))
    window_start = issue - timedelta(hours=args.back_hours)
    per_run = [(r, int((issue - max(r, window_start)).total_seconds() // 3600) // args.step_h + 1)
               for r in runs if max(r, window_start) <= issue]
    est = sum(n for _, n in per_run) * n_vars

    print(f"station        {station}")
    print(f"issue (as_of)  {issue:%Y-%m-%dT%H:%M}Z")
    print(f"model          {args.model}  ({n_vars} verification variables)")
    print(f"window         {window_start:%d %HZ} .. {issue:%d %HZ} every {args.step_h}h")
    for r, n in per_run:
        print(f"  run {r:%Y-%m-%dT%HZ}: {n} valid times "
              f"({max(r, window_start):%d/%HZ}..{issue:%d/%HZ})")
    print(f"estimated credits: ~{est}")
    if args.dry_run:
        print("\n--dry-run: nothing fetched, nothing billed.")
        return

    tmp = tempfile.mkdtemp(prefix="verif_test_")
    db = str(Path(tmp) / "scratch.duckdb")
    try:
        con = store.connect(db)
        store.init_schema(con)
        store.init_model_data_schema(con)
        con.close()

        obs = awc.load_metar(station, hours=args.back_hours + 6, before=issue, db_path=db)
        print(f"\nobs banked: {obs}")

        pull = modeldata.prefetch_verification(
            station, as_of=issue, models=(args.model,), hours_back=args.back_hours,
            n_runs=args.n_runs, step_h=args.step_h, db_path=db)
        print(f"model data: rows={pull.get('rows')} runs={pull.get('runs')} "
              f"credits charged={pull.get('credits_charged')} notes={pull.get('notes')}")

        # The exact bytes the model will read.
        rendered = tools.run_tool("get_model_verification",
                                  {"station": station, "model": args.model}, db_path=db).text
        con = store.connect(db, read_only=True)
        reference = reference_answer(con, station, args.model)
        con.close()
        print("\n=== TOOL OUTPUT (what the model sees) ===\n")
        print(rendered)

        cfg = agent.AgentConfig(
            model=args.llm_model,
            toolset=[tools.GET_MODEL_VERIFICATION, tools.GET_MODEL_STATE, tools.GET_LATEST],
            max_steps=6, max_tokens=8000, temperature=0.0,
            worksheet_mode="off", evidence=False, stop_on_clean_taf=False, db_path=db,
        )
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content":
                     f"You are writing the {issue:%d%H}00Z TAF for {station}. Before you "
                     f"start, check how the recent {args.model.upper()} runs have been "
                     f"verifying, then answer:\n\n{QUESTIONS}"}]
        result = agent.run_agent(messages, cfg, client=client)

        log = build_log(args, station, issue, est, pull, obs, rendered, reference, result)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path("logs") / f"modelverif_{station}_{stamp}.md"
        path.parent.mkdir(exist_ok=True)
        path.write_text(log, encoding="utf-8")
        print(f"\n=== ANSWER ===\n{result.steps[-1].answer or '(no final answer)'}")
        print(f"\ntools used: {dict(result.used)}  stop_reason={result.stop_reason}")
        print(f"log: {path}")
    finally:
        if args.keep_db:
            print(f"scratch DB kept at {db}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


def build_log(args, station, issue, est, pull, obs, rendered, reference, result) -> str:
    md = [f"# get_model_verification comprehension test -- {station}",
          f"_{datetime.now():%Y-%m-%d %H:%M:%S}_", "",
          f"- **LLM:** `{args.llm_model}`  |  **Endpoint:** {settings.llm_base_url}",
          f"- **Station / issue:** {station} / {issue:%Y-%m-%dT%H:%M}Z",
          f"- **NWP model:** {args.model}  |  {args.back_hours}h back, step {args.step_h}h, "
          f"runs compared: {pull.get('runs')}",
          f"- **Credits:** estimated ~{est}, charged {pull.get('credits_charged')} "
          f"(archive rows {pull.get('rows')})",
          f"- **Obs banked:** {obs}",
          "- **Toolset offered:** get_model_verification, get_model_state, get_latest_obs",
          "",
          "## Questions put to the model", "", QUESTIONS, "",
          "## DATA PASSED TO THE MODEL", "",
          "The verbatim text `get_model_verification` returned. This is exactly what the "
          "model read -- every number in its answer should be traceable to this table.", "",
          "```text", rendered, "```", "",
          "## Reference answer (computed independently from the archive)", "",
          reference, "",
          "## Model transcript", ""]
    for s in result.steps:
        md += [f"### Step {s.n}  ({s.finish_reason}, {s.completion_tokens} completion tok)"]
        if s.reasoning:
            md += ["", "<details><summary>reasoning</summary>", "", "```text",
                   s.reasoning.strip(), "```", "</details>"]
        for c in s.calls:
            md += ["", f"**tool call** `{c.get('name')}({c.get('args')})`"]
        if s.content:
            md += ["", s.content]
        if s.answer:
            md += ["", "**FINAL ANSWER**", "", s.answer]
        md += [""]
    md += ["## Result", "",
           f"- stop_reason: `{result.stop_reason}`",
           f"- tools used: `{dict(result.used)}`",
           f"- tokens: prompt {result.prompt_tokens} + completion {result.completion_tokens}"]
    return "\n".join(md)


if __name__ == "__main__":
    main()
