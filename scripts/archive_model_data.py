"""Archive the WHOLE roster's model-data neighborhood in ONE batched pull, on the MODEL-RUN
cadence -- run this ~4x/day just after each GFS run posts (00/06/12/18Z + ~5 h lag), NOT at
TAF issue time.

    uv run python scripts/archive_model_data.py             # all roster, freshest runs, 48 h
    uv run python scripts/archive_model_data.py --dry-run    # plan + credit estimate, no fetch
    uv run python scripts/archive_model_data.py --force      # run even if the tier is gated off
    uv run python scripts/archive_model_data.py --max-credits 12000   # raise the ceiling

WHY a separate job (see docs/gribstream_model_data.md): a model run (e.g. GFS 06Z) is used by
every TAF issued until the next run posts. Pulling at forecast time re-fetched the SAME run on
each cycle's offset grid (full credits each). Archiving once per run instead -- and, because
coordinates are FREE up to 500, unioning the whole roster (~420 coords) into ONE request-set --
covers all stations for the freshest run; forecasts then READ the archive for 0 credits
(collect.py copy_model_data). One pull/run, ~4/day, ~2 k credits each batched.

`as_of` defaults to NOW, so each model's freshest available run (<= now) is captured -- leakage-
safe for live collection (the archive never holds a run newer than a later forecast's issue).
Gated by MODEL_DATA_ENABLED so the cron can be installed dormant; --force overrides. BILLS
GRIBStream credits, so `--dry-run` prints a per-model estimate and `--max-credits` REFUSES a
pull that would exceed its ceiling (see DEFAULT_MAX_CREDITS).

HRRR/NBM CADENCE (settled 2026-07-28): they update HOURLY, but this job snapshots ~4x/day, so
"the freshest run at fire time" made the guidance behind a forecast an accident of cron timing.
`modeldata.archive_run_and_as_of` now pins them to the SAME 00/06/12/18Z synoptic cycles as
GFS and IFS, so every archived cycle NAMES a run. Zero credit impact; it trades up to ~6 h of
freshness for a cycle that is deterministic and comparable across runs -- the property a
replayable archive needs.
"""

import argparse
from datetime import datetime

from forecaster import modeldata, stations, store
from forecaster.config import settings

# Default per-invocation credit ceiling. This is a GUARDRAIL, not a budget: it exists so a
# mis-typed --hours, an accidentally huge --stations list, or a future variable-bundle change
# cannot quietly bill thousands of credits from cron. It REFUSES the whole pull rather than
# truncating it, because a partially-archived cycle is worse than none -- it looks complete
# and silently omits models or hours nobody would think to check. Round 1's 402 was this
# guardrail missing at the provider; this is the same guard on our side of the wire.
#
# Sized to pass the INTENDED configuration and trip on a doubling. A 10-station roster at
# 48h costs ~5,882 deterministic + ~4,743 for the full 31-member GEFS = ~10,625. The next
# real cliff is the second 500-coord LEVEL chunk at ~14 stations (~16,500 with the ensemble),
# which this refuses. A ceiling that blocks the configuration you meant to run is not a
# guardrail, it is a papercut -- so this must be re-checked if the station list grows.
DEFAULT_MAX_CREDITS = 12000


def _parse_asof(s: str | None) -> datetime | None:
    return datetime.strptime(s.replace("Z", ""), "%Y-%m-%dT%H:%M") if s else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", default=None, help="ISO issue time (default now = freshest runs)")
    ap.add_argument("--hours", type=int, default=48,
                    help="forecast horizon to archive (default 48h: covers every TAF issued "
                         "before the next run posts, +30h validity)")
    ap.add_argument("--step", type=int, default=2, help="surface valid-time step (h)")
    ap.add_argument("--hazard-step", type=int, default=3, help="pressure-level valid-time step (h)")
    ap.add_argument("--no-hazards", action="store_true")
    ap.add_argument("--ensemble", action="store_true",
                    help="also pull the GEFS ensemble for every station in ONE bundle "
                         "(members bill LINEARLY, so this is opt-in; the whole network costs "
                         "the same as one station)")
    ap.add_argument("--ensemble-members", type=int, default=modeldata.GEFS_N_MEMBERS,
                    help=f"GEFS members to archive (default all {modeldata.GEFS_N_MEMBERS}: "
                         "members not fetched cannot be recovered later)")
    ap.add_argument("--stations", default=None, help="override the roster (comma list)")
    ap.add_argument("--db", default=settings.db_path)
    ap.add_argument("--force", action="store_true", help="run even if MODEL_DATA_ENABLED is off")
    ap.add_argument("--dry-run", action="store_true", help="print the plan + estimate, no fetch")
    ap.add_argument("--max-credits", type=int, default=DEFAULT_MAX_CREDITS,
                    help=f"refuse the pull if the estimate exceeds this (default "
                         f"{DEFAULT_MAX_CREDITS}); 0 disables the ceiling")
    args = ap.parse_args()

    if not settings.model_data_enabled and not args.force and not args.dry_run:
        print("model-data tier OFF (MODEL_DATA_ENABLED=false) -- skipping archive. Use --force to override.")
        return

    icaos = ([s.strip().upper() for s in args.stations.split(",") if s.strip()]
             if args.stations else stations.icaos())
    hazards = not args.no_hazards
    as_of = _parse_asof(args.as_of)
    est = modeldata.estimate_prefetch_many(
        icaos, as_of=as_of, hours=args.hours, step_h=args.step,
        hazards=hazards, hazard_step_h=args.hazard_step)
    print(f"=== ARCHIVE model-data: {len(icaos)} station(s), {args.hours}h horizon ===")
    print(f"surface {args.hours}h/{args.step}h grid ({est['surface_times']} valid times, "
          f"{est['coords']} coords); levels {args.hazard_step}h grid "
          f"({est['level_times']} valid times, {est['hazard_coords']} coords)")
    for model, m in est["per_model"].items():
        print(f"  {model:9} surface {m['surface']:>6}  levels {m['levels']:>6} "
              f"({m['level_vars']} vars)")
    if est["steering_probe"]:
        print(f"  {'steering':9} probe   {est['steering_probe']:>6}")
    total = est["credits"]

    members = tuple(range(max(1, min(args.ensemble_members, modeldata.GEFS_N_MEMBERS))))
    if args.ensemble:
        ens = modeldata.estimate_ensemble(icaos, hours=args.hours, members=members)
        print(f"  {'gefs':9} ensemble{ens['credits']:>6} ({ens['valid_times']} times x "
              f"{ens['vars']} vars x {ens['members']} members; flat in station count)")
        total += ens["credits"]
    print(f"estimated credits (BATCHED union): ~{total}")
    for n in est["notes"]:
        print(f"  - {n}")

    # Refuse, do not truncate: a half-archived cycle looks complete and silently omits data.
    if args.max_credits and total > args.max_credits:
        print(f"\nREFUSED: estimate ~{total} exceeds the ceiling of {args.max_credits}. "
              "Nothing was fetched and no credits were spent. Narrow --stations/--hours, drop "
              "--ensemble, or raise --max-credits deliberately if this cost is intended.")
        raise SystemExit(2)
    if args.dry_run:
        return

    result = modeldata.prefetch_many(
        icaos, as_of=as_of, hours=args.hours, step_h=args.step,
        hazards=hazards, hazard_step_h=args.hazard_step, db_path=args.db,
    )
    print(f"as_of pinned: {result['as_of']:%Y-%m-%dT%H:%MZ}; union {result['coords']} coords, "
          f"{result['requests']} request(s); inserted {result['rows_inserted']} rows; "
          f"credits charged {result['credits_charged']}")
    if result["notes"]:
        for n in result["notes"]:
            print(f"  - {n}")

    if args.ensemble:
        ens = modeldata.prefetch_ensemble_many(
            icaos, as_of=as_of, hours=args.hours, members=members, db_path=args.db)
        print(f"GEFS ensemble: {ens['coords']} coord(s), {len(ens['members'])} members, "
              f"{ens['window']}; inserted {ens['inserted']} rows; "
              f"credits charged {ens['credits_charged']}")
        for n in ens["notes"]:
            print(f"  - {n}")

    con = store.connect(args.db, read_only=True)
    try:
        print(f"archive now holds {len(store.model_data_locations(con))} distinct locations")
    finally:
        con.close()


if __name__ == "__main__":
    main()
