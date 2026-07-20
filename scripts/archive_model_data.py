"""Archive the WHOLE roster's model-data neighborhood in ONE batched pull, on the MODEL-RUN
cadence -- run this ~4x/day just after each GFS run posts (00/06/12/18Z + ~5 h lag), NOT at
TAF issue time.

    uv run python scripts/archive_model_data.py             # all roster, freshest runs, 48 h
    uv run python scripts/archive_model_data.py --dry-run    # plan + credit estimate, no fetch
    uv run python scripts/archive_model_data.py --force      # run even if the tier is gated off

WHY a separate job (see docs/gribstream_model_data.md): a model run (e.g. GFS 06Z) is used by
every TAF issued until the next run posts. Pulling at forecast time re-fetched the SAME run on
each cycle's offset grid (full credits each). Archiving once per run instead -- and, because
coordinates are FREE up to 500, unioning the whole roster (~420 coords) into ONE request-set --
covers all stations for the freshest run; forecasts then READ the archive for 0 credits
(collect.py copy_model_data). One pull/run, ~4/day, ~2 k credits each batched.

`as_of` defaults to NOW, so each model's freshest available run (<= now) is captured -- leakage-
safe for live collection (the archive never holds a run newer than a later forecast's issue).
Gated by MODEL_DATA_ENABLED so the cron can be installed dormant; --force overrides. BILLS
GRIBStream credits.

NOTE (flagged for v2): HRRR/NBM update HOURLY but this job snapshots them only ~4x/day, so a
forecast can read an HRRR/NBM run up to ~6 h old -- fine for GFS/IFS (4x/day native), a real
tradeoff for the rapid-refresh models. Revisit their cadence before enabling v2.
"""

import argparse
import math
from datetime import datetime

from forecaster import modeldata, stations, store
from forecaster.config import settings


def _parse_asof(s: str | None) -> datetime | None:
    return datetime.strptime(s.replace("Z", ""), "%Y-%m-%dT%H:%M") if s else None


def _estimate(icaos: list[str], hours: int, step_h: int, hazards: bool, hazard_step_h: int) -> int:
    """Rough BATCHED credit estimate: coords union across the roster, then times x vars x
    ceil(coords/500). Points are free <=500, so the batch is far cheaper than the per-station sum."""
    surf = modeldata._dedupe([modeldata.coords_for(i) for i in icaos])
    haz = modeldata._dedupe([modeldata.hazard_coords(i) for i in icaos]) if hazards else []
    n_sfc, n_haz = hours // step_h + 1, hours // hazard_step_h + 1
    total = 0
    for model in modeldata.MODELS:
        total += n_sfc * len(modeldata._surface_vars(model)) * math.ceil(len(surf) / 500)
        if hazards and modeldata._hazard_vars(model):
            total += n_haz * len(modeldata._hazard_vars(model)) * math.ceil(len(haz) / 500)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--as-of", default=None, help="ISO issue time (default now = freshest runs)")
    ap.add_argument("--hours", type=int, default=48,
                    help="forecast horizon to archive (default 48h: covers every TAF issued "
                         "before the next run posts, +30h validity)")
    ap.add_argument("--step", type=int, default=2, help="surface valid-time step (h)")
    ap.add_argument("--hazard-step", type=int, default=3, help="pressure-level valid-time step (h)")
    ap.add_argument("--no-hazards", action="store_true")
    ap.add_argument("--stations", default=None, help="override the roster (comma list)")
    ap.add_argument("--db", default=settings.db_path)
    ap.add_argument("--force", action="store_true", help="run even if MODEL_DATA_ENABLED is off")
    ap.add_argument("--dry-run", action="store_true", help="print the plan + estimate, no fetch")
    args = ap.parse_args()

    if not settings.model_data_enabled and not args.force and not args.dry_run:
        print("model-data tier OFF (MODEL_DATA_ENABLED=false) -- skipping archive. Use --force to override.")
        return

    icaos = ([s.strip().upper() for s in args.stations.split(",") if s.strip()]
             if args.stations else stations.icaos())
    hazards = not args.no_hazards
    est = _estimate(icaos, args.hours, args.step, hazards, args.hazard_step)
    print(f"=== ARCHIVE model-data: {len(icaos)} roster stations, {args.hours}h horizon ===")
    print(f"surface {args.hours}h/{args.step}h grid, hazards {args.hours}h/{args.hazard_step}h grid")
    print(f"estimated credits (BATCHED union): ~{est}")
    if args.dry_run:
        return

    result = modeldata.prefetch_many(
        icaos, as_of=_parse_asof(args.as_of), hours=args.hours, step_h=args.step,
        hazards=hazards, hazard_step_h=args.hazard_step, db_path=args.db,
    )
    print(f"as_of pinned: {result['as_of']:%Y-%m-%dT%H:%MZ}; union {result['coords']} coords, "
          f"{result['requests']} request(s); inserted {result['rows_inserted']} rows; "
          f"credits charged {result['credits_charged']}")
    if result["notes"]:
        for n in result["notes"]:
            print(f"  - {n}")

    con = store.connect(args.db, read_only=True)
    try:
        print(f"archive now holds {len(store.model_data_locations(con))} distinct locations")
    finally:
        con.close()


if __name__ == "__main__":
    main()
