"""Pre-fetch a station's model-data neighborhood into the model_data archive for one cycle.

    uv run python scripts/prefetch_model_data.py --station KMSP
    uv run python scripts/prefetch_model_data.py --station KWRI --as-of 2026-07-17T18:00Z \
        --hours 30 --step 2 --no-hazards

Pulls GFS/HRRR/NBM surface fields (+ GFS/HRRR pressure-level hazards) via GRIBStream and
writes them to the benchmark DB under write_lock, leakage-safe by construction (as_of pins
the run cutoff). BILLS CREDITS -- credits = valid_times * variables * ceil(coords/500), so
points are free; discipline stays on --hours/--step and hazard levels. --dry-run prints the
coordinate + variable plan and the estimated credit cost WITHOUT fetching.
"""

import argparse
import math
from datetime import datetime

from forecaster import modeldata, store
from forecaster.config import settings


def _parse_asof(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.strptime(s.replace("Z", ""), "%Y-%m-%dT%H:%M")


def _estimate(station: str, hours: int, step_h: int, hazards: bool, hazard_step_h: int) -> int:
    """Rough credit estimate (valid_times * variables * ceil(coords/500)) for the plan."""
    coords = len(modeldata.coords_for(station))
    haz_coords = len(modeldata.hazard_coords(station)) if hazards else 0
    n_sfc = hours // step_h + 1
    n_haz = hours // hazard_step_h + 1
    total = 0
    for model in modeldata.MODELS:
        total += n_sfc * len(modeldata._surface_vars(model)) * math.ceil(coords / 500)
        if hazards and modeldata._hazard_vars(model):
            total += n_haz * len(modeldata._hazard_vars(model)) * math.ceil(haz_coords / 500)
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--station", help="one roster/ICAO station")
    g.add_argument("--stations", help="comma list for a BATCHED prefetch (shared issue time)")
    ap.add_argument("--as-of", default=None, help="ISO issue time, e.g. 2026-07-17T18:00Z (default now)")
    ap.add_argument("--hours", type=int, default=30)
    ap.add_argument("--step", type=int, default=2, help="surface valid-time step (h)")
    ap.add_argument("--hazard-step", type=int, default=3, help="pressure-level valid-time step (h)")
    ap.add_argument("--no-hazards", action="store_true", help="surface fields only (cheaper)")
    ap.add_argument("--models", default=None, help="comma list subset of gfs,hrrr,nbm")
    ap.add_argument("--db", default=settings.db_path)
    ap.add_argument("--dry-run", action="store_true", help="print the plan + credit estimate, no fetch")
    args = ap.parse_args()

    hazards = not args.no_hazards
    models = tuple(m.strip() for m in args.models.split(",")) if args.models else modeldata.MODELS
    stations_list = ([args.station.upper()] if args.station
                     else [s.strip().upper() for s in args.stations.split(",") if s.strip()])
    as_of = _parse_asof(args.as_of)

    est = sum(_estimate(s, args.hours, args.step, hazards, args.hazard_step) for s in stations_list)
    print(f"=== PREFETCH {', '.join(stations_list)} ===")
    print(f"stations: {len(stations_list)}; hazards={hazards}; models={','.join(models)}")
    print(f"surface {args.hours}h/{args.step}h grid, hazards {args.hours}h/{args.hazard_step}h grid")
    print(f"estimated credits (uncached, per-station sum; BATCHING points can lower this): ~{est}")
    if args.dry_run:
        for s in stations_list:
            coords = modeldata.station_coords(s, as_of=as_of, db_path=args.db)
            print(f"  {s}: {len(coords)} coords")
        return

    result = modeldata.prefetch_many(
        stations_list, as_of=as_of, models=models,
        hours=args.hours, step_h=args.step, hazards=hazards,
        hazard_step_h=args.hazard_step, db_path=args.db,
    )
    print(f"as_of pinned: {result['as_of']:%Y-%m-%dT%H:%MZ}; union {result['coords']} coords in "
          f"{result['requests']} request(s)")
    print(f"rows flattened {result['rows_flattened']}, inserted {result['rows_inserted']} "
          f"(dupes skipped = idempotent)")
    print(f"credits charged this run: {result['credits_charged']}")
    if result["notes"]:
        print("notes:")
        for n in result["notes"]:
            print(f"  - {n}")

    con = store.connect(args.db, read_only=True)
    try:
        locs = store.model_data_locations(con)
        print(f"\narchive now holds {len(locs)} distinct locations:")
        for lc in locs[:12]:
            print(f"  {lc['loc_id']:<10} ({lc['lat']:.4f},{lc['lon']:.4f}) "
                  f"models={lc['models']} rows={lc['n_rows']}")
        if len(locs) > 12:
            print(f"  ... +{len(locs) - 12} more")
    finally:
        con.close()


if __name__ == "__main__":
    main()
