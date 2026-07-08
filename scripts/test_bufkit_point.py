"""Spike: build a GenWx-style POINT forecast table from the BUFKIT surface section.

Reuses fcstsounding.fetch_raw to pull a model BUFKIT file, then parses its SURFACE
time-series block (one row per forecast hour: PMSL/T2MS/TD2M/UWND/VWND/LCLD/MCLD/HCLD/
P01M...) into a transposed table like docs/TarpViewer-GenWx.csv -- valid time across the
top, parameters down the side. Derives °F, RH, wind dir/speed, and a Fair/Partly/Mostly/
Overcast 'Gen Wx' from cloud cover. Throwaway spike to eyeball the format before we build
a pointfcst seam + tool. Writes the full table to data/charts/temp/ and prints a sample.
"""

import argparse
import csv
import math
from pathlib import Path

from forecaster import fcstsounding

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--station", default="kmsp", help="lowercase ICAO (default: kmsp)")
_ap.add_argument("--model", default="gfs", choices=list(fcstsounding.MODELS))
_ap.add_argument("--hours", type=int, default=12, help="how many hours to print (default: 12)")
_args = _ap.parse_args()


def parse_surface(text: str) -> tuple[list[str], list[list[str]]]:
    """Return (column names, rows) from the BUFKIT surface block. The header wraps over
    several lines; data starts at the first line whose first token is the numeric STN."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("STN YYMMDD"))
    cols: list[str] = []
    j = start
    while j < len(lines):
        toks = lines[j].split()
        if toks and toks[0].isdigit():        # first data row (station number) -> header done
            break
        cols += toks
        j += 1
    nums: list[str] = []
    for ln in lines[j:]:
        nums += ln.split()
    n = len(cols)
    rows = [nums[k:k + n] for k in range(0, len(nums) - n + 1, n)]
    return cols, rows


def magnus_rh(t_c: float, td_c: float) -> float:
    e = lambda x: 6.112 * math.exp(17.67 * x / (x + 243.5))  # noqa: E731
    return max(0.0, min(100.0, 100.0 * e(td_c) / e(t_c)))


def wind_dir_spd(u: float, v: float) -> tuple[int, int]:
    spd_kt = math.hypot(u, v) * 1.94384
    d = (270.0 - math.degrees(math.atan2(v, u))) % 360.0
    return round(d / 10) * 10, round(spd_kt)


def gen_wx(total_cloud: float) -> str:
    return ("Fair" if total_cloud <= 25 else "Partly Cloudy" if total_cloud <= 50
            else "Mostly Cloudy" if total_cloud <= 87 else "Overcast")


text = fcstsounding.fetch_raw(_args.station, _args.model)
cols, rows = parse_surface(text)
idx = {c: i for i, c in enumerate(cols)}


def col(row, name):
    return float(row[idx[name]])


# Build the derived point-forecast table: dict of param -> list across forecast hours.
valid, table = [], {k: [] for k in
                    ("Temp F", "Temp C", "DP F", "DP C", "RH %", "Wind", "MSLP hPa",
                     "Cloud L/M/H", "Gen Wx", "P01 mm")}
for r in rows:
    valid.append(r[idx["YYMMDD/HHMM"]])
    tc, tdc = col(r, "T2MS"), col(r, "TD2M")
    lcld, mcld, hcld = col(r, "LCLD"), col(r, "MCLD"), col(r, "HCLD")
    wd, ws = wind_dir_spd(col(r, "UWND"), col(r, "VWND"))
    table["Temp F"].append(round(tc * 9 / 5 + 32))
    table["Temp C"].append(round(tc))
    table["DP F"].append(round(tdc * 9 / 5 + 32))
    table["DP C"].append(round(tdc))
    table["RH %"].append(round(magnus_rh(tc, tdc)))
    table["Wind"].append(f"{wd:03d}/{ws}")
    table["MSLP hPa"].append(round(col(r, "PMSL")))
    table["Cloud L/M/H"].append(f"{lcld:.0f}/{mcld:.0f}/{hcld:.0f}")
    table["Gen Wx"].append(gen_wx(max(lcld, mcld, hcld)))
    table["P01 mm"].append(round(col(r, "P01M"), 1))

# Full transposed CSV (params as rows, times as columns) like TarpViewer
out = Path("data/charts/temp") / f"bufkit_point_{_args.model}_{_args.station}.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Valid", *valid])
    for name, vals in table.items():
        w.writerow([name, *vals])

# Console sample: first N hours
n = min(_args.hours, len(valid))
print(f"BUFKIT surface point forecast -- {_args.station.upper()} {_args.model} "
      f"(run {valid[0]}, {len(valid)} hrs; showing {n})\n")
head = f"{'param':<12}" + "".join(f"{v[-4:]:>7}" for v in valid[:n])   # HHMM col headers
print(head)
for name, vals in table.items():
    print(f"{name:<12}" + "".join(f"{str(x):>7}" for x in vals[:n]))
print(f"\nfull table -> {out} ({len(valid)} forecast hours)")
