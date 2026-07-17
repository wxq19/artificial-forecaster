"""Pure self-test for the spatial-awareness internals. No model, no network.

Covers the terrain descriptor logic (terrain._describe), the coastline detector
(terrain._nearest_ocean, which reads the bundled global-land-mask -- offline), and the
NEIGHBOR LEAKAGE GUARD: store.copy_obs must never copy an ob at/after the cutoff, which is
exactly what makes get_nearby_obs leakage-safe. Deterministic + fast, like test_tafgen.py.
"""

import shutil
import tempfile
from pathlib import Path

from forecaster import metar, store, terrain

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


NAZ = len(terrain._AZIMUTHS)
NR = len(terrain._RANGES_KM)


def _uniform(center: float, delta: float) -> list[list[float]]:
    """A grid where every sampled point sits `delta` m above (or below) the center."""
    return [[center + delta for _ in range(NR)] for _ in range(NAZ)]


# --- terrain._describe: landform classification from a synthetic grid ---
valley = terrain._describe(0.0, _uniform(0.0, +200.0))
check("valley landform", valley["landform"] == "valley/basin", valley["landform"])
check("valley all upslope", len(valley["upslope"]) == NAZ and not valley["downslope"])
check("valley relief", valley["relief_m"] == 200)

ridge = terrain._describe(1000.0, _uniform(1000.0, -200.0))
check("ridge landform", ridge["landform"] == "ridge/exposed", ridge["landform"])
check("ridge all downslope", len(ridge["downslope"]) == NAZ and not ridge["upslope"])

flat = terrain._describe(100.0, _uniform(100.0, +10.0))     # < 50 m slope threshold
check("flat landform", flat["landform"] == "flat", flat["landform"])
check("flat no slopes", not flat["upslope"] and not flat["downslope"])

# One azimuth (index 4 = due E, 90 deg) rises; the rest are flat -> 'sloped', E upslope only.
g = _uniform(0.0, 0.0)
for j in range(NR):
    g[4][j] = 300.0
sloped = terrain._describe(0.0, g)
check("sloped landform", sloped["landform"] == "sloped", sloped["landform"])
check("sloped upslope is E", sloped["upslope"] == ["E"], str(sloped["upslope"]))
check("sloped max_rise E", sloped["max_rise"][0] == "E", str(sloped["max_rise"]))

# --- terrain._nearest_ocean: coastal vs inland (global-land-mask, offline) ---
vbg = terrain._nearest_ocean(34.733, -120.583)       # Vandenberg: Pacific just to the west
check("Vandenberg is coastal", vbg is not None, "expected an ocean hit")
check("Vandenberg coast is westerly", vbg is not None and vbg[1] in ("W", "WNW", "WSW", "SW"),
      str(vbg))

tus = terrain._nearest_ocean(32.165, -110.887)       # Tucson: no ocean within 150 km
check("Tucson is inland", tus is None, str(tus))

# --- NEIGHBOR LEAKAGE GUARD: copy_obs enforces the cutoff in SQL ---
pre = metar.parse("METAR KTTN 010600Z 00000KT 10SM CLR 20/10 A3000")   # 06Z
post = metar.parse("METAR KTTN 011800Z 00000KT 10SM CLR 25/12 A2998")  # 18Z
tmp = Path(tempfile.mkdtemp())
bench = str(tmp / "bench.duckdb")
run = str(tmp / "run.duckdb")
bcon = store.connect(bench)
store.init_schema(bcon)
store.insert_obs(bcon, [pre, post], year=2026, month=7, source="test")
banked = sorted(r["obs_time"] for r in store.latest(bcon, "KTTN", 10))  # [06Z, 18Z]
bcon.close()

pre_t, cutoff = banked[0], banked[-1]                 # cutoff = 18Z; strictly-before must drop it
rcon = store.connect(run)
n = store.copy_obs(rcon, bench, "KTTN", before=cutoff, hours=48)
kept = store.latest(rcon, "KTTN", 10)
rcon.close()
check("copy_obs copied only the pre-cutoff ob", n == 1, f"copied {n}")
check("no neighbor ob at/after cutoff", all(r["obs_time"] < cutoff for r in kept),
      str([r["obs_time"] for r in kept]))
check("the pre-cutoff ob survived", any(r["obs_time"] == pre_t for r in kept))

shutil.rmtree(tmp)

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
