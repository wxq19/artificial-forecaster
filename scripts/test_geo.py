"""Self-test for the shared spatial primitive (geo.py). No model, no network.

Checks great-circle distance, initial bearing, 16-point compass rounding, and the
nearest-N ranking + guard against hand-computable cases and a couple of real station
pairs. Pure + deterministic -- a fast correctness gate like test_tafgen.py.
"""

from forecaster.geo import bearing_deg, compass16, destination, haversine_km, nearest_n

PASS, FAIL = 0, 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"PASS  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


# --- distance: one degree of longitude at the equator ~= 111.19 km ---
d = haversine_km(0.0, 0.0, 0.0, 1.0)
check("equator 1deg lon ~111.19km", abs(d - 111.19) < 0.5, f"got {d:.2f}")

# distance is symmetric and zero for identical points
check("zero distance", haversine_km(40.0, -74.0, 40.0, -74.0) == 0.0)
check("symmetric",
      abs(haversine_km(40, -74, 45, -80) - haversine_km(45, -80, 40, -74)) < 1e-9)

# --- bearing: cardinal directions from the origin ---
check("due north bearing=0", abs(bearing_deg(0, 0, 1, 0)) < 1e-6)
check("due east bearing=90", abs(bearing_deg(0, 0, 0, 1) - 90.0) < 1e-6)
check("due south bearing=180", abs(bearing_deg(1, 0, 0, 0) - 180.0) < 1e-6)
check("due west bearing=270", abs(bearing_deg(0, 1, 0, 0) - 270.0) < 1e-6)

# --- compass rounding (16-point) ---
check("0 -> N", compass16(0) == "N")
check("90 -> E", compass16(90) == "E")
check("315 -> NW", compass16(315) == "NW")
check("miss-tie 348.75 -> N", compass16(349) == "N")
check("wrap 360 -> N", compass16(360) == "N")

# --- nearest_n: ranking, guard, and bearing annotation ---
# Home at origin; three candidates E/N/far-E.
catalog = [
    ("EAST_50", 0.0, 0.45),     # ~50 km due east
    ("NORTH_100", 0.9, 0.0),    # ~100 km due north
    ("FAR_EAST", 0.0, 2.0),     # ~222 km east -- beyond a 150 km guard
]
res = nearest_n(0.0, 0.0, catalog, n=5, max_km=150.0)
check("guard drops the far site", len(res) == 2, f"got {[r[0] for r in res]}")
check("nearest first", res[0][0] == "EAST_50" and res[1][0] == "NORTH_100")
check("east bearing label", res[0][2] == "E", f"got {res[0][2]}")
check("north bearing label", res[1][2] == "N", f"got {res[1][2]}")

# n caps the list
check("n caps result", len(nearest_n(0.0, 0.0, catalog, n=1, max_km=150.0)) == 1)

# --- destination is the inverse of haversine + bearing ---
for brg in (0.0, 45.0, 137.0, 270.0, 359.0):
    dlat, dlon = destination(40.0, -74.0, brg, 80.0)
    back_d = haversine_km(40.0, -74.0, dlat, dlon)
    back_b = bearing_deg(40.0, -74.0, dlat, dlon)
    check(f"destination round-trip dist brg={brg}", abs(back_d - 80.0) < 0.5, f"got {back_d:.2f}")
    check(f"destination round-trip bearing brg={brg}",
          min(abs(back_b - brg), 360 - abs(back_b - brg)) < 0.5, f"got {back_b:.2f}")

# --- a real pair: KWRI (McGuire) -> KPHL (Philadelphia), ~58 km (~31 nm) SW ---
kwri = (40.016, -74.591)
kphl = (39.868, -75.241)
dkphl = haversine_km(*kwri, *kphl)
check("KWRI->KPHL ~58km", 50 < dkphl < 65, f"got {dkphl:.1f}")
check("KWRI->KPHL bearing SW-ish",
      compass16(bearing_deg(*kwri, *kphl)) in ("SW", "WSW", "SSW"),
      f"got {compass16(bearing_deg(*kwri, *kphl))}")

print(f"\n{PASS} passed, {FAIL} failed")
raise SystemExit(1 if FAIL else 0)
