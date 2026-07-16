"""Discover which MILITARY airfields have model BUFKIT coverage (network probe).

BUFKIT (ISU mtarchive) is the model-data source behind get_fcst_sounding /
get_point_forecast. A base is directly usable for collection when its own ICAO returns
a GFS BUFKIT file; a base with no BUFKIT needs a nearby civil proxy (e.g. KBAB -> KSMF).
This probes a curated candidate list against GFS (coverage gate) and, for GFS-passers,
HRRR + NAM (mesoscale = "self-covered" vs GFS-only), then prints a ready-to-review table.

Not a unit test -- it hits the network. Run when refreshing the station roster:
  uv run python scripts/probe_bufkit_stations.py
"""

from forecaster import fcstsounding

# (icao, name, region). Military/joint aerodromes across CONUS + OCONUS. Availability is
# the probe's job -- this is only the candidate pool.
CANDIDATES: list[tuple[str, str, str]] = [
    ("KWRI", "McGuire AFB NJ", "conus_ne"),
    ("KMIB", "Minot AFB ND", "conus_nplains"),
    ("KSSC", "Shaw AFB SC", "conus_se"),
    ("KBAB", "Beale AFB CA", "conus_w"),
    ("KLSV", "Nellis AFB NV", "conus_sw"),
    ("KBLV", "Scott AFB IL", "conus_mw"),
    ("KBKF", "Buckley SFB CO", "conus_w"),
    ("KFFO", "Wright-Patterson AFB OH", "conus_mw"),
    ("KOFF", "Offutt AFB NE", "conus_plains"),
    ("KSKF", "Lackland AFB TX", "conus_s"),
    ("KDLF", "Laughlin AFB TX", "conus_s"),
    ("KEND", "Vance AFB OK", "conus_s"),
    ("KLTS", "Altus AFB OK", "conus_s"),
    ("KDYS", "Dyess AFB TX", "conus_s"),
    ("KBAD", "Barksdale AFB LA", "conus_s"),
    ("KVPS", "Eglin AFB FL", "conus_se"),
    ("KVAD", "Moody AFB GA", "conus_se"),
    ("KWRB", "Robins AFB GA", "conus_se"),
    ("KGSB", "Seymour Johnson AFB NC", "conus_se"),
    ("KLFI", "Langley AFB VA", "conus_e"),
    ("KPOB", "Pope Field NC", "conus_se"),
    ("KDMA", "Davis-Monthan AFB AZ", "conus_sw"),
    ("KLUF", "Luke AFB AZ", "conus_sw"),
    ("KTIK", "Tinker AFB OK", "conus_s"),
    ("KVBG", "Vandenberg SFB CA", "conus_w"),
    ("KMUO", "Mountain Home AFB ID", "conus_w"),
    ("KHIF", "Hill AFB UT", "conus_w"),
    ("KRCA", "Ellsworth AFB SD", "conus_nplains"),
    ("KGFA", "Malmstrom AFB MT", "conus_nw"),
    ("KRDR", "Grand Forks AFB ND", "conus_nplains"),
    ("KSUU", "Travis AFB CA", "conus_w"),
    ("KSPS", "Sheppard AFB TX", "conus_s"),
    ("KCOF", "Patrick SFB FL", "conus_se"),
    ("KMCF", "MacDill AFB FL", "conus_se"),
    ("KCHS", "Charleston AFB SC (joint)", "conus_se"),
    ("KADW", "Andrews AFB MD", "conus_e"),
    ("KSZL", "Whiteman AFB MO", "conus_mw"),
    ("KGUS", "Grissom ARB IN", "conus_mw"),
    ("KMTC", "Selfridge ANGB MI", "conus_mw"),
    ("KLCK", "Rickenbacker ANGB OH", "conus_mw"),
    ("KHOP", "Fort Campbell KY (Army)", "conus_mw"),
    ("KFSI", "Fort Sill OK (Army)", "conus_s"),
    ("KFRI", "Fort Riley KS (Army)", "conus_plains"),
    ("KGRF", "Fort Lewis WA (Army)", "conus_nw"),
    ("KHLR", "Fort Hood TX (Army)", "conus_s"),
    ("KFTK", "Fort Knox KY (Army)", "conus_mw"),
    ("KNKX", "MCAS Miramar CA (USMC)", "conus_sw"),
    ("KNTU", "NAS Oceana VA (USN)", "conus_e"),
    ("KNID", "NAWS China Lake CA (USN)", "conus_sw"),
    ("KNFL", "NAS Fallon NV (USN)", "conus_w"),
    ("KNUW", "NAS Whidbey Island WA (USN)", "conus_nw"),
    ("KNJK", "NAF El Centro CA (USN)", "conus_sw"),
    ("KNBC", "MCAS Beaufort SC (USMC)", "conus_se"),
    ("PAED", "JB Elmendorf AK", "oconus_ak"),
    ("PAEI", "Eielson AFB AK", "oconus_ak"),
    ("PHIK", "JB Pearl Harbor-Hickam HI", "oconus_hi"),
    ("PGUA", "Andersen AFB Guam", "oconus_pac"),
    ("RJTY", "Yokota AB Japan", "oconus_pac"),
    ("RODN", "Kadena AB Japan", "oconus_pac"),
    ("ETAR", "Ramstein AB Germany", "oconus_eur"),
    ("OTBH", "Al Udeid AB Qatar", "oconus_swa"),
]


def _available(icao: str, model: str) -> bool:
    try:
        fcstsounding.fetch_raw(icao, model=model)
        return True
    except ValueError:
        return False
    except Exception as e:  # noqa: BLE001 -- a transient network error is not a "no"
        print(f"  ! {icao} {model}: {type(e).__name__}: {e}")
        return False


def main() -> int:
    covered: list[tuple[str, str, str, str]] = []   # icao, name, region, tier
    missing: list[tuple[str, str, str]] = []
    print(f"Probing {len(CANDIDATES)} candidate military airfields against BUFKIT (GFS gate)...\n")
    for icao, name, region in CANDIDATES:
        if not _available(icao, "gfs"):
            missing.append((icao, name, region))
            print(f"  [    ] {icao}  {name}")
            continue
        # GFS-passer: classify mesoscale coverage (HRRR densest, then NAM).
        meso = "hrrr" if _available(icao, "hrrr") else ("nam" if _available(icao, "nam") else None)
        tier = f"gfs+{meso}" if meso else "gfs-only"
        covered.append((icao, name, region, tier))
        print(f"  [ OK ] {icao}  {name:<34} {tier}")

    print(f"\n=== BUFKIT-COVERED: {len(covered)}/{len(CANDIDATES)} ===")
    for icao, name, region, tier in covered:
        print(f"  {icao}  {region:<14} {tier:<10} {name}")
    print(f"\n=== NO BUFKIT (need a civil proxy): {len(missing)} ===")
    for icao, name, region in missing:
        print(f"  {icao}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
