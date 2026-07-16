"""Expand the military-airfield roster: probe MORE candidates for BUFKIT, and for every
covered station derive its TAF issuance cycle from the current bulletin (network probe).

Two passes:
  1. BUFKIT gate (GFS) + mesoscale classify (HRRR/NAM) over an EXPANDED candidate pool
     (the 15 already-confirmed are carried in as known-covered; only the ADDITIONS are
     probed for coverage).
  2. For every covered station, fetch the current TAF and read issue time + valid window +
     bulletin type. The 30h AF main TAF is issued every 6h, so the cycle is inferred as the
     four 6-hourly slots at the valid-from phase. A single snapshot can't distinguish a
     routine cycle from an off-cycle amendment, so AMD/COR bulletins are flagged -- the
     live poller confirms each station's real schedule over 24h as new bulletins land.

  uv run python scripts/probe_stations_extended.py
"""

from forecaster import awc, fcstsounding
from forecaster.tafarchive import build_taf_row

# Already confirmed (scripts/probe_bufkit_stations.py -> forecaster.stations): icao, name, region, meso
KNOWN: list[tuple[str, str, str, str | None]] = [
    ("KWRI", "McGuire AFB NJ", "conus_ne", "hrrr"),
    ("KMIB", "Minot AFB ND", "conus_nplains", "hrrr"),
    ("KRCA", "Ellsworth AFB SD", "conus_nplains", None),
    ("KSSC", "Shaw AFB SC", "conus_se", "hrrr"),
    ("KGSB", "Seymour Johnson AFB NC", "conus_se", "hrrr"),
    ("KPOB", "Pope Field NC", "conus_se", "hrrr"),
    ("KCHS", "Charleston AFB SC", "conus_se", "hrrr"),
    ("KDMA", "Davis-Monthan AFB AZ", "conus_sw", None),
    ("KVBG", "Vandenberg SFB CA", "conus_w", None),
    ("KSPS", "Sheppard AFB TX", "conus_s", "hrrr"),
    ("KLCK", "Rickenbacker ANGB OH", "conus_mw", "hrrr"),
    ("KFTK", "Fort Knox KY", "conus_mw", "hrrr"),
    ("PAED", "JB Elmendorf AK", "oconus_ak", "hrrr"),
    ("PAEI", "Eielson AFB AK", "oconus_ak", "hrrr"),
    ("RJTY", "Yokota AB Japan", "oconus_pac", None),
]

# NEW candidates to probe (not in the first pass of 61). Mostly CONUS+AK (mtarchive is
# US-centric); a few OCONUS/Europe included to confirm the coverage edge.
ADDITIONS: list[tuple[str, str, str]] = [
    ("KHMN", "Holloman AFB NM", "conus_sw"),
    ("KCVS", "Cannon AFB NM", "conus_sw"),
    ("KLRF", "Little Rock AFB AR", "conus_s"),
    ("KSKA", "Fairchild AFB WA", "conus_nw"),
    ("KTCM", "McChord Field WA (JBLM)", "conus_nw"),
    ("KEDW", "Edwards AFB CA", "conus_w"),
    ("KDOV", "Dover AFB DE", "conus_e"),
    ("KRND", "Randolph AFB TX", "conus_s"),
    ("KLSF", "Lawson AAF Fort Benning GA", "conus_se"),
    ("KFBG", "Simmons AAF Fort Bragg NC", "conus_se"),
    ("KGRK", "Gray AAF Fort Cavazos TX", "conus_s"),
    ("KBIF", "Biggs AAF Fort Bliss TX", "conus_sw"),
    ("KGTB", "Wheeler-Sack AAF Fort Drum NY", "conus_ne"),
    ("KHUA", "Redstone AAF AL", "conus_se"),
    ("KOZR", "Cairns AAF Fort Rucker AL", "conus_se"),
    ("KFHU", "Libby AAF Fort Huachuca AZ", "conus_sw"),
    ("KTBN", "Forney AAF Fort Leonard Wood MO", "conus_mw"),
    ("KMUI", "Muir AAF Fort Indiantown Gap PA", "conus_ne"),
    ("KDAA", "Davison AAF Fort Belvoir VA", "conus_e"),
    ("KAPG", "Phillips AAF Aberdeen MD", "conus_e"),
    ("KRIV", "March ARB CA", "conus_sw"),
    ("KYUM", "MCAS Yuma AZ", "conus_sw"),
    ("KNZY", "NAS North Island CA", "conus_sw"),
    ("KNLC", "NAS Lemoore CA", "conus_w"),
    ("KNGU", "NS Norfolk VA", "conus_e"),
    ("KNHK", "NAS Patuxent River MD", "conus_e"),
    ("KNYG", "MCAF Quantico VA", "conus_e"),
    ("KNGP", "NAS Corpus Christi TX", "conus_s"),
    ("KNQI", "NAS Kingsville TX", "conus_s"),
    ("KNSE", "NAS Whiting Field FL", "conus_se"),
    ("KNPA", "NAS Pensacola FL", "conus_se"),
    ("KNMM", "NAS Meridian MS", "conus_se"),
    ("KNBG", "NAS JRB New Orleans LA", "conus_s"),
    ("PABI", "Ladd AAF Fort Wainwright AK", "oconus_ak"),
    ("PAFA", "Fairbanks Intl AK", "oconus_ak"),
    ("RKSO", "Osan AB Korea", "oconus_pac"),
    ("RKJK", "Kunsan AB Korea", "oconus_pac"),
    ("RKSG", "Camp Humphreys Korea", "oconus_pac"),
    ("RJSM", "Misawa AB Japan", "oconus_pac"),
    ("RJOI", "MCAS Iwakuni Japan", "oconus_pac"),
    ("EGUN", "RAF Mildenhall UK", "oconus_eur"),
    ("EGUL", "RAF Lakenheath UK", "oconus_eur"),
    ("ETAD", "Spangdahlem AB Germany", "oconus_eur"),
    ("LIPA", "Aviano AB Italy", "oconus_eur"),
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


def _meso(icao: str) -> str | None:
    return "hrrr" if _available(icao, "hrrr") else ("nam" if _available(icao, "nam") else None)


def _taf_cycle(icao: str) -> dict:
    """Fetch the current TAF and derive issue time, valid window, bulletin type, and the
    inferred 6-hourly issuance cycle from the valid-from phase."""
    try:
        tafs = awc.fetch_taf(icao)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    if not tafs:
        return {"error": "no TAF served"}
    issue, raw = tafs[0]
    try:
        row = build_taf_row(raw, issue_ref=issue)
    except Exception as e:  # noqa: BLE001
        return {"error": f"parse: {e}", "issue": issue}
    vf, vt = row["valid_from_utc"], row["valid_to_utc"]
    phase = vf.hour % 6
    cycle = [(phase + 6 * k) % 24 for k in range(4)]
    return {
        "issue": issue, "valid_from": vf, "valid_to": vt,
        "dur_h": round((vt - vf).total_seconds() / 3600),
        "type": row["bulletin_type"],
        "cycle": "/".join(f"{h:02d}Z" for h in cycle),
    }


def main() -> int:
    print(f"Probing {len(ADDITIONS)} NEW candidates for BUFKIT...\n")
    new_covered: list[tuple[str, str, str, str | None]] = []
    for icao, name, region in ADDITIONS:
        if not _available(icao, "gfs"):
            print(f"  [    ] {icao}  {name}")
            continue
        meso = _meso(icao)
        new_covered.append((icao, name, region, meso))
        print(f"  [ OK ] {icao}  {name:<34} gfs{'+' + meso if meso else '-only'}")

    covered = KNOWN + new_covered
    print(f"\nNewly covered: {len(new_covered)}   Total covered: {len(covered)}")
    print("\nFetching TAF cycle for each covered station...\n")

    print(f"{'ICAO':5} {'BUFKIT':10} {'IssueZ':7} {'Valid':12} {'Dur':4} {'Type':11} "
          f"{'Cycle (inferred)':20} Region / Name")
    print("-" * 108)
    rows = []
    for icao, name, region, meso in covered:
        buf = f"gfs+{meso}" if meso else "gfs-only"
        c = _taf_cycle(icao)
        if "error" in c and "issue" not in c:
            line = f"{icao:5} {buf:10} {'--':7} {'(' + c['error'][:34] + ')':<48}"
            print(line + f" {region} / {name}")
            rows.append((icao, name, region, buf, c))
            continue
        issue = c["issue"]
        valid = f"{c['valid_from']:%d%H}/{c['valid_to']:%d%H}"
        print(f"{icao:5} {buf:10} {issue:%H%M} {valid:12} {c['dur_h']:>3}h {c['type']:11} "
              f"{c['cycle']:20} {region} / {name}")
        rows.append((icao, name, region, buf, c))
    print("\nNote: cycle is INFERRED from one current bulletin (6-hourly AF main-TAF "
          "assumption); AMD/COR rows are off-cycle. The poller confirms the real schedule "
          "as new routine bulletins land over 24h.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
