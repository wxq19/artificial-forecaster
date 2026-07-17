"""Roster of military aerodromes the benchmark forecasts against.

Every entry passed four gates (confirmed live -- see scripts/probe_bufkit_stations.py,
probe_stations_extended.py, probe_taf_cycles.py):
  1. clearly a MILITARY airfield (AFB / SFB / Army airfield / overseas AB);
  2. model BUFKIT coverage for its OWN ICAO (get_fcst_sounding / get_point_forecast work
     without a proxy);
  3. a fetchable, MILITARY-FORMAT TAF on the AWC public feed -- vis in meters, per-group
     QNH____INS, and TX/TN temperature groups (the discriminator that drops civil/FAA-format
     TAFs, which omit TX/TN and report vis in statute miles);
  4. a 30-hour routine validity.

Dropped in the format pass: the 6-hourly civil-format fields (KCHS/KSPS/KLCK/PAFA -- no
TX/TN, SM vis), the BUFKIT bases with no public AWC TAF (KGSB/KPOB/PAEI/KYUM/RJSM), and
the irregular-reissue Army field KGRK. The survivors are all on an 8-HOURLY (3/day)
military cycle; `cycle` is the tuple of UTC issue hours.

`bufkit_proxy` names a nearby civil ICAO for the model-data tools if a base ever has no
BUFKIT of its own (None -- every roster station is self-covered). Regenerate/extend via
the probe scripts.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Station:
    icao: str
    name: str
    region: str
    meso: str | None                    # densest mesoscale BUFKIT (hrrr/nam); None = GFS-only
    cycle: tuple[int, ...]              # UTC issue hours of the routine TAF (8-hourly)
    taf_hours: int = 30                 # routine TAF validity length
    cycle_provisional: bool = False     # cycle inferred from <3 obs; poller confirms over time
    bufkit_proxy: str | None = None     # civil ICAO for model tools if the base has no BUFKIT


@dataclass(frozen=True)
class ArchiveStation:
    """A human-TAF ARCHIVE-ONLY site: the poller archives its official TAF and the scorer
    TAFVERs it against obs, but it is NEVER run through the model matrix. Deliberately a
    SEPARATE type from Station -- the scheduler iterates STATIONS, so an archive site is
    structurally incapable of entering a billed model run. Archive sites need no cycle /
    BUFKIT / proxy (they issue no model forecast); `regime` tags the dominant forecast-
    difficulty class for per-regime/per-hour TAFVER difficulty mining."""

    icao: str
    name: str
    branch: str                         # "AF" | "Army"
    regime: str                         # convective|fog|winter|terrain|tropical|monsoon


# 10 military-format, 30h, 8-hourly airfields: CONUS + Alaska + Western Pacific.
STATIONS: tuple[Station, ...] = (
    Station("KWRI", "McGuire AFB NJ", "conus_ne", "hrrr", (2, 10, 18)),
    Station("KMIB", "Minot AFB ND", "conus_nplains", "hrrr", (1, 9, 17)),
    Station("KRCA", "Ellsworth AFB SD", "conus_nplains", None, (3, 11, 19)),
    Station("KSSC", "Shaw AFB SC", "conus_se", "hrrr", (7, 15, 23)),
    Station("KDMA", "Davis-Monthan AFB AZ", "conus_sw", None, (3, 11, 19)),
    Station("KVBG", "Vandenberg SFB CA", "conus_w", None, (6, 14, 22), cycle_provisional=True),
    Station("KFTK", "Fort Knox KY (Army)", "conus_mw", "hrrr", (3, 11, 19), cycle_provisional=True),
    Station("PAED", "JB Elmendorf AK", "oconus_ak", "hrrr", (5, 13, 21)),
    Station("PABI", "Ladd AAF, Fort Wainwright AK (Army)", "oconus_ak", "hrrr", (6, 14, 22),
            cycle_provisional=True),
    Station("RJTY", "Yokota AB Japan", "oconus_pac", None, (5, 13, 21)),
)

BY_ICAO: dict[str, Station] = {s.icao: s for s in STATIONS}


def icaos() -> list[str]:
    """All roster ICAOs (the poller iterates these)."""
    return [s.icao for s in STATIONS]


def model_station(icao: str) -> str:
    """The ICAO to feed the model-data tools for a station: its proxy if one is set,
    else the station itself. Raises KeyError for an off-roster ICAO."""
    s = BY_ICAO[icao]
    return s.bufkit_proxy or s.icao


# Human-TAF archive-only sites (53): confirmed military-format on the AWC feed 2026-07-17
# (marker QNH____INS + ~30h validity). The poller archives + the scorer TAFVERs these to map
# forecast difficulty by site and hour; they never enter the billed model matrix. `regime`
# is the dominant difficulty class. NOT gated on BUFKIT (no model run), so much wider than
# STATIONS. Excluded as civil-format (no QNH INS): KABQ, LIPA, ETHA. Issued no TAF in the
# 17Z probe -- re-add + let the poller confirm: KVAD, PAEI, RJSM, PHIK, KGFA, KFSI, KFLV,
# KDAA, KWSD.
ARCHIVE_STATIONS: tuple[ArchiveStation, ...] = (
    # --- Air Force (38) ---
    ArchiveStation("KWRB", "Robins AFB GA", "AF", "convective"),
    ArchiveStation("KCBM", "Columbus AFB MS", "AF", "convective"),
    ArchiveStation("KBIX", "Keesler AFB MS", "AF", "convective"),
    ArchiveStation("KVPS", "Eglin AFB FL", "AF", "convective"),
    ArchiveStation("KHRT", "Hurlburt Field FL", "AF", "convective"),
    ArchiveStation("KPAM", "Tyndall AFB FL", "AF", "convective"),
    ArchiveStation("KMCF", "MacDill AFB FL", "AF", "convective"),
    ArchiveStation("KBAD", "Barksdale AFB LA", "AF", "convective"),
    ArchiveStation("KTIK", "Tinker AFB OK", "AF", "convective"),
    ArchiveStation("KLTS", "Altus AFB OK", "AF", "convective"),
    ArchiveStation("KDYS", "Dyess AFB TX", "AF", "convective"),
    ArchiveStation("KCVS", "Cannon AFB NM", "AF", "convective"),
    ArchiveStation("KSZL", "Whiteman AFB MO", "AF", "convective"),
    ArchiveStation("KOFF", "Offutt AFB NE", "AF", "convective"),
    ArchiveStation("KSKF", "Kelly Field/Lackland TX", "AF", "convective"),
    ArchiveStation("KRND", "Randolph AFB TX", "AF", "convective"),
    ArchiveStation("KDLF", "Laughlin AFB TX", "AF", "convective"),
    ArchiveStation("KLSV", "Nellis AFB NV", "AF", "monsoon"),
    ArchiveStation("KLUF", "Luke AFB AZ", "AF", "monsoon"),
    ArchiveStation("KHMN", "Holloman AFB NM", "AF", "monsoon"),
    ArchiveStation("KBAB", "Beale AFB CA", "AF", "fog"),
    ArchiveStation("KSUU", "Travis AFB CA", "AF", "fog"),
    ArchiveStation("KLFI", "Langley AFB VA", "AF", "fog"),
    ArchiveStation("KDOV", "Dover AFB DE", "AF", "fog"),
    ArchiveStation("KADW", "JB Andrews MD", "AF", "fog"),
    ArchiveStation("EGUN", "RAF Mildenhall UK", "AF", "fog"),
    ArchiveStation("EGUL", "RAF Lakenheath UK", "AF", "fog"),
    ArchiveStation("ETAR", "Ramstein AB Germany", "AF", "fog"),
    ArchiveStation("ETAD", "Spangdahlem AB Germany", "AF", "fog"),
    ArchiveStation("KHIF", "Hill AFB UT", "AF", "winter"),
    ArchiveStation("KFFO", "Wright-Patterson AFB OH", "AF", "winter"),
    ArchiveStation("KBLV", "Scott AFB IL", "AF", "winter"),
    ArchiveStation("KMUO", "Mountain Home AFB ID", "AF", "terrain"),
    ArchiveStation("KEDW", "Edwards AFB CA", "AF", "terrain"),
    ArchiveStation("PGUA", "Andersen AFB Guam", "AF", "tropical"),
    ArchiveStation("RODN", "Kadena AB Japan", "AF", "tropical"),
    ArchiveStation("RKSO", "Osan AB Korea", "AF", "monsoon"),
    ArchiveStation("RKJK", "Kunsan AB Korea", "AF", "monsoon"),
    # --- Army (15) ---
    ArchiveStation("KLSF", "Lawson AAF, Fort Moore GA", "Army", "convective"),
    ArchiveStation("KFBG", "Simmons AAF, Fort Liberty NC", "Army", "convective"),
    ArchiveStation("KOZR", "Cairns AAF, Fort Novosel AL", "Army", "convective"),
    ArchiveStation("KHOP", "Campbell AAF, Fort Campbell KY", "Army", "convective"),
    ArchiveStation("KGRK", "Robert Gray AAF, Fort Cavazos TX", "Army", "convective"),
    ArchiveStation("KFRI", "Marshall AAF, Fort Riley KS", "Army", "convective"),
    ArchiveStation("KFHU", "Libby AAF, Fort Huachuca AZ", "Army", "monsoon"),
    ArchiveStation("KGRF", "Gray AAF, JB Lewis-McChord WA", "Army", "fog"),
    ArchiveStation("KFAF", "Felker AAF, Fort Eustis VA", "Army", "fog"),
    ArchiveStation("KMUI", "Muir AAF, Fort Indiantown Gap PA", "Army", "winter"),
    ArchiveStation("KFCS", "Butts AAF, Fort Carson CO", "Army", "terrain"),
    ArchiveStation("ETIC", "Grafenwoehr AAF Germany", "Army", "terrain"),
    ArchiveStation("ETOU", "Wiesbaden AAF Germany", "Army", "fog"),
    ArchiveStation("PHHI", "Wheeler AAF Hawaii", "Army", "tropical"),
    ArchiveStation("RKSG", "Desiderio AAF, Camp Humphreys Korea", "Army", "monsoon"),
)

ARCHIVE_BY_ICAO: dict[str, ArchiveStation] = {a.icao: a for a in ARCHIVE_STATIONS}


def poll_icaos() -> list[str]:
    """Every ICAO the human-TAF poller archives: the model roster PLUS the archive-only net.
    The scheduler still uses icaos()/STATIONS, so archive-only sites are never billed. Roster
    first, then archive-only; de-duped in case a site is ever on both lists."""
    seen: dict[str, None] = {}
    for i in icaos() + [a.icao for a in ARCHIVE_STATIONS]:
        seen.setdefault(i, None)
    return list(seen)
