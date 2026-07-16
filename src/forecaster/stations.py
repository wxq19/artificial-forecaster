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
