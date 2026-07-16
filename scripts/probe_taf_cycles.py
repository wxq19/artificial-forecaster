"""Confirm each roster station's TAF issuance cycle from the OGIMET archive (network).

A single current bulletin cannot reveal the cycle (its valid-from phase is ambiguous
between a 6-hourly and an 8-hourly schedule). OGIMET keeps a rolling archive, so a ~36h
window shows every ROUTINE issuance -- the real cycle, read off the issue times. AMD/COR
bulletins are off-cycle and excluded from the inference (counted for context).

Polite: a descriptive UA + a multi-second throttle between stations (OGIMET blocks bursts).

  uv run python scripts/probe_taf_cycles.py
"""

import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

UA = "artificial-forecaster research (wquinten@proton.me)"
THROTTLE_S = 6.0
WINDOW_H = 36

# The confirmed-usable roster (BUFKIT + fetchable AWC TAF).
STATIONS = ["KWRI", "KMIB", "KRCA", "KSSC", "KCHS", "KDMA", "KVBG", "KSPS",
            "KLCK", "KFTK", "KGRK", "PAED", "PABI", "PAFA", "RJTY"]

# TAF ICAO DDHHMMZ ... , with an optional AMD/COR modifier.
_TAF = re.compile(r"\bTAF (AMD |COR )?([A-Z]{4}) (\d{2})(\d{2})(\d{2})Z")


def _url(icao: str, start: datetime, end: datetime) -> str:
    q = {
        "lang": "en", "lugar": icao.lower(), "tipo": "ALL", "ord": "REV", "nil": "SI",
        "fmt": "html", "ano": start.year, "mes": f"{start.month:02d}", "day": f"{start.day:02d}",
        "hora": f"{start.hour:02d}", "anof": end.year, "mesf": f"{end.month:02d}",
        "dayf": f"{end.day:02d}", "horaf": f"{end.hour:02d}", "minf": f"{end.minute:02d}",
        "send": "send",
    }
    return "https://www.ogimet.com/display_metars2.php?" + urllib.parse.urlencode(q)


def _fetch(icao: str, start: datetime, end: datetime) -> str:
    req = urllib.request.Request(_url(icao, start, end), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", "replace")


def _cycle(routine_hours: list[int]) -> str:
    """Infer the interval from the most common gap between consecutive distinct issue
    hours, then render the cycle. Robust to a missed/extra issuance in the window."""
    hrs = sorted(set(routine_hours))
    if len(hrs) < 2:
        return f"insufficient ({hrs})"
    gaps = [(b - a) for a, b in zip(hrs, hrs[1:])] + [hrs[0] + 24 - hrs[-1]]
    interval = min(g for g in gaps if g > 0)                 # smallest true spacing
    phase = hrs[0] % interval
    slots = [(phase + interval * k) % 24 for k in range(24 // interval)]
    return f"{interval}h: " + "/".join(f"{h:02d}Z" for h in sorted(slots))


def main() -> int:
    end = datetime.now(timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
    start = end - timedelta(hours=WINDOW_H)
    print(f"OGIMET TAF-cycle confirmation, window {start:%Y-%m-%d %HZ} .. {end:%Y-%m-%d %H:%MZ}\n")
    print(f"{'ICAO':5} {'routine issue hours (Z)':32} {'AMD/COR':8} {'confirmed cycle':22}")
    print("-" * 74)
    for icao in STATIONS:
        try:
            html = _fetch(icao, start, end)
        except Exception as e:  # noqa: BLE001 -- a fetch failure is reported, not fatal to the pass
            print(f"{icao:5} ERROR {type(e).__name__}: {e}")
            time.sleep(THROTTLE_S)
            continue
        routine: list[int] = []
        n_mod = 0
        seen: set[str] = set()
        for mod, ic, dd, hh, mm in _TAF.findall(html):
            if ic != icao:
                continue
            key = mod + dd + hh + mm
            if key in seen:                      # OGIMET can repeat a bulletin; dedupe
                continue
            seen.add(key)
            if mod.strip():
                n_mod += 1
            else:
                routine.append(int(hh))
        hours_str = " ".join(f"{h:02d}" for h in sorted(set(routine))) or "(none)"
        print(f"{icao:5} {hours_str:32} {n_mod:<8} {_cycle(routine)}")
        time.sleep(THROTTLE_S)
    print("\nCycle = smallest spacing between distinct routine issue hours over the window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
