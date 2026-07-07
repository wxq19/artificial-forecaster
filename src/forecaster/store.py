"""DuckDB persistence — the ONLY file that imports duckdb or writes SQL.

Same seam idea as llm.py: the rest of the app calls these functions and never
sees the database. A MetarObs maps to columns HERE and travels no further. A
METAR line carries only day+time, so the real year/month is attached at persist
time (see insert_obs). All times are UTC.
"""

import json
import re
from datetime import datetime, timezone

import duckdb

from forecaster.config import settings
from forecaster.metar import MetarObs

# A COR modifier marks a corrected report (re-issued with fixed values for the same
# station+time). Detected from the raw so a correction can win the ON CONFLICT race.
_COR = re.compile(r"\bCOR\b")

_OBS_DDL = """
CREATE TABLE IF NOT EXISTS obs (
    station        VARCHAR   NOT NULL,
    obs_time       TIMESTAMP NOT NULL,   -- UTC; year/month + METAR day/time
    report_type    VARCHAR,              -- 'METAR' (routine) or 'SPECI' (weather-forced)
    auto           BOOLEAN,
    cavok          BOOLEAN,
    wind_dir_deg   INTEGER,
    wind_dir_card  VARCHAR,
    wind_speed     INTEGER,
    wind_gust      INTEGER,
    wind_unit      VARCHAR,
    visibility     VARCHAR,              -- reported string (fidelity)
    vis_sm         DOUBLE,               -- numeric statute miles (tools do math)
    vis_m          INTEGER,              -- numeric meters
    vis_flag       VARCHAR,              -- 'M' (<), 'P' (>), or NULL (exact)
    ceiling_ft     INTEGER,              -- lowest BKN/OVC or VV; NULL = unlimited
    vertical_visibility_ft INTEGER,
    temp_c         INTEGER,
    dewpoint_c     INTEGER,
    altimeter_inhg DOUBLE,
    altimeter_hpa  DOUBLE,
    weather        JSON,                 -- list[str] present-weather groups
    clouds         JSON,                 -- list[{cover, height_ft, type}]
    remarks        VARCHAR,
    raw            VARCHAR   NOT NULL,
    source         VARCHAR,              -- data lineage: 'iem' | 'awc' | 'manual'
    corrected      BOOLEAN,              -- a COR report; a correction wins the ON CONFLICT race
    PRIMARY KEY (station, obs_time)      -- natural key -> idempotent re-ingest
);
"""

_INSERT = """
INSERT INTO obs VALUES (
    $station, $obs_time, $report_type, $auto, $cavok,
    $wind_dir_deg, $wind_dir_card, $wind_speed, $wind_gust, $wind_unit,
    $visibility, $vis_sm, $vis_m, $vis_flag,
    $ceiling_ft, $vertical_visibility_ft,
    $temp_c, $dewpoint_c, $altimeter_inhg, $altimeter_hpa,
    $weather, $clouds, $remarks, $raw, $source, $corrected
)
ON CONFLICT (station, obs_time) DO UPDATE SET
    report_type = excluded.report_type, auto = excluded.auto, cavok = excluded.cavok,
    wind_dir_deg = excluded.wind_dir_deg, wind_dir_card = excluded.wind_dir_card,
    wind_speed = excluded.wind_speed, wind_gust = excluded.wind_gust, wind_unit = excluded.wind_unit,
    visibility = excluded.visibility, vis_sm = excluded.vis_sm, vis_m = excluded.vis_m,
    vis_flag = excluded.vis_flag, ceiling_ft = excluded.ceiling_ft,
    vertical_visibility_ft = excluded.vertical_visibility_ft,
    temp_c = excluded.temp_c, dewpoint_c = excluded.dewpoint_c,
    altimeter_inhg = excluded.altimeter_inhg, altimeter_hpa = excluded.altimeter_hpa,
    weather = excluded.weather, clouds = excluded.clouds, remarks = excluded.remarks,
    raw = excluded.raw, source = excluded.source, corrected = excluded.corrected
WHERE excluded.corrected AND NOT COALESCE(obs.corrected, FALSE)
"""
# Policy (#8b): keep-first stays the default (idempotent re-ingest), with ONE new
# behavior -- an incoming COR overwrites a previously-stored non-COR. The WHERE guard
# means a correction always wins regardless of arrival order, and a plain report can
# never downgrade a stored correction.


def connect(
    path: str = settings.db_path, *, read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    """Open the DuckDB file. The agent/tool path passes read_only=True so the
    model can never trigger a write; the ingestion path uses the default."""
    return duckdb.connect(path, read_only=read_only)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create the obs table if absent. Idempotent — safe on every startup. The ADD
    COLUMN migrates a DB created before the `corrected` column existed (#8b)."""
    con.execute(_OBS_DDL)
    con.execute("ALTER TABLE obs ADD COLUMN IF NOT EXISTS corrected BOOLEAN")


def _row(o: MetarObs, year: int, month: int, source: str) -> dict:
    """Map a MetarObs to the obs columns, attaching the real year/month.
    datetime() raises if the day is impossible for that month — a free guard."""
    return {
        "station": o.station,
        "obs_time": datetime(year, month, o.day, o.time.hour, o.time.minute),
        "report_type": o.report_type,
        "auto": o.auto,
        "cavok": o.cavok,
        "wind_dir_deg": o.wind_dir_deg,
        "wind_dir_card": o.wind_dir_card,
        "wind_speed": o.wind_speed,
        "wind_gust": o.wind_gust,
        "wind_unit": o.wind_unit,
        "visibility": o.visibility,
        "vis_sm": o.vis_sm,
        "vis_m": o.vis_m,
        "vis_flag": o.vis_flag,
        "ceiling_ft": o.ceiling_ft,
        "vertical_visibility_ft": o.vertical_visibility_ft,
        "temp_c": o.temp_c,
        "dewpoint_c": o.dewpoint_c,
        "altimeter_inhg": o.altimeter_inhg,
        "altimeter_hpa": o.altimeter_hpa,
        "weather": json.dumps(o.weather),
        "clouds": json.dumps([c.model_dump() for c in o.clouds]),
        "remarks": o.remarks,
        "raw": o.raw,
        "source": source,
        "corrected": bool(_COR.search(o.raw)),
    }


def count(con: duckdb.DuckDBPyConnection, station: str | None = None) -> int:
    """Row count, optionally scoped to one station."""
    if station is not None:
        return con.execute(
            "SELECT count(*) FROM obs WHERE station = ?", [station]
        ).fetchone()[0]
    return con.execute("SELECT count(*) FROM obs").fetchone()[0]


def insert_obs(
    con: duckdb.DuckDBPyConnection,
    obs: list[MetarObs],
    *,
    year: int,
    month: int,
    source: str = "manual",
) -> int:
    """Persist observations, attaching the real year/month (a METAR has only
    day+time). Idempotent: a repeat (station, obs_time) is a no-op. Assumes every
    ob falls in the given year/month — the caller splits batches that cross a
    month boundary. Returns the number of rows actually added."""
    before = count(con)
    for o in obs:
        con.execute(_INSERT, _row(o, year, month, source))
    return count(con) - before


def latest(con: duckdb.DuckDBPyConnection, station: str, limit: int = 1) -> list[dict]:
    """Most-recent obs for a station, newest first — a basic round-trip read. Like
    window(), deserializes the JSON columns so callers get `weather: list[str]` and
    `clouds: list[dict]`, not raw JSON text — the store boundary always hands back
    Python objects."""
    cur = con.execute(
        "SELECT * FROM obs WHERE station = ? ORDER BY obs_time DESC LIMIT ?",
        [station, limit],
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    for row in rows:
        row["weather"] = json.loads(row["weather"]) if row["weather"] else []
        row["clouds"] = json.loads(row["clouds"]) if row["clouds"] else []
    return rows


def _to_naive_utc(dt: datetime) -> datetime:
    """obs_time is stored naive-UTC; coerce any tz-aware bound to naive UTC so a
    'Z'/offset suffix can't shift the window by the session's local offset (DuckDB
    reconciles a tz-aware param against the naive column via local time)."""
    return dt.astimezone(timezone.utc).replace(tzinfo=None) if dt.tzinfo else dt


def window(
    con: duckdb.DuckDBPyConnection,
    station: str,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """Obs for a station within [start, end] (UTC), chronological. The JSON
    columns are deserialized back to Python here — callers (e.g. the agent tool)
    get `weather: list[str]` and `clouds: list[dict]`, not raw JSON text."""
    start, end = _to_naive_utc(start), _to_naive_utc(end)
    cur = con.execute(
        "SELECT * FROM obs WHERE station = ? AND obs_time BETWEEN ? AND ? "
        "ORDER BY obs_time",
        [station, start, end],
    )
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for row in rows:
        row["weather"] = json.loads(row["weather"]) if row["weather"] else []
        row["clouds"] = json.loads(row["clouds"]) if row["clouds"] else []
    return rows
