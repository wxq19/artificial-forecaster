"""DuckDB persistence — the ONLY file that imports duckdb or writes SQL.

Same seam idea as llm.py: the rest of the app calls these functions and never
sees the database. A MetarObs maps to columns HERE and travels no further. A
METAR line carries only day+time, so the real year/month is attached at persist
time (see insert_obs). All times are UTC.
"""

import json
from datetime import datetime

import duckdb

from forecaster.config import settings
from forecaster.metar import MetarObs

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
    $weather, $clouds, $remarks, $raw, $source
)
ON CONFLICT DO NOTHING
"""


def connect(
    path: str = settings.db_path, *, read_only: bool = False
) -> duckdb.DuckDBPyConnection:
    """Open the DuckDB file. The agent/tool path passes read_only=True so the
    model can never trigger a write; the ingestion path uses the default."""
    return duckdb.connect(path, read_only=read_only)


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create the obs table if absent. Idempotent — safe on every startup."""
    con.execute(_OBS_DDL)


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
    """Most-recent obs for a station, newest first — a basic round-trip read."""
    cur = con.execute(
        "SELECT * FROM obs WHERE station = ? ORDER BY obs_time DESC LIMIT ?",
        [station, limit],
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def window(
    con: duckdb.DuckDBPyConnection,
    station: str,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """Obs for a station within [start, end] (UTC), chronological. The JSON
    columns are deserialized back to Python here — callers (e.g. the agent tool)
    get `weather: list[str]` and `clouds: list[dict]`, not raw JSON text."""
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
