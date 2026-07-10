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


# ---------------------------------------------------------------------------
# Climatology product tables (climo_*). Separate from obs on purpose: the raw
# multi-year history used to BUILD climo is thrown away (see climo.py); only these
# final product rows persist, so the runtime read tools never anchor on stale
# history. All climo SQL lives HERE, same seam rule as obs.
# ---------------------------------------------------------------------------

_CLIMO_META_DDL = """
CREATE TABLE IF NOT EXISTS climo_meta (
    station               VARCHAR   NOT NULL,
    lat                   DOUBLE,
    lon                   DOUBLE,
    tzname                VARCHAR,
    utc_offset_hours_std  DOUBLE,               -- fixed standard-time offset; NULL = unknown
    source                VARCHAR,              -- metadata provenance (iem-api / latlon-fallback / lon15-fallback)
    computed_at           TIMESTAMP,
    PRIMARY KEY (station)
);
"""

_CLIMO_MONTHLY_DDL = """
CREATE TABLE IF NOT EXISTS climo_monthly (
    station         VARCHAR   NOT NULL,
    month           INTEGER   NOT NULL,         -- 1-12
    por_start_year  INTEGER,
    por_end_year    INTEGER,
    n_years_used    INTEGER,
    n_days          INTEGER,                    -- distinct local days with >=1 ob (TX/TN denominator)
    n_obs_routine   INTEGER,                    -- frequency denominator
    n_obs_all       INTEGER,                    -- temperature/extreme denominator
    tx_mean DOUBLE, tx_p10 DOUBLE, tx_p50 DOUBLE, tx_p90 DOUBLE,
    tx_record INTEGER, tx_record_date DATE,
    tn_mean DOUBLE, tn_p10 DOUBLE, tn_p50 DOUBLE, tn_p90 DOUBLE,
    tn_record INTEGER, tn_record_date DATE,
    pct_ts DOUBLE, pct_fog DOUBLE, pct_fzprecip DOUBLE, pct_sn DOUBLE, pct_ra DOUBLE,
    alt_mean DOUBLE, alt_min DOUBLE, alt_max DOUBLE,
    PRIMARY KEY (station, month)
);
"""

_CLIMO_HOURLY_DDL = """
CREATE TABLE IF NOT EXISTS climo_hourly (
    station     VARCHAR   NOT NULL,
    month       INTEGER   NOT NULL,             -- 1-12 (LOCAL month membership)
    hour_utc    INTEGER   NOT NULL,             -- 0-23 (UTC hour key)
    n_obs           INTEGER,                    -- routine obs at this (month, hour)
    temp_mean_c DOUBLE,
    wind_mean_kt DOUBLE, wind_p90_kt DOUBLE,
    calm_pct DOUBLE, gust_pct DOUBLE, gust_p90_kt DOUBLE,
    dir_mode_sector VARCHAR, dir_mode_pct DOUBLE,
    pct_cig_lt_3000 DOUBLE, pct_cig_lt_1500 DOUBLE, pct_cig_lt_1000 DOUBLE,
    pct_cig_lt_500 DOUBLE, pct_cig_lt_200 DOUBLE,
    pct_vis_lt_5 DOUBLE, pct_vis_lt_3 DOUBLE, pct_vis_lt_2 DOUBLE,
    pct_vis_lt_1 DOUBLE, pct_vis_lt_half DOUBLE,
    pct_ts DOUBLE, pct_fog DOUBLE,
    PRIMARY KEY (station, month, hour_utc)
);
"""


def init_climo_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create the climo_* product tables if absent. Idempotent. SEPARATE from
    init_schema: schema creation is a BUILD-path side effect only (climo.build calls
    both). connect() never creates tables, so the read-only tool path can't run this;
    it reads the tables and treats a missing/empty one as feedback (see tools.py)."""
    con.execute(_CLIMO_META_DDL)
    con.execute(_CLIMO_MONTHLY_DDL)
    con.execute(_CLIMO_HOURLY_DDL)


# The aggregation. `obs_local = obs_time + offset` gives the LOCAL timestamp; month
# membership is filtered on the LOCAL month (NOT a plain UTC-month predicate) so the
# +/-1 day ingest buffer correctly reclassifies edge obs. The UTC hour is the diurnal
# key. Temps (daily TX/TN + records) use ALL obs (SPECIs aren't temperature-triggered);
# frequencies + wind/gust use ROUTINE only (SPECIs are weather/gust-forced). Validated
# against DuckDB 1.5.4.
_CLIMO_MONTHLY_SELECT = """
WITH base AS (
  SELECT report_type, temp_c, altimeter_inhg, weather,
         obs_time + ($off * INTERVAL 1 HOUR) AS obs_local
  FROM src.obs WHERE station = $station
),
scoped AS (
  SELECT *, CAST(extract('month' FROM obs_local) AS INT) AS mon,
            CAST(extract('year'  FROM obs_local) AS INT) AS yr,
            CAST(obs_local AS DATE) AS local_day
  FROM base
  WHERE CAST(extract('month' FROM obs_local) AS INT) IN ({months})
),
daily AS (
  SELECT mon, local_day, max(temp_c) tx, min(temp_c) tn
  FROM scoped WHERE temp_c IS NOT NULL GROUP BY mon, local_day
),
tstat AS (
  SELECT mon,
    avg(tx) tx_mean, quantile_cont(tx,0.1) tx_p10, quantile_cont(tx,0.5) tx_p50, quantile_cont(tx,0.9) tx_p90,
    avg(tn) tn_mean, quantile_cont(tn,0.1) tn_p10, quantile_cont(tn,0.5) tn_p50, quantile_cont(tn,0.9) tn_p90,
    count(*) n_days
  FROM daily GROUP BY mon
),
rstat AS (
  SELECT mon, max(temp_c) tx_record,
              (list(local_day ORDER BY temp_c DESC, local_day ASC))[1] tx_record_date,
              min(temp_c) tn_record,
              (list(local_day ORDER BY temp_c ASC, local_day ASC))[1] tn_record_date
  FROM scoped WHERE temp_c IS NOT NULL GROUP BY mon
),
fstat AS (
  SELECT mon,
    100.0*avg(CASE WHEN regexp_matches(weather,'TS') THEN 1 ELSE 0 END) pct_ts,
    100.0*avg(CASE WHEN regexp_matches(weather,'FG|BR') THEN 1 ELSE 0 END) pct_fog,
    100.0*avg(CASE WHEN regexp_matches(weather,'FZRA|FZDZ') THEN 1 ELSE 0 END) pct_fzprecip,
    100.0*avg(CASE WHEN regexp_matches(weather,'SN|SG') THEN 1 ELSE 0 END) pct_sn,
    100.0*avg(CASE WHEN regexp_matches(weather,'RA') THEN 1 ELSE 0 END) pct_ra
  FROM scoped WHERE report_type = 'METAR' GROUP BY mon
),
cstat AS (
  SELECT mon, count(*) n_obs_all,
    count(*) FILTER (WHERE report_type='METAR') n_obs_routine,
    min(yr) por_start_year, max(yr) por_end_year, count(DISTINCT yr) n_years_used,
    avg(altimeter_inhg) alt_mean, min(altimeter_inhg) alt_min, max(altimeter_inhg) alt_max
  FROM scoped GROUP BY mon
)
SELECT $station AS station, cstat.mon AS month, por_start_year, por_end_year, n_years_used,
  n_days, n_obs_routine, n_obs_all,
  tx_mean, tx_p10, tx_p50, tx_p90, tx_record, tx_record_date,
  tn_mean, tn_p10, tn_p50, tn_p90, tn_record, tn_record_date,
  pct_ts, pct_fog, pct_fzprecip, pct_sn, pct_ra,
  alt_mean, alt_min, alt_max
FROM cstat JOIN tstat USING (mon) JOIN rstat USING (mon) JOIN fstat USING (mon)
"""

_CLIMO_HOURLY_SELECT = """
WITH base AS (
  SELECT report_type, temp_c, wind_speed, wind_gust, wind_dir_deg, ceiling_ft, vis_sm, weather,
         extract('hour' FROM obs_time) AS hour_utc,
         obs_time + ($off * INTERVAL 1 HOUR) AS obs_local
  FROM src.obs WHERE station = $station
),
scoped AS (
  SELECT *, CAST(extract('month' FROM obs_local) AS INT) AS mon,
    CASE WHEN wind_dir_deg IS NULL THEN NULL ELSE
      ['N','NNE','NE','ENE','E','ESE','SE','SSE','S','SSW','SW','WSW','W','WNW','NW','NNW'][
        (CAST(floor((wind_dir_deg % 360)/22.5 + 0.5) AS INT) % 16) + 1 ] END AS sec
  FROM base
  WHERE CAST(extract('month' FROM obs_local) AS INT) IN ({months})
),
routine AS (SELECT * FROM scoped WHERE report_type='METAR'),
msec AS (SELECT mon, hour_utc, mode(sec) dmode, count(sec) nsec FROM routine GROUP BY mon, hour_utc),
dom AS (
  SELECT r.mon, r.hour_utc, m.dmode dir_mode_sector,
         round(100.0*sum(CASE WHEN r.sec = m.dmode THEN 1 ELSE 0 END)/nullif(m.nsec,0),0) dir_mode_pct
  FROM routine r JOIN msec m USING (mon, hour_utc) GROUP BY r.mon, r.hour_utc, m.dmode, m.nsec
),
agg AS (
  SELECT mon, hour_utc,
    count(*) n_obs,
    avg(temp_c) temp_mean_c,
    avg(wind_speed) wind_mean_kt, quantile_cont(wind_speed,0.9) wind_p90_kt,
    100.0*avg(CASE WHEN wind_speed = 0 THEN 1 ELSE 0 END) calm_pct,
    100.0*avg(CASE WHEN wind_gust IS NOT NULL THEN 1 ELSE 0 END) gust_pct,
    quantile_cont(wind_gust,0.9) gust_p90_kt,
    100.0*avg(CASE WHEN ceiling_ft < 3000 THEN 1 ELSE 0 END) pct_cig_lt_3000,
    100.0*avg(CASE WHEN ceiling_ft < 1500 THEN 1 ELSE 0 END) pct_cig_lt_1500,
    100.0*avg(CASE WHEN ceiling_ft < 1000 THEN 1 ELSE 0 END) pct_cig_lt_1000,
    100.0*avg(CASE WHEN ceiling_ft < 500 THEN 1 ELSE 0 END) pct_cig_lt_500,
    100.0*avg(CASE WHEN ceiling_ft < 200 THEN 1 ELSE 0 END) pct_cig_lt_200,
    100.0*avg(CASE WHEN vis_sm < 5 THEN 1 ELSE 0 END) pct_vis_lt_5,
    100.0*avg(CASE WHEN vis_sm < 3 THEN 1 ELSE 0 END) pct_vis_lt_3,
    100.0*avg(CASE WHEN vis_sm < 2 THEN 1 ELSE 0 END) pct_vis_lt_2,
    100.0*avg(CASE WHEN vis_sm < 1 THEN 1 ELSE 0 END) pct_vis_lt_1,
    100.0*avg(CASE WHEN vis_sm < 0.5 THEN 1 ELSE 0 END) pct_vis_lt_half,
    100.0*avg(CASE WHEN regexp_matches(weather,'TS') THEN 1 ELSE 0 END) pct_ts,
    100.0*avg(CASE WHEN regexp_matches(weather,'FG|BR') THEN 1 ELSE 0 END) pct_fog
  FROM routine GROUP BY mon, hour_utc
)
SELECT $station AS station, agg.mon AS month, agg.hour_utc, n_obs, temp_mean_c,
  wind_mean_kt, wind_p90_kt, calm_pct, gust_pct, gust_p90_kt,
  dir_mode_sector, dir_mode_pct,
  pct_cig_lt_3000, pct_cig_lt_1500, pct_cig_lt_1000, pct_cig_lt_500, pct_cig_lt_200,
  pct_vis_lt_5, pct_vis_lt_3, pct_vis_lt_2, pct_vis_lt_1, pct_vis_lt_half,
  pct_ts, pct_fog
FROM agg LEFT JOIN dom USING (mon, hour_utc)
"""


def rebuild_climo(
    con: duckdb.DuckDBPyConnection,
    scratch_db_path: str,
    station: str,
    months: list[int],
    *,
    utc_offset_hours: float | None,
    lat: float | None,
    lon: float | None,
    tzname: str | None,
    source: str,
) -> dict:
    """Materialize climo_* rows for one station/months from the raw obs in a scratch
    DuckDB (ATTACHed read-only). ONE transaction: drop this station's rows for the
    requested months, upsert climo_meta, INSERT..SELECT the monthly + hourly aggregates.
    Idempotent per-month drop-and-rebuild; other months/stations are untouched. The
    scratch DB holds the throwaway history; only these product rows persist here.
    Requires init_climo_schema(con) first. Returns a per-month summary."""
    if not months:
        raise ValueError("rebuild_climo needs at least one month")
    months_in = ",".join(str(int(m)) for m in months)   # validated ints; safe to inline
    offset = 0.0 if utc_offset_hours is None else utc_offset_hours
    monthly_sql = _CLIMO_MONTHLY_SELECT.format(months=months_in)
    hourly_sql = _CLIMO_HOURLY_SELECT.format(months=months_in)
    params = {"off": offset, "station": station}

    con.execute(f"ATTACH '{scratch_db_path}' AS src (READ_ONLY)")
    try:
        con.execute("BEGIN")
        con.execute(
            f"DELETE FROM climo_monthly WHERE station = ? AND month IN ({months_in})", [station]
        )
        con.execute(
            f"DELETE FROM climo_hourly WHERE station = ? AND month IN ({months_in})", [station]
        )
        con.execute(
            "DELETE FROM climo_meta WHERE station = ?", [station]
        )
        con.execute(
            "INSERT INTO climo_meta VALUES (?, ?, ?, ?, ?, ?, now())",
            [station, lat, lon, tzname, utc_offset_hours, source],
        )
        con.execute(f"INSERT INTO climo_monthly {monthly_sql}", params)
        con.execute(f"INSERT INTO climo_hourly {hourly_sql}", params)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.execute("DETACH src")

    written = con.execute(
        f"SELECT month, n_obs_routine, n_obs_all, n_days, por_start_year, por_end_year, "
        f"n_years_used FROM climo_monthly WHERE station = ? AND month IN ({months_in}) "
        f"ORDER BY month",
        [station],
    ).fetchall()
    cols = ["month", "n_obs_routine", "n_obs_all", "n_days",
            "por_start_year", "por_end_year", "n_years_used"]
    return {"station": station, "months": [dict(zip(cols, r)) for r in written]}


def climo_meta(con: duckdb.DuckDBPyConnection, station: str) -> dict | None:
    """The station's climo metadata row, or None if not built."""
    cur = con.execute("SELECT * FROM climo_meta WHERE station = ?", [station])
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    return dict(zip(cols, row)) if row else None


def climo_month(con: duckdb.DuckDBPyConnection, station: str, month: int) -> dict | None:
    """The station's monthly climo row for one month, or None if that month isn't built."""
    cur = con.execute(
        "SELECT * FROM climo_monthly WHERE station = ? AND month = ?", [station, month]
    )
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    return dict(zip(cols, row)) if row else None


def climo_hours(con: duckdb.DuckDBPyConnection, station: str, month: int) -> list[dict]:
    """The 24 hourly climo rows for one station-month, in hour_utc order (empty if
    not built)."""
    cur = con.execute(
        "SELECT * FROM climo_hourly WHERE station = ? AND month = ? ORDER BY hour_utc",
        [station, month],
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]
