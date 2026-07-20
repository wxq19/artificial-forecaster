"""DuckDB persistence — the ONLY file that imports duckdb or writes SQL.

Same seam idea as llm.py: the rest of the app calls these functions and never
sees the database. A MetarObs maps to columns HERE and travels no further. A
METAR line carries only day+time, so the real year/month is attached at persist
time (see insert_obs). All times are UTC.
"""

import contextlib
import fcntl
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb

from forecaster.config import settings
from forecaster.metar import MetarObs


@contextlib.contextmanager
def write_lock(db_path: str | None = None):
    """Exclusive advisory file lock guarding SINGLE-WRITER access to the benchmark DB.
    The TAF poller (archive), the collector (persist_run), and score_taf --pending all
    take it, so their writes to the one .duckdb never overlap on the Pi. Blocks until
    acquired (DuckDB itself allows only one writer; this serializes the processes cleanly
    instead of letting one fail to open). Lock file sits beside the DB. Read-only tool
    connections do NOT need it."""
    path = Path(db_path or settings.db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / (path.name + ".lock")
    with open(lock_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

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


def copy_climo(con: duckdb.DuckDBPyConnection, src_db_path: str, station: str) -> int:
    """Copy a station's climo_* PRODUCT rows from another benchmark DB into `con` (e.g. a
    per-run collection DB) so get_climo works there without a rebuild. Leakage-safe: climo
    is a historical product (end_year = last complete year). Creates the climo schema on
    `con`, ATTACHes the source read-only, and copies meta/monthly/hourly for the station.
    Returns the monthly rows copied (0 if the source has no climo for the station yet).
    Hold write_lock on src if the poller may be writing it."""
    init_climo_schema(con)
    con.execute(f"ATTACH '{src_db_path}' AS climosrc (READ_ONLY)")
    try:
        for tbl in ("climo_meta", "climo_monthly", "climo_hourly"):
            con.execute(f"INSERT INTO {tbl} SELECT * FROM climosrc.{tbl} WHERE station = ?",
                        [station])
        return con.execute(
            "SELECT count(*) FROM climo_monthly WHERE station = ?", [station]).fetchone()[0]
    except duckdb.CatalogException:
        # ONLY "source has no climo tables yet" reads as not-built -> 0. Anything else
        # (missing/locked/corrupt source) must raise: swallowing it made a real failure
        # indistinguishable from an unbuilt climo.
        return 0
    finally:
        con.execute("DETACH climosrc")


def copy_obs(con: duckdb.DuckDBPyConnection, src_db_path: str, station: str, *,
             before: datetime, hours: float) -> int:
    """Copy a station's obs rows from the benchmark DB into `con` (a per-run collection
    DB), STRICTLY before the `before` cutoff and no older than `hours` back from it --
    the leakage guard enforced in SQL, so the per-run DB can never hold an ob at or
    after issue time. One fetch feeds both DBs: the collector banks truth obs into the
    benchmark DB (no cutoff -- truth wants everything), then copies the pre-cutoff
    back-window here for the model's read tools. Returns rows copied. The caller holds
    write_lock on the source."""
    init_schema(con)
    cutoff = _to_naive_utc(before)
    start = cutoff - timedelta(hours=hours)
    con.execute(f"ATTACH '{src_db_path}' AS obssrc (READ_ONLY)")
    try:
        before_n = con.execute("SELECT count(*) FROM obs").fetchone()[0]
        con.execute(
            "INSERT INTO obs SELECT * FROM obssrc.obs "
            "WHERE station = ? AND obs_time < ? AND obs_time >= ? "
            "ON CONFLICT (station, obs_time) DO NOTHING",
            [station, cutoff, start])
        return con.execute("SELECT count(*) FROM obs").fetchone()[0] - before_n
    finally:
        con.execute("DETACH obssrc")


# ---------------------------------------------------------------------------
# Model-data archive (model_data). Point forecast time series pulled from GRIBStream
# (gribstream.py) for a TAF site + its neighbors + a coarse upstream grid, ONE tall
# float table indexed by coordinate so per-location read tools serve the agent for free.
# Distinct from obs on the leakage axis: a model FORECAST issued before the TAF issue
# time was legitimately available (the human had it too), so the only guard is
# run <= issue_time, enforced at PREFETCH via asOf (modeldata.py) -- the archive needs no
# valid_time read-cutoff. Same seam rule as obs/climo: all model_data SQL lives HERE.
# ---------------------------------------------------------------------------

# lat/lon are stored ROUNDED so an archive read matches a requested point by equality
# (the coordinate list is deterministic geometry, not free text, so rounding is lossless
# for our purposes). Keep this in sync between insert and every read filter.
_LL_DP = 4


def _round_ll(v: float) -> float:
    return round(float(v), _LL_DP)


_MODEL_DATA_DDL = """
CREATE TABLE IF NOT EXISTS model_data (
    model       VARCHAR   NOT NULL,   -- gfs|hrrr|nbm|ifsoper
    run         TIMESTAMP NOT NULL,   -- forecasted_at (naive UTC) = the model cycle
    valid_time  TIMESTAMP NOT NULL,   -- forecasted_time (naive UTC)
    lat         DOUBLE    NOT NULL,   -- rounded to _LL_DP so reads match by equality
    lon         DOUBLE    NOT NULL,
    loc_id      VARCHAR,              -- coordinate name (neighbor ICAO or grid id)
    variable    VARCHAR   NOT NULL,   -- alias (t2m, td2m, rh500, ...)
    value       DOUBLE,               -- native units; NULL if masked
    member      INTEGER   NOT NULL DEFAULT 0,   -- ensemble id (0 = deterministic)
    as_of       TIMESTAMP,            -- the asOf cutoff used at fetch (leakage provenance)
    fetched_at  TIMESTAMP,            -- wall-clock of the pull
    PRIMARY KEY (model, run, valid_time, lat, lon, variable, member)
);
"""


def init_model_data_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create the model_data table if absent. Idempotent. Write-path side effect (the
    read-only tool conn never runs it); the prefetch orchestrator calls it before insert."""
    con.execute(_MODEL_DATA_DDL)


def insert_model_data(con: duckdb.DuckDBPyConnection, rows: list[dict]) -> int:
    """Bulk-insert model_data rows (one per model x run x valid_time x coord x variable x
    member). IMMUTABLE archive: a repeat PK is a no-op (ON CONFLICT DO NOTHING) -- a model
    run's value for a valid time never changes, so re-ingest is idempotent like copy_obs.
    Rounds lat/lon on the way in. Each row dict carries: model, run, valid_time, lat, lon,
    loc_id, variable, value, member (default 0), as_of, fetched_at. Returns rows added."""
    if not rows:
        return 0
    before = con.execute("SELECT count(*) FROM model_data").fetchone()[0]
    # The long-format archive is model x run x valid_time x coord x variable, so a cycle carries
    # tens of thousands of rows. A per-row (or even executemany) INSERT ... ON CONFLICT runs one
    # statement per row -- ~13 ms each, minutes total, CPU-bound on the Pi. DuckDB is columnar:
    # register the batch as ONE relation and do a single set-based INSERT ... SELECT (measured
    # ~1000x faster). Fall back to executemany if pandas is unavailable (correctness unchanged).
    cols = ("model", "run", "valid_time", "lat", "lon", "loc_id",
            "variable", "value", "member", "as_of", "fetched_at")
    tuples = [
        (
            r["model"],
            _to_naive_utc(r["run"]) if r.get("run") else None,
            _to_naive_utc(r["valid_time"]),
            _round_ll(r["lat"]),
            _round_ll(r["lon"]),
            r.get("loc_id"),
            r["variable"],
            r.get("value"),
            int(r.get("member") or 0),
            _to_naive_utc(r["as_of"]) if r.get("as_of") else None,
            _to_naive_utc(r["fetched_at"]) if r.get("fetched_at") else None,
        )
        for r in rows
    ]
    try:
        import pandas as pd

        con.register("_md_incoming", pd.DataFrame(tuples, columns=cols))
        try:
            con.execute("INSERT INTO model_data SELECT * FROM _md_incoming ON CONFLICT DO NOTHING")
        finally:
            con.unregister("_md_incoming")
    except ImportError:
        con.executemany(
            "INSERT INTO model_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
            tuples,
        )
    return con.execute("SELECT count(*) FROM model_data").fetchone()[0] - before


def model_data_series(
    con: duckdb.DuckDBPyConnection,
    model: str,
    lat: float,
    lon: float,
    *,
    start: datetime,
    end: datetime,
    variables: list[str] | None = None,
) -> list[dict]:
    """One location's model rows for a model over [start, end] (valid_time inclusive),
    ordered by valid_time then variable. `variables` optionally restricts the alias set.
    Matches the point by ROUNDED lat/lon equality (see _round_ll). Returns tall dict rows
    (model, run, valid_time, lat, lon, loc_id, variable, value, member); the caller pivots
    variable->value per valid time (preferring the latest run if several are present)."""
    sql = ("SELECT * FROM model_data WHERE model = ? AND lat = ? AND lon = ? "
           "AND valid_time BETWEEN ? AND ?")
    params: list = [model, _round_ll(lat), _round_ll(lon),
                    _to_naive_utc(start), _to_naive_utc(end)]
    if variables:
        sql += f" AND variable IN ({','.join('?' * len(variables))})"
        params += list(variables)
    sql += " ORDER BY valid_time, variable"
    cur = con.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def model_data_locations(
    con: duckdb.DuckDBPyConnection, *, run: datetime | None = None
) -> list[dict]:
    """Distinct pre-fetched locations in the archive: (loc_id, lat, lon, models, n_rows),
    nearest-name-first is not meaningful so ordered by loc_id. A tool advertises these so
    the model knows which points it may query. Optional `run` restricts to one cycle. In a
    per-run collection DB this returns exactly the copied station neighborhood."""
    sql = ("SELECT loc_id, lat, lon, string_agg(DISTINCT model, ',') AS models, "
           "count(*) AS n_rows FROM model_data")
    params: list = []
    if run is not None:
        sql += " WHERE run = ?"
        params.append(_to_naive_utc(run))
    sql += " GROUP BY loc_id, lat, lon ORDER BY loc_id"
    cur = con.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def model_data_field(
    con: duckdb.DuckDBPyConnection,
    model: str,
    variable: str,
    *,
    valid_time: datetime,
    run: datetime | None = None,
) -> list[dict]:
    """One variable's value at ONE valid time across ALL pre-fetched locations -- the
    spatial slice a gradient/advection tool reads. Returns [{loc_id, lat, lon, value, run}]
    ordered by loc_id, keeping the LATEST run per location if several are present."""
    sql = ("SELECT loc_id, lat, lon, value, run FROM model_data "
           "WHERE model = ? AND variable = ? AND valid_time = ?")
    params: list = [model, variable, _to_naive_utc(valid_time)]
    if run is not None:
        sql += " AND run = ?"
        params.append(_to_naive_utc(run))
    sql += " ORDER BY loc_id, run DESC"
    cur = con.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    seen: set = set()
    out: list[dict] = []
    for r in rows:  # rows already run-DESC; first per loc_id is the latest run
        if r["loc_id"] in seen:
            continue
        seen.add(r["loc_id"])
        out.append(r)
    return out


def model_data_valid_times(
    con: duckdb.DuckDBPyConnection, model: str, lat: float, lon: float
) -> list[datetime]:
    """Distinct valid times stored for a model at a point (rounded), ascending. Lets a tool
    default a window/valid-time to what the archive actually holds."""
    cur = con.execute(
        "SELECT DISTINCT valid_time FROM model_data WHERE model = ? AND lat = ? AND lon = ? "
        "ORDER BY valid_time",
        [model, _round_ll(lat), _round_ll(lon)],
    )
    return [r[0] for r in cur.fetchall()]


def copy_model_data(
    con: duckdb.DuckDBPyConnection,
    src_db_path: str,
    *,
    coords: list[tuple[float, float, str]] | None = None,
) -> int:
    """Copy model_data rows from the benchmark DB into `con` (a per-run collection DB) so
    the model-data tools work there for 0 credits -- mirrors copy_obs/copy_climo. Leakage-
    safe by construction: the archive was pre-fetched with asOf = issue_time, so every
    row's run <= issue_time already (no read-cutoff needed here, unlike copy_obs).

    `coords` is the station's coordinate set (site + neighbors + grid, the SAME list the
    prefetch used) as (lat, lon, name); rows are filtered to those ROUNDED points so only
    the relevant neighborhood is copied. If None, copies ALL model_data (small scratch DBs
    only). Returns rows copied. The caller holds write_lock on the source if a writer runs."""
    init_model_data_schema(con)
    con.execute(f"ATTACH '{src_db_path}' AS mdsrc (READ_ONLY)")
    try:
        before = con.execute("SELECT count(*) FROM model_data").fetchone()[0]
        if coords:
            pairs = {(_round_ll(la), _round_ll(lo)) for la, lo, _ in coords}
            values = ",".join(f"({la},{lo})" for la, lo in pairs)
            con.execute(
                "INSERT INTO model_data SELECT * FROM mdsrc.model_data "
                f"WHERE (lat, lon) IN (VALUES {values}) ON CONFLICT DO NOTHING"
            )
        else:
            con.execute("INSERT INTO model_data SELECT * FROM mdsrc.model_data "
                        "ON CONFLICT DO NOTHING")
        return con.execute("SELECT count(*) FROM model_data").fetchone()[0] - before
    except duckdb.CatalogException:
        # Source has no model_data table yet -> nothing to copy (not an error).
        return 0
    finally:
        con.execute("DETACH mdsrc")


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


# ---------------------------------------------------------------------------
# TAF worksheet tables (taf_worksheet*). The agent's pre-emit reasoning artifact
# (worksheet.py) is persisted here as the human-auditable record of WHY a TAF looks
# the way it does. The heavy JSON lives in one row; referenced evidence (tool
# name/args/receipt) is split into a child table so the worksheet record stays small
# and we can later analyze which tools were actually used. Same seam rule as obs/climo:
# all worksheet SQL lives HERE.
# ---------------------------------------------------------------------------

_WORKSHEET_DDL = """
CREATE TABLE IF NOT EXISTS taf_worksheets (
    worksheet_id          VARCHAR   NOT NULL,
    created_at            TIMESTAMP,
    completed_at          TIMESTAMP,
    station               VARCHAR,
    forecast_type         VARCHAR,
    valid_from_utc        VARCHAR,             -- as authored (ISO or DDHHMMZ); not a window read
    valid_to_utc          VARCHAR,
    mode                  VARCHAR,             -- off | advisory | required
    evidence_mode         VARCHAR,             -- off | key_claims | strict
    model                 VARCHAR,
    worksheet_json        JSON      NOT NULL,  -- the full TafWorksheet
    final_taf_text        VARCHAR,             -- the TAF emitted from this worksheet (if any)
    taf_product_json      JSON,                -- the TafProduct behind final_taf_text
    checker_findings_json JSON,                -- validate() findings at accept time
    status                VARCHAR,             -- e.g. accepted | advisory | superseded
    PRIMARY KEY (worksheet_id)
);
"""

_WORKSHEET_EVIDENCE_DDL = """
CREATE TABLE IF NOT EXISTS taf_worksheet_evidence (
    worksheet_id   VARCHAR   NOT NULL,
    evidence_id    VARCHAR   NOT NULL,         -- ev_001 ... (per-tool-call id the loop threads)
    tool_name      VARCHAR,
    tool_args_json JSON,
    receipt_text   VARCHAR,                    -- short receipt line, not the full output
    created_at     TIMESTAMP,
    PRIMARY KEY (worksheet_id, evidence_id)
);
"""


def init_worksheet_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create the taf_worksheet* tables if absent. Idempotent. Like init_climo_schema
    this is a WRITE-path side effect (the read-only tool conn never runs it); a driver
    calls it before persisting a worksheet."""
    con.execute(_WORKSHEET_DDL)
    con.execute(_WORKSHEET_EVIDENCE_DDL)


def insert_worksheet(
    con: duckdb.DuckDBPyConnection,
    *,
    worksheet_id: str,
    worksheet_json: str,
    station: str | None = None,
    forecast_type: str | None = None,
    valid_from_utc: str | None = None,
    valid_to_utc: str | None = None,
    mode: str | None = None,
    evidence_mode: str | None = None,
    model: str | None = None,
    final_taf_text: str | None = None,
    taf_product_json: str | None = None,
    checker_findings_json: str | None = None,
    status: str | None = None,
    created_at: datetime | None = None,
    completed_at: datetime | None = None,
    evidence: list[dict] | None = None,
) -> None:
    """Persist one final worksheet (+ its evidence rows) in a single transaction.
    Idempotent per worksheet_id: a re-run replaces the row and its evidence (the loop
    may re-submit before acceptance). `evidence` items are dicts with keys evidence_id,
    tool_name, tool_args_json, receipt_text. Requires init_worksheet_schema(con)."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    try:
        con.execute("BEGIN")
        con.execute("DELETE FROM taf_worksheets WHERE worksheet_id = ?", [worksheet_id])
        con.execute("DELETE FROM taf_worksheet_evidence WHERE worksheet_id = ?", [worksheet_id])
        con.execute(
            "INSERT INTO taf_worksheets VALUES ($worksheet_id, $created_at, $completed_at, "
            "$station, $forecast_type, $valid_from_utc, $valid_to_utc, $mode, $evidence_mode, "
            "$model, $worksheet_json, $final_taf_text, $taf_product_json, $checker_findings_json, "
            "$status)",
            {
                "worksheet_id": worksheet_id,
                "created_at": created_at or now,
                "completed_at": completed_at or now,
                "station": station,
                "forecast_type": forecast_type,
                "valid_from_utc": valid_from_utc,
                "valid_to_utc": valid_to_utc,
                "mode": mode,
                "evidence_mode": evidence_mode,
                "model": model,
                "worksheet_json": worksheet_json,
                "final_taf_text": final_taf_text,
                "taf_product_json": taf_product_json,
                "checker_findings_json": checker_findings_json,
                "status": status,
            },
        )
        for e in evidence or []:
            con.execute(
                "INSERT INTO taf_worksheet_evidence VALUES (?, ?, ?, ?, ?, ?)",
                [worksheet_id, e["evidence_id"], e.get("tool_name"),
                 e.get("tool_args_json"), e.get("receipt_text"), e.get("created_at") or now],
            )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise


def worksheet(con: duckdb.DuckDBPyConnection, worksheet_id: str) -> dict | None:
    """One persisted worksheet row (JSON columns come back as strings; json.loads at the
    boundary), or None. Evidence rows are read separately via worksheet_evidence()."""
    cur = con.execute("SELECT * FROM taf_worksheets WHERE worksheet_id = ?", [worksheet_id])
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    return dict(zip(cols, row)) if row else None


def worksheet_evidence(con: duckdb.DuckDBPyConnection, worksheet_id: str) -> list[dict]:
    """The evidence rows for one worksheet, in evidence_id order (empty if none)."""
    cur = con.execute(
        "SELECT * FROM taf_worksheet_evidence WHERE worksheet_id = ? ORDER BY evidence_id",
        [worksheet_id],
    )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# ---------------------------------------------------------------------------
# TAF scoring: the immutable TAF archive (`tafs`) + the shared `evaluations` spine
# (scoring-design sec 5.1 / 11). Scoring requires a TIME-ALIGNED forecast captured
# BEFORE truth is evaluated, so a scored TAF is frozen here byte-for-byte and never
# mutated. Same seam rule: all scoring SQL lives HERE; init is a WRITE-path side
# effect (the read-only tool conn never creates these). The scorers read these tables
# plus the half-open truth reader below; scoring MATH stays in the scorer modules.
# ---------------------------------------------------------------------------

_TAFS_DDL = """
CREATE TABLE IF NOT EXISTS tafs (
    taf_id                    VARCHAR   NOT NULL,   -- stable id (content-derived or UUID)
    station                   VARCHAR   NOT NULL,
    issue_time_utc            TIMESTAMP,            -- authoritative; never reconstructed later
    valid_from_utc            TIMESTAMP,            -- absolute UTC, normalized at insert
    valid_to_utc              TIMESTAMP,
    original_cycle_start_utc  TIMESTAMP,            -- anchors the TX/TN 24h temp window
    bulletin_type             VARCHAR,              -- routine|amendment|correction|cancellation
    producer_kind             VARCHAR,              -- artificial|official|human|model|baseline
    producer_name             VARCHAR,              -- model+run, unit, source, or baseline name
    source                    VARCHAR,              -- worksheet|awc_snapshot|import|baseline_synth
    canonical                 BOOLEAN,              -- False for unprovable post-hoc imports
    raw_taf                   VARCHAR   NOT NULL,    -- exact received bulletin, byte-for-byte
    parse_body                VARCHAR,              -- remark-stripped text fed to tafparse (audit)
    taf_product_json          JSON,                 -- TafProduct JSON for artificial TAFs (lossless)
    construction_json         JSON,                 -- synthetic-baseline construction inputs
    worksheet_id              VARCHAR,
    experiment_id             VARCHAR,
    run_id                    VARCHAR,
    parent_taf_id             VARCHAR,              -- amendment/correction lineage
    supersedes_taf_id         VARCHAR,
    content_sha256            VARCHAR,              -- dedup / tamper evidence
    archived_at               TIMESTAMP,            -- ingest time (distinct from issue time)
    PRIMARY KEY (taf_id)
);
"""

_EVALUATIONS_DDL = """
CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id         VARCHAR   NOT NULL,
    station               VARCHAR,
    taf_id                VARCHAR,                  -- the subject TAF in `tafs`
    valid_from            TIMESTAMP,
    valid_to              TIMESTAMP,
    status                VARCHAR,                  -- pending | scored | partial
    obs_hash              VARCHAR,                  -- SHA-256 over the canonical truth rows
    truth_policy_json     JSON,
    truth_policy_hash     VARCHAR,
    profile_snapshot_json JSON,                     -- the FULL profile, not only its hash
    profile_hash          VARCHAR,
    coverage_manifest_json JSON,
    created_at            TIMESTAMP,
    scored_at             TIMESTAMP,                -- set when status flips off pending
    PRIMARY KEY (evaluation_id)
);
"""

# Columns added after the evaluations table first shipped (CREATE TABLE IF NOT
# EXISTS never adds columns to an existing table; bring older DBs forward).
_EVALUATIONS_MIGRATIONS = (("taf_id", "VARCHAR"), ("scored_at", "TIMESTAMP"))


def init_scoring_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create the tafs archive + evaluations spine if absent. Idempotent. SEPARATE
    from init_schema (a build/scoring-path side effect only), like init_climo_schema
    and init_worksheet_schema; the read-only tool conn never runs this."""
    con.execute(_TAFS_DDL)
    con.execute(_EVALUATIONS_DDL)
    for col, typ in _EVALUATIONS_MIGRATIONS:
        con.execute(f"ALTER TABLE evaluations ADD COLUMN IF NOT EXISTS {col} {typ}")


def insert_taf(con: duckdb.DuckDBPyConnection, taf: dict) -> bool:
    """Archive one TAF immutably. Idempotent by taf_id (ON CONFLICT DO NOTHING) --
    never mutates an existing row, so a re-archive of identical content is a no-op.
    `taf` keys are the tafs columns (missing keys default to NULL). Returns True if a
    new row was inserted."""
    cols = [
        "taf_id", "station", "issue_time_utc", "valid_from_utc", "valid_to_utc",
        "original_cycle_start_utc", "bulletin_type", "producer_kind", "producer_name",
        "source", "canonical", "raw_taf", "parse_body", "taf_product_json",
        "construction_json", "worksheet_id", "experiment_id", "run_id",
        "parent_taf_id", "supersedes_taf_id", "content_sha256", "archived_at",
    ]
    before = con.execute("SELECT count(*) FROM tafs").fetchone()[0]
    con.execute(
        f"INSERT INTO tafs ({', '.join(cols)}) VALUES ({', '.join('$' + c for c in cols)}) "
        f"ON CONFLICT (taf_id) DO NOTHING",
        {c: taf.get(c, None) for c in cols},
    )
    return con.execute("SELECT count(*) FROM tafs").fetchone()[0] > before


def taf(con: duckdb.DuckDBPyConnection, taf_id: str) -> dict | None:
    """One archived TAF row (JSON columns come back as strings; json.loads at the
    boundary if needed), or None."""
    cur = con.execute("SELECT * FROM tafs WHERE taf_id = ?", [taf_id])
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    return dict(zip(cols, row)) if row else None


def previous_human_taf(con: duckdb.DuckDBPyConnection, station: str,
                       before: datetime | None = None,
                       valid_before: datetime | None = None) -> dict | None:
    """The most recent HUMAN/official TAF for a station, optionally issued STRICTLY before
    `before` (naive UTC). Feeds the leakage-safe get_previous_taf: the collector reads this
    from the benchmark archive and loads exactly that one bulletin into the per-run DB, so
    the model sees the forecast a human had in hand, never the one it is scored against.

    `valid_before` additionally requires valid_from_utc STRICTLY before it -- pass the
    agent's own valid_from so the current cycle's bulletin can NEVER qualify, no matter
    how early the human posted it. The issue-time buffer alone breaks silently if a
    station starts posting more than the buffer ahead of the hour (KBLV was observed
    posting 30 min early); a previous CYCLE is exact. Returns the tafs row dict (JSON
    columns as strings) or None."""
    sql = "SELECT * FROM tafs WHERE station = ? AND producer_kind IN ('human', 'official')"
    params: list = [station]
    if before is not None:
        sql += " AND issue_time_utc < ?"
        params.append(before)
    if valid_before is not None:
        sql += " AND valid_from_utc < ?"
        params.append(valid_before)
    sql += " ORDER BY issue_time_utc DESC LIMIT 1"
    cur = con.execute(sql, params)
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    return dict(zip(cols, row)) if row else None


def human_taf_for_window(con: duckdb.DuckDBPyConnection, station: str,
                         valid_from: datetime) -> dict | None:
    """The paired human ROUTINE TAF for a scoring window: same station, same
    valid_from (the agent's valid_from is pinned to the cycle hour, and roster
    routine TAFs are valid from that same hour). Latest issue wins if a correction
    re-posted the cycle. None if the poller never archived one -- the scorer then
    reports the human baseline unavailable, never guesses."""
    cur = con.execute(
        "SELECT * FROM tafs WHERE station = ? AND producer_kind IN ('human', 'official') "
        "AND bulletin_type = 'routine' AND valid_from_utc = ? "
        "ORDER BY issue_time_utc DESC LIMIT 1",
        [station, _to_naive_utc(valid_from)])
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    return dict(zip(cols, row)) if row else None


def human_bulletins_for_window(con: duckdb.DuckDBPyConnection, station: str,
                               valid_from: datetime, valid_to: datetime) -> list[dict]:
    """The human's OPERATIONAL product over a scoring window (T9): the routine TAF (as
    human_taf_for_window selects it) PLUS every human/official amendment/correction for the
    same station issued within [routine issue_time, valid_to), issue-time ascending. Empty
    if no routine is archived. Feeds tafstate.composite_taf. JSON columns as strings."""
    routine = human_taf_for_window(con, station, valid_from)
    if routine is None:
        return []
    cur = con.execute(
        "SELECT * FROM tafs WHERE station = ? AND producer_kind IN ('human', 'official') "
        "AND bulletin_type IN ('amendment', 'correction') "
        "AND issue_time_utc >= ? AND issue_time_utc < ? "
        "ORDER BY issue_time_utc ASC",
        [station, routine["issue_time_utc"], _to_naive_utc(valid_to)])
    cols = [d[0] for d in cur.description]
    amds = [dict(zip(cols, r)) for r in cur.fetchall()]
    return [routine, *amds]


def archived_human_tafs(con: duckdb.DuckDBPyConnection, *, before: datetime,
                        station: str | None = None,
                        routine_only: bool = True) -> list[dict]:
    """Archived human/official TAFs whose validity has FULLY elapsed (valid_to_utc < before,
    naive UTC) -- the population for standalone TAFVER difficulty scoring (score_taf.py
    --archive-difficulty), which scores a human TAF against obs with NO model run involved.
    routine_only keeps the scheduled-cycle bulletins (one difficulty score per station per
    cycle), dropping amendments. Returns tafs row dicts, oldest window first."""
    sql = ("SELECT * FROM tafs WHERE producer_kind IN ('human', 'official') "
           "AND valid_to_utc < ?")
    params: list = [_to_naive_utc(before)]
    if routine_only:
        sql += " AND bulletin_type = 'routine'"
    if station is not None:
        sql += " AND station = ?"
        params.append(station)
    sql += " ORDER BY valid_from_utc"
    cur = con.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def insert_evaluation(con: duckdb.DuckDBPyConnection, ev: dict) -> None:
    """Upsert one evaluation-spine row (idempotent replace by evaluation_id). `ev`
    keys are the evaluations columns; missing keys default to NULL."""
    cols = [
        "evaluation_id", "station", "taf_id", "valid_from", "valid_to", "status",
        "obs_hash", "truth_policy_json", "truth_policy_hash", "profile_snapshot_json",
        "profile_hash", "coverage_manifest_json", "created_at", "scored_at",
    ]
    con.execute("DELETE FROM evaluations WHERE evaluation_id = ?", [ev["evaluation_id"]])
    con.execute(
        f"INSERT INTO evaluations ({', '.join(cols)}) VALUES ({', '.join('$' + c for c in cols)})",
        {c: ev.get(c, None) for c in cols},
    )


def pending_evaluations(con: duckdb.DuckDBPyConnection,
                        before: datetime | None = None) -> list[dict]:
    """Pending evaluation rows whose validity window has fully ELAPSED (valid_to <=
    `before`, default now) -- the work list for score_taf.py --pending. Oldest first,
    so a backlog scores in collection order."""
    cutoff = _to_naive_utc(before) if before else datetime.now(timezone.utc).replace(tzinfo=None)
    cur = con.execute(
        "SELECT * FROM evaluations WHERE status = 'pending' AND valid_to <= ? "
        "ORDER BY valid_to, evaluation_id", [cutoff])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def finalize_evaluation(con: duckdb.DuckDBPyConnection, evaluation_id: str, *,
                        status: str, obs_hash: str | None = None,
                        truth_policy_json: str | None = None,
                        truth_policy_hash: str | None = None,
                        profile_snapshot_json: str | None = None,
                        profile_hash: str | None = None,
                        coverage_manifest_json: str | None = None) -> None:
    """Flip a pending evaluation to scored|partial and stamp the truth provenance
    (sec 11/12). UPDATE-in-place on the spine row; the created_at/identity columns
    are never touched."""
    con.execute(
        "UPDATE evaluations SET status = ?, obs_hash = ?, truth_policy_json = ?, "
        "truth_policy_hash = ?, profile_snapshot_json = ?, profile_hash = ?, "
        "coverage_manifest_json = ?, scored_at = ? WHERE evaluation_id = ?",
        [status, obs_hash, truth_policy_json, truth_policy_hash, profile_snapshot_json,
         profile_hash, coverage_manifest_json,
         datetime.now(timezone.utc).replace(tzinfo=None), evaluation_id])


def evaluation(con: duckdb.DuckDBPyConnection, evaluation_id: str) -> dict | None:
    """One evaluation row, or None."""
    cur = con.execute("SELECT * FROM evaluations WHERE evaluation_id = ?", [evaluation_id])
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    return dict(zip(cols, row)) if row else None


# ---------------------------------------------------------------------------
# Run provenance: one row per AGENT run (M4 persistence layer). The queryable
# summary of a run -- model, config, tokens, convergence, and REFERENCES to the
# emitted TAF (tafs), the worksheet (taf_worksheets), and the frozen transcript
# blob on disk. The transcript itself (the full messages array + images) is a FILE,
# not a column, per the DB rule. Written by runlog.persist_run; failures are rows
# too (taf_id NULL, fatal set) -- a model that never converged is benchmark data.
# ---------------------------------------------------------------------------

_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id            VARCHAR   NOT NULL,
    experiment_id     VARCHAR,              -- grouping key: one station x issue-time collection event
    station           VARCHAR,
    issue_time_utc    TIMESTAMP,            -- when the run was collected (~ human TAF issue time)
    valid_from_utc    TIMESTAMP,            -- the TAF window the model was asked to forecast
    valid_to_utc      TIMESTAMP,
    producer_kind     VARCHAR,              -- 'artificial' (the agent); a human TAF lives in tafs
    model             VARCHAR,              -- the model id we REQUESTED
    served_model      VARCHAR,              -- the model id the API RETURNED (may differ / drift)
    system_fingerprint VARCHAR,             -- provider build id; often NULL on Together
    base_url          VARCHAR,              -- endpoint that served the run (local/Together/vLLM)
    temperature       DOUBLE,
    max_tokens        INTEGER,
    seed              INTEGER,              -- determinism knob, if set
    toolset_hash      VARCHAR,              -- sha of the sorted tool names offered this run
    worksheet_mode    VARCHAR,              -- off | advisory | required
    config_id         VARCHAR,              -- hash of the full matrix cell (see runlog.config_id_for)
    harness_git_sha   VARCHAR,              -- code version that produced the run
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    n_steps           INTEGER,
    n_tool_calls      INTEGER,
    tools_used_json   JSON,                 -- {tool_name: call_count}
    stop_reason       VARCHAR,              -- emitted_clean | no_tool_call | max_steps | fatal
    convergence       VARCHAR,              -- unprompted | nudged | never
    first_emit_step   INTEGER,
    nudge_step        INTEGER,
    taf_id            VARCHAR,              -- the emitted TAF in tafs; NULL on a no-emit/fatal run
    taf_clean         BOOLEAN,              -- emitted a validate()-clean TAF?
    worksheet_id      VARCHAR,              -- the worksheet in taf_worksheets, if any
    transcript_path   VARCHAR,              -- path to messages.json (the frozen transcript blob)
    fatal             VARCHAR,              -- error string if the run crashed
    window_mismatch   VARCHAR,              -- set if the emitted TAF's window != the requested one (no evaluation created); NULL = matched
    duration_s        DOUBLE,               -- wall-clock seconds the collect process spent on the run (disambiguates queue-vs-runtime)
    tool_errors_json  JSON,                 -- {tool_name: n_failed_calls}; {} = a run where every tool call succeeded
    created_at        TIMESTAMP,
    PRIMARY KEY (run_id)
);
"""


# Provenance columns added after the runs table first shipped. CREATE TABLE IF NOT
# EXISTS never adds columns to an existing table, so bring older DBs forward explicitly.
_RUNS_MIGRATIONS = (
    ("served_model", "VARCHAR"), ("system_fingerprint", "VARCHAR"), ("base_url", "VARCHAR"),
    ("temperature", "DOUBLE"), ("max_tokens", "INTEGER"), ("seed", "INTEGER"),
    ("toolset_hash", "VARCHAR"), ("window_mismatch", "VARCHAR"), ("duration_s", "DOUBLE"),
    ("tool_errors_json", "JSON"),
)


def init_runs_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create the runs table if absent, then add any provenance columns missing from an
    older DB. Idempotent WRITE-path side effect (the read-only tool conn never runs it),
    like init_scoring_schema/init_worksheet_schema."""
    con.execute(_RUNS_DDL)
    for col, typ in _RUNS_MIGRATIONS:
        con.execute(f"ALTER TABLE runs ADD COLUMN IF NOT EXISTS {col} {typ}")


def insert_run(con: duckdb.DuckDBPyConnection, run: dict) -> None:
    """Upsert one run-provenance row (idempotent replace by run_id, so re-persisting a
    run overwrites rather than duplicates). `run` keys are the runs columns; missing keys
    default to NULL. Requires init_runs_schema(con)."""
    cols = [
        "run_id", "experiment_id", "station", "issue_time_utc", "valid_from_utc",
        "valid_to_utc", "producer_kind", "model", "served_model", "system_fingerprint",
        "base_url", "temperature", "max_tokens", "seed", "toolset_hash",
        "worksheet_mode", "config_id", "harness_git_sha", "prompt_tokens",
        "completion_tokens", "n_steps", "n_tool_calls", "tools_used_json", "stop_reason",
        "convergence", "first_emit_step", "nudge_step", "taf_id", "taf_clean",
        "worksheet_id", "transcript_path", "fatal", "window_mismatch", "duration_s",
        "tool_errors_json", "created_at",
    ]
    con.execute("DELETE FROM runs WHERE run_id = ?", [run["run_id"]])
    con.execute(
        f"INSERT INTO runs ({', '.join(cols)}) VALUES ({', '.join('$' + c for c in cols)})",
        {c: run.get(c, None) for c in cols},
    )


def run(con: duckdb.DuckDBPyConnection, run_id: str) -> dict | None:
    """One run-provenance row (JSON columns come back as strings), or None."""
    cur = con.execute("SELECT * FROM runs WHERE run_id = ?", [run_id])
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    return dict(zip(cols, row)) if row else None


def _deserialize_obs(rows: list[dict]) -> list[dict]:
    for row in rows:
        row["weather"] = json.loads(row["weather"]) if row["weather"] else []
        row["clouds"] = json.loads(row["clouds"]) if row["clouds"] else []
    return rows


def scoring_window(
    con: duckdb.DuckDBPyConnection,
    station: str,
    valid_from: datetime,
    valid_to: datetime,
) -> list[dict]:
    """Truth obs for scoring (sec 5.4): the HALF-OPEN in-window obs [valid_from,
    valid_to) -- so an ob exactly at valid_to is excluded, unlike window()'s inclusive
    BETWEEN -- PLUS the last ob at/before valid_from (carry-in) and the first ob
    at/after valid_to (interval terminator; never scored). Chronological, JSON
    deserialized at the boundary."""
    start, end = _to_naive_utc(valid_from), _to_naive_utc(valid_to)
    cur = con.execute(
        "SELECT * FROM ("
        "  (SELECT * FROM obs WHERE station = ? AND obs_time >= ? AND obs_time < ?)"
        "  UNION"
        "  (SELECT * FROM obs WHERE station = ? AND obs_time < ? ORDER BY obs_time DESC LIMIT 1)"
        "  UNION"
        "  (SELECT * FROM obs WHERE station = ? AND obs_time >= ? ORDER BY obs_time LIMIT 1)"
        ") ORDER BY obs_time",
        [station, start, end, station, start, station, end],
    )
    cols = [d[0] for d in cur.description]
    return _deserialize_obs([dict(zip(cols, r)) for r in cur.fetchall()])


# ---------------------------------------------------------------------------
# Per-scorer RESULT tables (scoring-design sec 11) -- the M4 deferred half.
# One shared shape: a `*_runs` provenance row per (evaluation, taf, subject,
# policy hash, scorer version) plus tall detail tables FK'd by scorer_run_id.
# Immutability: a rerun with IDENTICAL inputs finds the existing run and returns
# it -- runs are never replaced; any change (truth/policy/scorer version) is a
# NEW run. History is append-only. Scoring MATH stays in the scorer modules;
# these writers take plain dicts (Score.model_dump()) and map fields to columns.
# ---------------------------------------------------------------------------

SCORING_SCHEMA_VERSION = "1"    # bump when any result-table shape changes

# Provenance columns shared by all three *_runs tables.
_SCORER_RUN_COLS = """
    scorer_run_id     VARCHAR   NOT NULL,
    evaluation_id     VARCHAR   NOT NULL,
    taf_id            VARCHAR,              -- NULL for a synthetic baseline (persistence)
    subject           VARCHAR,              -- subject | persistence | human
    producer_kind     VARCHAR,
    producer_name     VARCHAR,
    policy_name       VARCHAR,
    policy_version    VARCHAR,
    policy_json       JSON,
    policy_hash       VARCHAR,
    scorer_version    VARCHAR,
    schema_version    VARCHAR,
    created_at        TIMESTAMP,
"""

_RESULTS_DDL = [
    f"""
CREATE TABLE IF NOT EXISTS tafver_runs (
    {_SCORER_RUN_COLS}
    provisional         BOOLEAN,
    combined_earned     DOUBLE,
    combined_available  INTEGER,
    combined_percent    DOUBLE,
    pw_accuracy         DOUBLE,
    obs_hash            VARCHAR,
    profile_hash        VARCHAR,
    category_stats_json JSON,
    pw_event_bias_json  JSON,
    PRIMARY KEY (scorer_run_id)
);
""",
    """
CREATE TABLE IF NOT EXISTS tafver_hourly (
    scorer_run_id    VARCHAR   NOT NULL,
    group_index      INTEGER,
    group_type       VARCHAR,
    interval_start   TIMESTAMP,
    interval_end     TIMESTAMP,
    lead_hr          INTEGER,
    element          VARCHAR,
    fcst_value       VARCHAR,
    obs_value        VARCHAR,
    fcst_cat         VARCHAR,
    obs_cat          VARCHAR,
    points_earned    DOUBLE,
    points_available INTEGER,
    status           VARCHAR,
    reason           VARCHAR
);
""",
    """
CREATE TABLE IF NOT EXISTS tafver_summary (
    scorer_run_id VARCHAR   NOT NULL,
    element       VARCHAR,              -- 'combined' = a group-type bucket row
    bucket        VARCHAR,              -- ALL | INITIAL | FM | BECMG | TEMPO | PROB
    earned        DOUBLE,
    available     INTEGER,
    percent       DOUBLE
);
""",
    f"""
CREATE TABLE IF NOT EXISTS tafamend_runs (
    {_SCORER_RUN_COLS}
    trigger_count              INTEGER,
    hours_scored               INTEGER,
    hours_in_spec              INTEGER,
    in_spec_fraction           DOUBLE,
    hours_after_amd_service    INTEGER,
    triggers_after_amd_service INTEGER,
    per_rule_episodes_json     JSON,
    rules_not_scored_json      JSON,
    category_series_json       JSON,
    PRIMARY KEY (scorer_run_id)
);
""",
    """
CREATE TABLE IF NOT EXISTS tafamend_rule_hours (
    scorer_run_id     VARCHAR   NOT NULL,
    hour              TIMESTAMP,
    rule              VARCHAR,
    result            VARCHAR,              -- pass | fail | unavailable
    reason            VARCHAR,
    fcst              VARCHAR,
    obs               VARCHAR,
    detail            VARCHAR,
    after_amd_service BOOLEAN
);
""",
    """
CREATE TABLE IF NOT EXISTS tafamend_events (
    scorer_run_id     VARCHAR   NOT NULL,
    kind              VARCHAR,              -- rule_episode | trigger
    rule              VARCHAR,              -- NULL on a trigger row
    onset             TIMESTAMP,
    end_time          TIMESTAMP,            -- NULL on a trigger row
    hours             INTEGER,
    worst_detail      VARCHAR,
    rules_json        JSON,                 -- trigger rows: the merged rule list
    after_amd_service BOOLEAN
);
""",
    f"""
CREATE TABLE IF NOT EXISTS tafskill_runs (
    {_SCORER_RUN_COLS}
    catalog_version      VARCHAR,
    mace                 DOUBLE,
    signed_mace_mean     DOUBLE,
    worst_excursion_json JSON,
    hours_scored         INTEGER,
    hours_unavailable    INTEGER,
    element_stats_json   JSON,
    contingency_json     JSON,
    deltas_json          JSON,              -- subject-vs-baseline deltas (subject row only)
    category_series_json JSON,
    PRIMARY KEY (scorer_run_id)
);
""",
    """
CREATE TABLE IF NOT EXISTS tafskill_element_rows (
    scorer_run_id VARCHAR   NOT NULL,
    grain         VARCHAR,                  -- hour | group | taf
    hour          TIMESTAMP,
    lead_hr       INTEGER,
    group_type    VARCHAR,
    element       VARCHAR,
    fcst_value    DOUBLE,
    obs_value     DOUBLE,
    signed_error  DOUBLE,
    abs_error     DOUBLE,
    status        VARCHAR,
    reason        VARCHAR
);
""",
    """
CREATE TABLE IF NOT EXISTS tafskill_event_hours (
    scorer_run_id VARCHAR   NOT NULL,
    hour          TIMESTAMP,
    event         VARCHAR,
    fcst          BOOLEAN,
    obs           BOOLEAN,                  -- NULL = not evaluable this hour
    via_tempo     BOOLEAN,
    cell          VARCHAR                   -- hit | miss | false_alarm | correct_negative
);
""",
    """
CREATE TABLE IF NOT EXISTS tafskill_episodes (
    scorer_run_id VARCHAR   NOT NULL,
    event         VARCHAR,
    disposition   VARCHAR,                  -- matched | missed | false_alarm
    obs_onset     TIMESTAMP,
    obs_end       TIMESTAMP,
    fcst_onset    TIMESTAMP,
    fcst_end      TIMESTAMP,
    onset_error_h DOUBLE,
    end_error_h   DOUBLE
);
""",
]


def init_results_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create the per-scorer result tables if absent. Idempotent; WRITE-path side
    effect only, like init_scoring_schema."""
    for ddl in _RESULTS_DDL:
        con.execute(ddl)


def archive_evaluation_id(taf_id: str) -> str:
    """Synthetic evaluation_id for standalone archive-difficulty scoring: a human TAF scored
    with NO model run has no evaluations-spine row, so its result rows key off this instead.
    Stable per taf_id (reruns stay idempotent via scorer_run_id) and the 'archdiff:' prefix
    keeps these rows separable from real model-run evaluations in the shared result tables."""
    return f"archdiff:{taf_id}"


def archive_difficulty_scored(con: duckdb.DuckDBPyConnection, taf_id: str) -> bool:
    """True if a TAFVER difficulty result already exists for this archived TAF -- lets the
    driver skip re-fetching obs/IEM (the persist itself is idempotent regardless). Assumes
    init_results_schema has run."""
    row = con.execute(
        "SELECT 1 FROM tafver_runs WHERE evaluation_id = ? AND taf_id = ? LIMIT 1",
        [archive_evaluation_id(taf_id), taf_id]).fetchone()
    return row is not None


def scorer_run_id(evaluation_id: str, taf_id: str | None, subject: str, scorer: str,
                  policy_hash: str, scorer_version: str) -> str:
    """Deterministic id for one scorer run: identical inputs -> identical id, which
    is what makes reruns append-only no-ops (sec 11)."""
    key = "|".join([evaluation_id, taf_id or "", subject, scorer, policy_hash,
                    scorer_version, SCORING_SCHEMA_VERSION])
    return f"sr_{hashlib.sha256(key.encode()).hexdigest()[:12]}"


def _j(payload) -> str | None:
    """JSON-serialize a python object for a JSON column (None passes through)."""
    return None if payload is None else json.dumps(payload, default=str)


def _insert_scorer_run(con, table: str, meta: dict, extra: dict) -> tuple[str, bool]:
    """Shared *_runs insert. Returns (scorer_run_id, created). created=False means an
    identical run already exists and the caller must NOT re-insert tall rows."""
    sid = scorer_run_id(meta["evaluation_id"], meta.get("taf_id"), meta["subject"],
                        table, meta["policy_hash"], meta["scorer_version"])
    if con.execute(f"SELECT 1 FROM {table} WHERE scorer_run_id = ?", [sid]).fetchone():
        return sid, False
    row = {
        "scorer_run_id": sid, "evaluation_id": meta["evaluation_id"],
        "taf_id": meta.get("taf_id"), "subject": meta["subject"],
        "producer_kind": meta.get("producer_kind"), "producer_name": meta.get("producer_name"),
        "policy_name": meta.get("policy_name"), "policy_version": meta.get("policy_version"),
        "policy_json": _j(meta.get("policy")), "policy_hash": meta["policy_hash"],
        "scorer_version": meta["scorer_version"], "schema_version": SCORING_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
        **extra,
    }
    cols = list(row)
    con.execute(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('$' + c for c in cols)})",
        row)
    return sid, True


def _insert_tall(con, table: str, sid: str, rows: list[dict], colmap: dict) -> None:
    """Bulk-insert tall detail rows. colmap maps table column -> source dict key
    (or a callable on the row dict)."""
    if not rows:
        return
    cols = ["scorer_run_id", *colmap]
    sql = f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
    params = [[sid] + [(src(r) if callable(src) else r.get(src)) for src in colmap.values()]
              for r in rows]
    con.executemany(sql, params)


def insert_tafver_result(con: duckdb.DuckDBPyConnection, meta: dict, score: dict) -> tuple[str, bool]:
    """Persist one TAFVER run: provenance row + hourly + summary tall rows, one txn.
    `meta` carries evaluation_id/taf_id/subject/producer/policy provenance; `score`
    is TafverScore.model_dump(). Idempotent by scorer_run_id (append-only reruns)."""
    con.execute("BEGIN TRANSACTION")
    try:
        sid, created = _insert_scorer_run(con, "tafver_runs", meta, {
            "provisional": score["provisional"],
            "combined_earned": score["combined_earned"],
            "combined_available": score["combined_available"],
            "combined_percent": score["combined_percent"],
            "pw_accuracy": score.get("pw_accuracy"),
            "obs_hash": score["obs_hash"], "profile_hash": score["profile_hash"],
            "category_stats_json": _j(score.get("category_stats")),
            "pw_event_bias_json": _j(score.get("pw_event_bias")),
        })
        if created:
            _insert_tall(con, "tafver_hourly", sid, score["rows"], {
                "group_index": "group_index", "group_type": "group_type",
                "interval_start": "interval_start", "interval_end": "interval_end",
                "lead_hr": "lead_hr", "element": "element",
                "fcst_value": "fcst_value", "obs_value": "obs_value",
                "fcst_cat": "fcst_cat", "obs_cat": "obs_cat",
                "points_earned": "points_earned", "points_available": "points_available",
                "status": "status", "reason": "reason"})
            summary = score["element_summaries"] + score["group_type_summaries"]
            _insert_tall(con, "tafver_summary", sid, summary, {
                "element": "element", "bucket": "bucket", "earned": "earned",
                "available": "available", "percent": "percent"})
        con.execute("COMMIT")
        return sid, created
    except Exception:
        con.execute("ROLLBACK")
        raise


def insert_tafamend_result(con: duckdb.DuckDBPyConnection, meta: dict, score: dict) -> tuple[str, bool]:
    """Persist one amendment-bust run: provenance row + rule-hours + events, one txn.
    `score` is TafAmendScore.model_dump(). Idempotent by scorer_run_id."""
    con.execute("BEGIN TRANSACTION")
    try:
        sid, created = _insert_scorer_run(con, "tafamend_runs", meta, {
            "trigger_count": score["trigger_count"],
            "hours_scored": score["hours_scored"],
            "hours_in_spec": score["hours_in_spec"],
            "in_spec_fraction": score["in_spec_fraction"],
            "hours_after_amd_service": score["hours_after_amd_service"],
            "triggers_after_amd_service": score["triggers_after_amd_service"],
            "per_rule_episodes_json": _j(score.get("per_rule_episodes")),
            "rules_not_scored_json": _j(score.get("rules_not_scored")),
            "category_series_json": _j(score.get("category_series")),
        })
        if created:
            _insert_tall(con, "tafamend_rule_hours", sid, score["hourly_results"], {
                "hour": "hour", "rule": "rule", "result": "result", "reason": "reason",
                "fcst": "fcst", "obs": "obs", "detail": "detail",
                "after_amd_service": "after_amd_service"})
            events = ([{**e, "kind": "rule_episode", "rules_json": None}
                       for e in score["rule_episodes"]]
                      + [{"kind": "trigger", "rule": None, "onset": t["onset"], "end": None,
                          "hours": None, "worst_detail": None,
                          "rules_json": _j(t["rules"]),
                          "after_amd_service": t["after_amd_service"]}
                         for t in score["triggers"]])
            _insert_tall(con, "tafamend_events", sid, events, {
                "kind": "kind", "rule": "rule", "onset": "onset", "end_time": "end",
                "hours": "hours", "worst_detail": "worst_detail", "rules_json": "rules_json",
                "after_amd_service": "after_amd_service"})
        con.execute("COMMIT")
        return sid, created
    except Exception:
        con.execute("ROLLBACK")
        raise


def insert_tafskill_result(con: duckdb.DuckDBPyConnection, meta: dict, score: dict,
                           deltas: dict | None = None) -> tuple[str, bool]:
    """Persist one skill run: provenance row + element rows + event hours + episodes,
    one txn. `score` is TafSkillScore.model_dump(); `deltas` (subject rows only) is
    the skill_deltas() output vs the persistence baseline. Idempotent by scorer_run_id."""
    con.execute("BEGIN TRANSACTION")
    try:
        sid, created = _insert_scorer_run(con, "tafskill_runs", meta, {
            "catalog_version": score["catalog_version"],
            "mace": score["mace"], "signed_mace_mean": score["signed_mace_mean"],
            "worst_excursion_json": _j(score.get("worst_excursion")),
            "hours_scored": score["hours_scored"],
            "hours_unavailable": score["hours_unavailable"],
            "element_stats_json": _j(score.get("element_stats")),
            "contingency_json": _j(score.get("contingency")),
            "deltas_json": _j(deltas),
            "category_series_json": _j(score.get("category_series")),
        })
        if created:
            _insert_tall(con, "tafskill_element_rows", sid, score["element_rows"], {
                "grain": "grain", "hour": "hour", "lead_hr": "lead_hr",
                "group_type": "group_type", "element": "element",
                "fcst_value": "fcst_value", "obs_value": "obs_value",
                "signed_error": "signed_error", "abs_error": "abs_error",
                "status": "status", "reason": "reason"})
            _insert_tall(con, "tafskill_event_hours", sid, score["event_hours"], {
                "hour": "hour", "event": "event", "fcst": "fcst", "obs": "obs",
                "via_tempo": "via_tempo", "cell": "cell"})
            _insert_tall(con, "tafskill_episodes", sid, score["episodes"], {
                "event": "event", "disposition": "disposition",
                "obs_onset": "obs_onset", "obs_end": "obs_end",
                "fcst_onset": "fcst_onset", "fcst_end": "fcst_end",
                "onset_error_h": "onset_error_h", "end_error_h": "end_error_h"})
        con.execute("COMMIT")
        return sid, created
    except Exception:
        con.execute("ROLLBACK")
        raise


# --- batch aggregators (sec 11): SUM tall rows across runs; scoring math beyond
# --- SUM/COUNT (percentages, contingency scores) stays in the scorer modules.

def tafver_points(con: duckdb.DuckDBPyConnection, *, subject: str = "subject",
                  station: str | None = None) -> list[dict]:
    """Pooled TAFVER points per element across runs: SUM(earned), SUM(available)
    over scored hourly rows. The caller divides (anti-averaging: points pool, never
    percentages)."""
    sql = ("SELECT h.element, SUM(h.points_earned) AS earned, "
           "SUM(h.points_available) AS available, COUNT(*) AS n_rows "
           "FROM tafver_hourly h JOIN tafver_runs r USING (scorer_run_id) "
           "JOIN evaluations e ON r.evaluation_id = e.evaluation_id "
           "WHERE h.status = 'scored' AND r.subject = ?")
    params: list = [subject]
    if station:
        sql += " AND e.station = ?"
        params.append(station)
    cur = con.execute(sql + " GROUP BY h.element ORDER BY h.element", params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def skill_errors(con: duckdb.DuckDBPyConnection, *, subject: str = "subject",
                 station: str | None = None) -> list[dict]:
    """Pooled element errors across runs: n, mean signed error (bias), mean abs
    error (MAE) over scored hour-grain rows."""
    sql = ("SELECT w.element, COUNT(*) AS n, AVG(w.signed_error) AS bias, "
           "AVG(w.abs_error) AS mae "
           "FROM tafskill_element_rows w JOIN tafskill_runs r USING (scorer_run_id) "
           "JOIN evaluations e ON r.evaluation_id = e.evaluation_id "
           "WHERE w.status = 'scored' AND w.grain = 'hour' AND r.subject = ?")
    params: list = [subject]
    if station:
        sql += " AND e.station = ?"
        params.append(station)
    cur = con.execute(sql + " GROUP BY w.element ORDER BY w.element", params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def skill_cells(con: duckdb.DuckDBPyConnection, *, subject: str = "subject",
                station: str | None = None) -> list[dict]:
    """Pooled 2x2 contingency cells per event across runs (a=hit, b=false_alarm,
    c=miss, d=correct_negative). Feed tafskill.contingency_scores for POD/FAR/CSI/HSS
    -- summed cells FIRST, scores second (pooling rule, sec 9.2)."""
    sql = ("SELECT w.event, "
           "SUM(CASE WHEN w.cell = 'hit' THEN 1 ELSE 0 END) AS a, "
           "SUM(CASE WHEN w.cell = 'false_alarm' THEN 1 ELSE 0 END) AS b, "
           "SUM(CASE WHEN w.cell = 'miss' THEN 1 ELSE 0 END) AS c, "
           "SUM(CASE WHEN w.cell = 'correct_negative' THEN 1 ELSE 0 END) AS d "
           "FROM tafskill_event_hours w JOIN tafskill_runs r USING (scorer_run_id) "
           "JOIN evaluations e ON r.evaluation_id = e.evaluation_id "
           "WHERE w.cell IS NOT NULL AND r.subject = ?")
    params: list = [subject]
    if station:
        sql += " AND e.station = ?"
        params.append(station)
    cur = con.execute(sql + " GROUP BY w.event ORDER BY w.event", params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]
