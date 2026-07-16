"""DuckDB persistence — the ONLY file that imports duckdb or writes SQL.

Same seam idea as llm.py: the rest of the app calls these functions and never
sees the database. A MetarObs maps to columns HERE and travels no further. A
METAR line carries only day+time, so the real year/month is attached at persist
time (see insert_obs). All times are UTC.
"""

import contextlib
import fcntl
import json
import re
from datetime import datetime, timezone
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
    PRIMARY KEY (evaluation_id)
);
"""


def init_scoring_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Create the tafs archive + evaluations spine if absent. Idempotent. SEPARATE
    from init_schema (a build/scoring-path side effect only), like init_climo_schema
    and init_worksheet_schema; the read-only tool conn never runs this."""
    con.execute(_TAFS_DDL)
    con.execute(_EVALUATIONS_DDL)


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


def insert_evaluation(con: duckdb.DuckDBPyConnection, ev: dict) -> None:
    """Upsert one evaluation-spine row (idempotent replace by evaluation_id). `ev`
    keys are the evaluations columns; missing keys default to NULL."""
    cols = [
        "evaluation_id", "station", "valid_from", "valid_to", "status", "obs_hash",
        "truth_policy_json", "truth_policy_hash", "profile_snapshot_json",
        "profile_hash", "coverage_manifest_json", "created_at",
    ]
    con.execute("DELETE FROM evaluations WHERE evaluation_id = ?", [ev["evaluation_id"]])
    con.execute(
        f"INSERT INTO evaluations ({', '.join(cols)}) VALUES ({', '.join('$' + c for c in cols)})",
        {c: ev.get(c, None) for c in cols},
    )


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
    created_at        TIMESTAMP,
    PRIMARY KEY (run_id)
);
"""


# Provenance columns added after the runs table first shipped. CREATE TABLE IF NOT
# EXISTS never adds columns to an existing table, so bring older DBs forward explicitly.
_RUNS_MIGRATIONS = (
    ("served_model", "VARCHAR"), ("system_fingerprint", "VARCHAR"), ("base_url", "VARCHAR"),
    ("temperature", "DOUBLE"), ("max_tokens", "INTEGER"), ("seed", "INTEGER"),
    ("toolset_hash", "VARCHAR"),
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
        "worksheet_id", "transcript_path", "fatal", "created_at",
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
