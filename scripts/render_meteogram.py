"""Dev script: render a meteogram PNG from obs already in the DB so we can EYEBALL
it before wiring it into the agent loop. No model, no network - pulls the last N
hours for a station from DuckDB and writes a PNG under data/charts/temp/ (a
throwaway dir: model/dev charts are recreatable on demand and shouldn't bloat the
workspace or git)."""

from datetime import timedelta
from pathlib import Path

from forecaster import charts, store
from forecaster.config import settings

STATION, HOURS = "KORD", 24

con = store.connect(settings.db_path, read_only=True)
anchor = store.latest(con, STATION, 1)
end = anchor[0]["obs_time"]
rows = store.window(con, STATION, end - timedelta(hours=HOURS), end)
con.close()

png = charts.meteogram(rows, station=STATION, hours=HOURS)
out = Path("data/charts/temp")
out.mkdir(parents=True, exist_ok=True)
path = out / f"meteogram_{STATION}_{end:%Y%m%d_%H%MZ}.png"
path.write_bytes(png)
print(f"{len(rows)} obs -> {path} ({len(png)} bytes)")
