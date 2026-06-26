"""Dev script: render the present-weather color timeline (charts.wx_timeline) from
obs already in the DB, to eyeball Stage 1 before curating it onto the meteogram.
Writes to data/charts/temp/ (throwaway). No model, no network."""

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

png = charts.wx_timeline(rows, station=STATION, hours=HOURS)
out = Path("data/charts/temp")
out.mkdir(parents=True, exist_ok=True)
path = out / f"wxtimeline_{STATION}_{end:%Y%m%d_%H%MZ}.png"
path.write_bytes(png)
print(f"{len(rows)} obs -> {path} ({len(png)} bytes)")
