#!/usr/bin/env bash
# Harvest the benchmark from the Pi (the single writer / source of truth) to this laptop
# for analysis, back it up to the cloud, and optionally prune old transcript blobs off the
# Pi so it does not hoard space. Run from the laptop, a few times/week (or from cron).
#
# Layout on the laptop mirrors the Pi's (DB + runs/ in ONE dir) so the DB's RELATIVE
# transcript_path values resolve locally -- the DB and its transcripts are portable together.
#
#   scripts/harvest.sh                 # pull + cloud backup, no pruning
#   PRUNE=1 RETAIN_DAYS=5 scripts/harvest.sh   # also prune Pi transcripts >5 days old
#
# Env overrides: PI, PI_DIR, DEST, RCLONE_REMOTE, RETAIN_DAYS, PRUNE.
set -euo pipefail

PI=${PI:-pi@192.168.0.21}
PI_DIR=${PI_DIR:-/home/pi/artificial-forecaster}
DEST=${DEST:-$(cd "$(dirname "$0")/.." && pwd)/data/benchmark}   # laptop canonical copy (gitignored)
RCLONE_REMOTE=${RCLONE_REMOTE:-onedrive:artificial-forecaster}   # MIT OneDrive target
RETAIN_DAYS=${RETAIN_DAYS:-7}      # transcripts newer than this stay on the Pi
PRUNE=${PRUNE:-0}                  # 1 = prune the Pi after a confirmed pull

mkdir -p "$DEST"

echo "[harvest] 1/5 flock-consistent DB snapshot on the Pi"
ssh "$PI" "cd '$PI_DIR' && flock data/forecaster.duckdb.lock cp data/forecaster.duckdb data/forecaster.duckdb.snap"

echo "[harvest] 2/5 pull DB snapshot -> $DEST/forecaster.duckdb"
rsync -az "$PI:$PI_DIR/data/forecaster.duckdb.snap" "$DEST/forecaster.duckdb"
ssh "$PI" "rm -f '$PI_DIR/data/forecaster.duckdb.snap'"

echo "[harvest] 3/5 pull transcript archive (incremental, append-only) -> $DEST/runs/"
rsync -az "$PI:$PI_DIR/data/runs/" "$DEST/runs/"

echo "[harvest] 4/5 cloud backup -> $RCLONE_REMOTE"
if command -v rclone >/dev/null 2>&1 && rclone listremotes 2>/dev/null | grep -q "^${RCLONE_REMOTE%%:*}:"; then
  rclone copy "$DEST/forecaster.duckdb" "$RCLONE_REMOTE/"
  rclone copy "$DEST/runs" "$RCLONE_REMOTE/runs"
  echo "  cloud backup complete"
else
  echo "  SKIP: rclone remote '${RCLONE_REMOTE%%:*}' not configured. One-time setup: rclone config"
fi

echo "[harvest] 5/5 prune Pi transcripts >$RETAIN_DAYS days (PRUNE=$PRUNE)"
if [ "$PRUNE" = "1" ]; then
  # Safe: step 3 already copied every blob to the laptop (and step 4 to the cloud).
  ssh "$PI" "find '$PI_DIR/data/runs' -name messages.json.gz -mtime +$RETAIN_DAYS -delete; \
             find '$PI_DIR/data/runs' -mindepth 1 -type d -empty -delete"
  echo "  pruned Pi transcripts older than $RETAIN_DAYS days"
else
  echo "  pruning OFF (set PRUNE=1 to enable once you trust the pull+backup)"
fi

echo "[harvest] done. Analyze on the laptop at: $DEST/forecaster.duckdb"
