# CLAUDE.md — Artificial Forecaster

Guidance for Claude Code working in this repo. Read this fully before acting.

## Working style (read first — this governs everything)
I am learning this codebase deeply and intentionally. Default to **advisor, not autopilot**:
- DO NOT autonomously implement large changes or multi-file edits.
- Prefer **explaining**, answering questions, and showing **small snippets** I can
  type or paste myself.
- **Ask before editing any file.** When I say "how should I…", I want the reasoning
  and a minimal example — not a finished implementation.
- Plan mode is my default. Propose an approach and wait; don't write files unless I say so.
- When you do show code, keep it small and explain *why*, so I understand it, not just copy it.
- `README.md` is MY personal context/tracker. Do not edit, overwrite, or restructure it.

## What this project is
Testing whether a multi-modal LLM (VLM) can replicate a human Air Force weather
forecaster. An agentic VLM ingests forecasting data (METARs, TAFs, NWP GRIB files,
satellite/radar imagery) and produces AF forecast products, scored against AF
verification metrics (TAFVER, OPVER, WARNVER) versus human forecasters and raw NWP
output (GFS / GALWEM). End goal: a scalable benchmark.

## Core design rule: the portability seam (do not break this)
The model lives behind an **OpenAI-compatible HTTP endpoint**. Application code NEVER
talks to a provider-specific SDK — only to a `base_url`. Swapping local → cloud → HPC
must remain a `.env` edit with ZERO code change. Preserve this seam in any suggestion.
- The ONLY file that constructs the client is `src/forecaster/llm.py`.
- All config flows through `src/forecaster/config.py` (typed, reads `.env`).
- Nothing else should hardcode a URL, key, or model name.

## Environments (the seam in action)
- **Local (dev):** Ollama serving a small Qwen3-VL at `http://localhost:11434/v1`.
  CPU-only laptop, no GPU — fine for testing logic, too slow for real vision inference.
- **Cloud (real inference):** Together AI, OpenAI-compatible at
  `https://api.together.ai/v1`. Current dev model: `Qwen/Qwen3.5-9B` (vision, supports
  function calling). Used because the laptop has no GPU.
- **HPC (final target):** MIT SuperCloud. Slurm scheduler, Volta V100 GPUs
  (`--gres=gpu:volta:1`), Podman containers (GPU via `--device nvidia.com/gpu=all`),
  vLLM serving the weights. Compute nodes have NO internet — weights/images pre-staged
  via the download partition (`-p download`), loaded from local paths.

## Model choices
- Local dev: `qwen3-vl:2b` / `:4b` (small, just for plumbing).
- Cloud: `Qwen/Qwen3.5-9B` on Together. Stay serverless (per-token), not dedicated.
  Use the Batch API (50% discount) for the big eval run.
- SuperCloud: 8B–32B class VLM (fits V100s).
- KEEP THE SAME MODEL TIER across environments so benchmark numbers are comparable.

## Tech stack
- **App / serving code:** Python, managed with `uv`. Pure-PyPI deps (openai,
  pydantic-settings, python-dotenv). Run things with `uv run python ...`.
  Lint/format with `uv run ruff ...`.
- **Geospatial / GRIB tools:** eccodes, cfgrib, xarray, cartopy, matplotlib. These are
  C-library-heavy — use **conda-forge**, NOT pip. (Not built yet.)
- Don't mix the two: app code stays uv/PyPI; the geospatial stack stays conda-forge.

## Architecture (planned)
- **Split images on HPC:** a GPU serving image (vLLM + VLM) and a CPU tools image
  (GRIB/charting, conda-forge). Data prep is CPU work and shouldn't hold a GPU node.
- **Tools run live in the agent loop:** model emits a tool call → our code runs
  cfgrib/cartopy → returns a rendered chart (PNG) → fed back to the VLM as an image.
  Example tools: skew-T sounding, 200mb wind/isotach chart.
- **The model is stateless.** It only knows what's in the `messages` array on each call.
  WE own context — building, trimming, and managing the `messages` list is our code's job.
- **GRIB/imagery do NOT go in a relational DB.** Keep arrays/images as files
  (GRIB/NetCDF/Zarr, PNGs); a DB stores structured records + file references only.
- **Postgres (or DuckDB/SQLite) is for:** run/experiment tracking, parsed
  METAR/TAF/PIREP observations, and verification scoring — the relational parts.

## Project structure
```
artificial-forecaster/
├── .env                  # real config + keys — GITIGNORED, never commit
├── .env.example          # template with blank values (committed)
├── README.md             # MY personal tracker — do not edit
├── CLAUDE.md             # this file
├── pyproject.toml        # has [tool.hatch.build.targets.wheel] + [tool.uv] package=true
├── src/forecaster/
│   ├── config.py         # typed settings, reads .env (the ONLY config source)
│   └── llm.py            # the ONLY file that builds the OpenAI client
└── test_endpoint.py      # text + vision smoke test
```

## Secrets hygiene (hard rule)
- API keys live in `.env` ONLY. Never in code, never in `.env.example`, never committed.
- `.gitignore` excludes `.env`, `.venv/`, `models/`, `*.tar`, `data/`.
- If you ever notice a key in tracked content, stop and flag it immediately.

## Status
- WSL2 + uv project scaffolded; git initialized; first commit done.
- Config seam (`config.py` + `llm.py`) built and working.
- Local Ollama VLM works; text + vision smoke test passes (slow on CPU).
- Cloud endpoint (Together, `Qwen/Qwen3.5-9B`) confirmed working.
- METAR ingestion built: `src/forecaster/metar.py` — `parse()` → typed `MetarObs`
  (the seam; the library object never escapes) + `render()` for the messages array.
  Library: `metar-taf-parser-mivek`. Pressure is read EXACT from the raw `A####`/`Q####`
  token (both inHg + hPa; the library's integer hPa is lossy). Present weather, multi-
  layer clouds, gusts/VRB/calm, and fractional/meters visibility all handled. Validated
  on CONUS (KBAB, KMSN) + OCONUS (RJFK, VEAT) — 107 obs, no errors.
- Verified the model reasons correctly over rendered METARs end-to-end via
  `scripts/test_metar.py` (writes a self-contained markdown log to `logs/*.md`).
  NOTE: Qwen3.5 is a REASONING model — chain-of-thought lands in a separate `reasoning`
  field and counts against `max_tokens`; set it high (8192) or `content` returns empty.
- Dev/run scripts now live in `scripts/`; reference imagery under `data/charts/` (untracked).
- Known v1 METAR deferrals (all survive in the retained raw line): trend groups
  (NOSIG/BECMG/TEMPO), RVR, variable-wind range, and the full month/year datetime.
- METAR fields extended: `auto`, `cavok`, `vertical_visibility_ft`, and derived
  `ceiling_ft` (lowest BKN/OVC, or a VV indefinite ceiling — VV counts per AFMAN
  15-111 11.4.4.6). Numeric visibility added as `vis_sm`/`vis_m`/`vis_flag`, converted
  via a Table 8.1 LOOKUP (not physics): both units stored, OCONUS capped at 9999m
  ("≥10 km" → >6SM), off-table meters snap to nearest row (ties → lower/pessimistic),
  CAVOK with no vis group → P6SM. `vis_flag` is 'M' (<), 'P' (>), or None (exact).
- DuckDB store built: `src/forecaster/store.py` — the ONLY file that imports `duckdb`
  or writes SQL (seam like `llm.py`). `connect(read_only=)`, `init_schema`, `insert_obs`
  (attaches year/month + a `source` lineage col + `report_type` METAR/SPECI; idempotent
  via PK `(station, obs_time)` + ON CONFLICT DO NOTHING), `count`, `latest`, `window`
  (time-range read, deserializes JSON). `weather`/`clouds` stored as JSON (derived scalars
  + JSON fidelity); JSON reads back as a STRING — `json.loads` at the boundary. Read-only
  conn rejects writes at the engine level (verified). `db_path` = `data/forecaster.duckdb`
  (gitignored); created on first `connect`+`init_schema`+insert.
- METAR/SPECI tagged: `MetarObs.report_type` ('METAR'|'SPECI'|None). `parse()` reads it
  from the leading keyword when a source keeps it (AWC/Skyvector); IEM strips it, so the
  loader supplies it. A SPECI = weather forced an off-cycle ob (a significance signal).
- IEM loader: `src/forecaster/iem.py` — ingestion orchestrator (uses metar+store seams,
  no SQL/duckdb of its own). Pulls historical METARs WITH authoritative UTC timestamps
  (no year/month inference), groups by month, `insert_obs(source='iem')`. Fetches
  report_type 3 (routine) and 4 (SPECI) SEPARATELY so each ob's type is certain — and
  EXCLUDES the 5-minute MADIS stream (report_type=1), which the AF workflow never uses
  (AWC/Skyvector show routine+SPECI only). Module-level min-interval throttle spaces every
  request (IEM rate-limits bursts hard). Validated: KORD Jan 2024 snowstorm → 97 obs
  (48 METAR + 49 SPECI), 0 MADISHF.
- First agent TOOL + loop built: `src/forecaster/tools.py` exposes read tools →
  `store` on a `read_only=True` conn. Returns a decoded summary + the RAW METAR/SPECI
  beneath each ob (so RMK/RVR/SLP/peak-wind aren't lost) + the type tag. The model
  CANNOT reach IEM — only DB reads are on its menu. `scripts/test_iem_tool.py` drives
  the end-to-end loop (NL question → tool call → answer) with a markdown log; skips
  ingest if the station is already loaded.
- SECOND tool: `get_latest_obs` (most recent N obs, newest-first) → `store.latest`.
  `TOOLS = [query_obs, get_latest]`. Tested two-tool SELECTION (`scripts/test_latest_tool.py`:
  no-range "what now" → picks get_latest, not query_obs) and a DEPENDENT two-call chain
  (`scripts/test_vis_match.py`: get_latest anchors "now" → query_obs builds the 24h window).
  Parallel vs sequential: independent calls go in ONE turn (`for tc in msg.tool_calls`);
  dependent calls need the outer turn loop (B's args come from A's result).
- Three seam/render bugs shaken out by those tests, each a benchmark-relevant finding:
  (a) `store.latest` now deserializes the JSON cols like `window` — it was returning JSON
  STRINGS, so `_fmt` joined chars and garbled present-wx. (b) `_fmt`'s decoded line now
  prints a FULL ISO UTC stamp (`2024-01-13T23:51Z`) not bare `DDHHMMZ` — killed the model's
  DDHHMM→HH:MM:SS misparse AND its year-GUESSING (year now read from data; the raw line
  still shows DDHHMMZ for fidelity). (c) `store.window` coerces tz-aware bounds to naive
  UTC via `_to_naive_utc` — a `Z`-suffixed start/end shifted the window by the host's local
  offset (8h) → silent undercount even though the model reasoned PERFECTLY. The seam owns
  the naive-UTC contract; this is the "infra bug masquerading as a model error" class.
- Qwen3.5 RUMINATES: on a multi-step count it re-derived the same (correct) answer ~10× and
  spilled the whole answer into the `reasoning` field, leaving `content` EMPTY (finish_reason
  `stop`, not `length` — so it's not a token cap). Mitigated by a "state it ONCE and stop"
  instruction in the prompt (8340→5050 completion tok, content populated, answer still right).
  NOT eliminated — see harness guard in next steps.

## Likely next steps
1. **v2 — METAR store (DONE).** `src/forecaster/store.py` is the ONLY file that touches
   DuckDB (seam like `llm.py`). Chosen over Postgres deliberately: single-tenant,
   reproducible (a `.duckdb` file you can ship/hash), embedded (works on the air-gapped
   SuperCloud node, zero ops), analytics-first, queries Parquet/Arrow in place. `obs`
   table built; `MetarObs` rows persist with year/month attached here. Numeric visibility
   (Table 8.1) done. `climo`/`runs`/`scores`/`assets` deferred (YAGNI) — add each when the
   code that writes it exists; keep all SQL inside `store.py`.
2. **v2.5 — first agent TOOL + IEM ingestion (DONE).** `query_obs` tool on a read-only
   conn + `iem.py` loader + end-to-end agent loop, all verified on the KORD snowstorm.
   Remaining polish when needed: an intent-check that echoes the date range on manual
   ingest; AWC API path for recent obs; copy-paste path for ad-hoc.
3. **Harden + grow the toolset (IN PROGRESS).** `get_latest_obs` added; tool-selection +
   dependent-chain both verified. Candidate tools still open: a trend query (ceiling/vis/
   wind over last N hours), station metadata, climatology lookup. Consider thinning/paging
   for very wide windows. Watch model reasoning errors (timestamp conflation, rumination) —
   these are what the benchmark must score. On tool count: don't fold tools into one
   mode-switch mega-tool (moves the choice into arg-filling, muddies descriptions); keep
   distinct verbs, namespace by data domain, and subset `TOOLS` per task phase as the
   catalog grows — the "5-10 tools" limit is about CONFUSABILITY, not raw count.
   - HARNESS GUARD (TODO, deferred this session): the agent loop reads the answer from
     `msg.content` only. When a reasoning model leaves `content` empty but `reasoning` is
     non-empty (the rumination case above), a CORRECT answer logs as blank — a silent
     scoring bug. Add: if `content` is empty + `reasoning` present + finish_reason `stop`,
     surface/flag the reasoning. Cheap insurance once the eval harness scores real runs.
4. Build the agent loop + first GRIB tool (skew-T or 200mb winds) — Weeks 1-5.
5. Build the AF metric harness (TAFVER / OPVER / WARNVER) — Weeks 6-8.
6. Stand up SuperCloud: Podman images, pre-stage weights, vLLM serve job.

## Open questions to confirm with MIT SuperCloud (supercloud@mit.edu)
- V100 variant (16 vs 32 GB) on assigned nodes.
- Max GPU job wall-time (persistent-server vs batch eval pattern).
- Recommended vLLM / container workflow, if any.

## Important caveats
- Model names and provider prices shift week to week — verify model strings against
  provider docs at build time rather than trusting any hardcoded list.
- If real (non-open-source) AF weather data is ever used, hosting may need an
  authorized DoD environment, not commercial cloud. Flag this if it comes up.
