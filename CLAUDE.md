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
- **No emojis, ever.** Not in source, comments, docstrings, or generated log/Markdown
  output. Use plain text (e.g. `PASS`/`FAIL`, not check/cross marks).

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
│   ├── llm.py            # the ONLY file that builds the OpenAI client (seam)
│   ├── metar.py          # METAR text <-> typed MetarObs (input seam)
│   ├── store.py          # the ONLY file that touches DuckDB (seam)
│   ├── iem.py            # historical METAR ingestion (IEM)
│   ├── wxcodes.py        # present-weather classify + deterministic severity rule
│   ├── charts.py         # the ONLY file that imports matplotlib (meteogram/wx_timeline)
│   └── tools.py          # agent-facing read tools + loop plumbing (-> agent.py later)
├── scripts/              # dev + end-to-end test drivers (markdown logs -> logs/)
├── docs/                 # references (FMH-1 wx table, AFMAN 15-124, AFH 15-101)
├── data/                 # GITIGNORED: forecaster.duckdb, charts/temp/ (throwaway)
└── logs/                 # run transcripts (markdown)
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
  NOT eliminated — mitigated by the harness guard below.
- HARNESS GUARD (DONE): `tools.final_answer(msg, finish_reason)` recovers a correct answer
  stranded in the `reasoning` field (content empty + reasoning present + finish_reason `stop`)
  instead of logging blank — the rumination case above. Wired into the drivers.
- THIRD tool = a METEOGRAM image (the architecture keystone): `get_trend` renders a 5-panel
  meteogram (T/Td, wind, vis, ceiling, pressure) + a CURATED colored present-weather band → PNG,
  fed BACK to the VLM as a base64 image. `src/forecaster/charts.py` is the ONLY file that imports
  matplotlib (seam; Agg backend; returns PNG bytes, no disk I/O). Two charts: `meteogram()` (band
  = top-2-frequent ∪ top-3-severe families + a small phenomenon×intensity key) and standalone
  `wx_timeline()` (ALL families + full legend). matplotlib is PyPI-clean → stays in the uv app
  tier (conda is still only for cfgrib/cartopy later). Model/dev charts write to `data/charts/temp/`
  (throwaway, recreatable). Default look-back 24h, cap 48h.
- Present-weather SEVERITY is a deterministic manual rule: `src/forecaster/wxcodes.py` maps each
  METAR wx group to (family=color, intensity=opacity, severity 0-10). Grounded in the FMH-1 table
  (docs/Present Weather Values.png). Fixed tiers for convective/frozen precip (TS/FZ descriptors
  dominate; left-most-in-tier worse via decimal fractions); +/-/VC adjusts precip; OBSCURATIONS are
  vis-driven (severity = associated visibility via FAA flight-category buckets) since they have no
  -/+ — except VA (engine hazard, fixed high). DZ treated as an obscuration. Deferred as cosmetic:
  BL/MI/SH/UP descriptor nuances (severity already comes from the code + values).
- IMAGE-RETURN plumbing: a tool reply is text-only in the OpenAI format, so a chart can't ride in
  the `tool` message. `run_tool` returns `ToolResult(text, images: list[bytes], window)`;
  `tools.tool_messages(call_id, result)` emits the required `tool` receipt PLUS a follow-up `user`
  message carrying each PNG as a base64 image_url. `images` is a LIST so one call can return several
  charts (v2). Verified: model calls get_trend → reads the meteogram → reasons over panels AND the
  color band.
- IMAGE-vs-TEXT finding (benchmark-relevant): the model reads the chart's SHAPE well, but raw-METAR
  TEXT caught what the image can't — wind DIRECTION shift + BLSN + exact values. Using BOTH is best
  ONLY when the two tools cover the SAME window; when windows desync the model fuses contradictory
  inputs into a confident WRONG answer and won't self-flag the conflict.
- TIME-CORRELATION fixes (1+2+3) from that finding, all in `tools.py`:
  - Fix 1: `query_obs` gained a RELATIVE `hours` mode anchored SERVER-SIDE on the latest ob —
    identical anchor to get_trend, so "last 24h" resolves to the SAME window via our code; the model
    never computes timestamps. Absolute start/end kept for historical ranges.
  - Fix 2: every time-bounded result echoes a canonical `window: <start>Z .. <end>Z` line.
  - Fix 3: `tools.window_conflict(windows)` — conversation-wide, flags ANY distinct windows with a
    non-blocking advisory injected as a `user` message before the next turn. `ToolResult.window`
    carries the machine-readable span.
  - Two bugs found+fixed: (a) tz-AWARE (`fromisoformat('...Z')`) vs naive obs_time windows compared
    unequal → false-positive note; normalized to naive UTC (`_naive_utc`) — the naive-UTC seam
    contract again. (b) a mid-conversation `role:system` message → Together 400; inject notes as
    `role:user`. Finding: Fixes 1+2 are robust enough the model self-aligns even when coerced toward
    absolute dates (chains get_latest to anchor, or copies the echoed window); Fix 3 is the backstop.
    All loop plumbing (final_answer, tool_messages, window_conflict) → migrate to a future `agent.py`.
- AWC LIVE client + TAF PARSE SEAM (this session). `src/forecaster/awc.py` — live aviationweather.gov
  client (seam like `iem.py`, no SQL/duckdb of its own). `fetch_metar`/`fetch_taf` use format=json so every
  report arrives with an AUTHORITATIVE epoch/ISO timestamp (and, for METARs, metarType=METAR/SPECI — no
  second fetch like IEM needs); the raw TAF is ONE line (the raw text format wraps). Serves the military
  aerodromes IEM does NOT. `awc.load_metar` = the orchestrator half (fetch→metar.parse→store.insert_obs,
  source='awc'); idempotent + dedups against IEM rows via the (station, obs_time) PK. Verified KORD/KIND live.
- `src/forecaster/taf.py` — TAF text → typed `TafObs` (the forecast INPUT seam, symmetric to metar.py). Uses
  the SAME library's `TAFParser` (no new dep). Recursive shape: one `TafGroup` models the PREVAILING period
  AND each change group (FM/BECMG/TEMPO, optional PROBxx). Library handles the full ICAO/US grammar (US-SM vs
  intl-meters vis, CAVOK, PROBxx, TX/TN, FM minutes); object never escapes; raw retained.
- TAF library gaps found empirically (all survive in raw): (a) QNHxxxxINS (US-military altimeter) dropped →
  read EXACT from raw, PER-GROUP (`TafGroup.qnh_inhg`), rendered as a column ONLY for military TAFs. (b) NSW
  ('no significant weather' = weather forecast to END) dropped → re-attached to the group's weather so the
  decoded view isn't a misleading blank. (c) VCSH / bare SH (a showers descriptor with NO phenomenon) silently
  dropped — the library keeps VCTS/VCFG/-SHRA. Recovered from raw in BOTH taf.py AND metar.py (same library,
  same blind spot; `_SH_ONLY` lives in metar.py, imported by taf). Per-group recovery rides a `_split_periods`
  chunk-align (PROBxx absorbs a trailing TEMPO/BECMG = one boundary); skips if it can't align 1:1.
- TAF/METAR vis DISPLAY: never show the library's '>10000' (9999 = unrestricted, never reported as >10000) —
  both modules show the reported token (P6SM / 9999). TAF meters vis carries the SM equivalent inline
  ('9999 (P6SM)', '4800 (3SM)') via `metar._parse_vis` (Table 8.1, single source) so the AF SM reader (TAFVER
  unit) isn't left converting; numeric `vis_sm/vis_m/vis_flag` on `TafGroup`. TAF raw view is REFLOWED (json is
  one flat line) onto conventional per-group lines. Render bugs shaken out by the drivers: the 'm'-suffix wart;
  a BARE PROBxx group printing 'PROB30 PROB ...' (change_type IS 'PROB') → 'PROB30 2621/2701'; and metar.py's
  prose 'cols:' legend → a real aligned column header.
- Drivers: `scripts/test_taf.py` (live fetch → parse → render → model; military+civil+intl spread) and
  `scripts/test_taf_verify.py` (the FULL new path: load_metar ingest → store read-back → render → manual TAFVER
  reasoning over the elapsed validity). Both use the final_answer guard + a 'state once' nudge + MAX_TOKENS 12288.
- MODEL FINDING (benchmark-relevant): Qwen3.5 RUMINATES to the token cap on TAF tasks (finish_reason=length,
  content empty) — model-specific, NOT prompt-fixable; the guard flags it cleanly vs a silent blank. Gemma
  (`google/gemma-4-31B-it`) answered the SAME tasks cleanly+concisely (stop, ~800-1300 tok) and read every
  render feature (dual-unit vis, per-line QNH, ceiling=lowest BKN/OVC, recovered NSW/VCSH). Tier choice matters.
- TAFVER TIME-ALIGNMENT finding (from test_taf_verify): a FRESHLY-issued TAF has ~0 elapsed validity, so
  'current TAF + last 24h obs' is degenerate — only ob(s) after valid_from overlap (1 ob in the KIND run; the
  other 29 predate the TAF). The verification REASONING works; it needs a TIME-ALIGNED TAF (an archived TAF
  valid over the window, or persist-then-score via a future `taf` table). Deferred with TAFVER (step 7).

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
3. **Harden + grow the toolset (DONE).** `get_latest_obs` + `get_trend` (meteogram image),
   tool-selection + dependent-chain + image-return all verified; harness guard + time-correlation
   Fixes 1-3 landed (see Status). Tool-count guidance still holds: distinct verbs, namespace by
   data domain, subset `TOOLS` per phase — the "5-10 tools" limit is about CONFUSABILITY, not raw
   count. Still-open candidate tools if needed: station metadata, climatology lookup.
4. **TAF + METAR LIVE grabber from AWC (DONE).** `src/forecaster/awc.py` (fetch_metar/fetch_taf +
   load_metar) pulls live obs AND TAFs — incl. the military sites IEM does NOT serve — and the
   `src/forecaster/taf.py` PARSE seam (TafObs) is built and validated (see Status). Live METARs persist
   via `load_metar` (source='awc'); TAFs are fetch+parse only (no `taf` table yet — see step 7).
5. **TAF output seam + parse checker (from AFMAN 15-124 Ch.1, in docs/).** Define a `TafProduct`
   structure (validity period, FM/BECMG/TEMPO/PROB groups, wind/vis/wx/sky) that renders to VALID
   TAF text AND parses back — the OUTPUT seam symmetric to `MetarObs` on the input side — plus a
   format/parse checker that validates a TAF against the AFMAN 15-124 rules.
6. **Generate a first TAF.** Agent: obs up to a cutoff T + the trend meteogram → emit a TafProduct.
   A persistence-only TAF is KNOWN to verify badly — expected; the point is the end-to-end product.
7. **TAFVER scoring (LATER).** Score a generated TAF against the observed METARs T→T+24h — we
   already own the truth in the DB, so self-scoring needs NO TAF source. Compare vs human TAF + raw
   NWP (GFS/GALWEM) once those inputs exist. NOTE (from test_taf_verify): verification needs a
   TIME-ALIGNED TAF — a current TAF barely overlaps PAST obs. Build the deferred `taf` table here
   (persist each TAF as issued → score once obs accumulate under its validity), or pull an archived TAF.
8. Later: first GRIB tool (skew-T / 200mb winds) + the conda geospatial stack; AF metric harness
   (OPVER/WARNVER); SuperCloud (Podman images, pre-stage weights, vLLM serve job).

## Open questions to confirm with MIT SuperCloud (supercloud@mit.edu)
- V100 variant (16 vs 32 GB) on assigned nodes.
- Max GPU job wall-time (persistent-server vs batch eval pattern).
- Recommended vLLM / container workflow, if any.

## Important caveats
- Model names and provider prices shift week to week — verify model strings against
  provider docs at build time rather than trusting any hardcoded list.
- If real (non-open-source) AF weather data is ever used, hosting may need an
  authorized DoD environment, not commercial cloud. Flag this if it comes up.
