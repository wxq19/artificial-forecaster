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
  pydantic-settings, python-dotenv, matplotlib, metpy). Run things with `uv run python ...`.
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
│   ├── tafparse.py       # TAF text -> typed TafObs (input seam; was taf.py)
│   ├── tafgen.py         # typed TafProduct -> valid AF TAF text (OUTPUT seam) + validate/roundtrip
│   ├── awc.py            # live aviationweather.gov client (METARs + TAFs; serves military fields)
│   ├── store.py          # the ONLY file that touches DuckDB (seam)
│   ├── iem.py            # historical METAR ingestion (IEM)
│   ├── wxcodes.py        # present-weather classify + deterministic severity rule
│   ├── charts.py         # the ONLY file that imports matplotlib/metpy (meteogram/wx_timeline/skewt)
│   ├── soundings.py      # live skew-T image client (SPC/Wyoming); fetch pixels, don't draw (seam)
│   ├── wxmaps.py         # live synoptic map client (WPC/OPC/SPC-meso/TT GFS); fetch charts (seam)
│   ├── fcstsounding.py   # model BUFKIT forecast-sounding fetch+parse (rendered by charts.skewt) (seam)
│   └── tools.py          # agent read tools + emit_taf OUTPUT tool + loop plumbing (-> agent.py later)
├── scripts/              # dev + end-to-end test drivers (markdown logs -> logs/)
├── docs/                 # references (FMH-1 wx table, AFMAN 15-124, AFH 15-101)
├── data/                 # GITIGNORED: forecaster.duckdb, charts/temp/, soundings/, maps/, fcstsoundings/ (throwaway)
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
  valid over the window, or persist-then-score via a future `taf` table). Deferred with TAFVER.
- TAF SEAM RENAME: `taf.py` -> `tafparse.py` (input seam) so the new `tafgen.py` (output seam) can't be
  confused with it. `awc.py` does NOT import it; the two test_taf scripts import `tafparse as taf`.
- TAF OUTPUT SEAM (`tafgen.py`, step 5 DONE): `TafProduct`/`TafProductGroup` pydantic models -> `render_taf()`
  emits canonical AF TAF text (AFMAN 15-124 ch.1, docs/TAF Coding.pdf). AF-specific vs the civil parse seam:
  30h validity, vis in METERS (Table 1.1, round DOWN), QNH on prevailing/FM/BECMG never TEMPO, NO PROB30/40
  (deliberately no `prob` field), TX/TN, LAST NO AMDS / LIMITED METWATCH remark helpers. Validated BYTE-EXACT
  against AFMAN Figures 1.3/1.4/1.6/1.7.
- HAZARD GROUPS baked in early (both seams): non-convective wind shear (WS), volcanic ash (VA), icing (6IchihihitL),
  turbulence (5BhBhBhBtL), Tables 1.4-1.7 + total-obscuration VV. `tafparse` RECOVERS all five from raw (the
  library drops/ignores them, like QNH/NSW); the four hazard sub-models live in `tafparse` and `tafgen` imports
  them (one shape, no circular import).
- `validate(p)` = AFMAN rule checker (returns findings, doesn't raise): 30h span, midnight END=24 not 00, wind /10,
  gust>mean (NOT a 10kt gap — AF gusts can be 10kt over the LULL, which isn't encoded), vis<9999 needs a cause,
  cloud summation + first-OVC, TS->CB, BECMG <=2h, TEMPO no-QNH/no-WS, FM self-contained, chronological groups,
  TX>=TN in first 24h. `roundtrip(p)` = render -> tafparse.parse -> compare (proves the two seams agree); EXCLUDES
  free-text remarks (AF remarks have NO delimiter, so a parser folds them into the groups — strip before parsing
  HUMAN TAFs for TAFVER). Round-trip caught a latent bug: FM minutes were read from the library's old typo
  `strart_minutes`; this version spells it `start_minutes` -> FM minutes were silently zeroed. Fixed.
- ERGONOMICS + GUARDRAILS: `TafProduct.issue(...)` computes the 30h end (midnight 00/24) so a routine TAF can't
  have the wrong span; `TafProduct.amend(orig, at=...)` clips validity + drops expired groups (1.3.2.1.2.1).
  Pydantic guardrails reject IMPOSSIBLE values at construction (station 4-letter, change in FM/BECMG/TEMPO,
  wind_dir 0-360, cover FEW/SCT/BKN/OVC); validate() catches well-formed-but-rule-breaking values. Three layers:
  guardrails -> validate() -> roundtrip(). Self-test `scripts/test_tafgen.py` (no model/network): 8/8, byte-exact
  figures + a negative case + an amend() carry-forward case.
- emit_taf OUTPUT TOOL (`tools.py`, step 6 DONE): the model's first OUTPUT tool. Parameter schema IS
  `TafProduct.model_json_schema()` (the one class is tool contract AND validator). `_emit_taf` builds the product
  (guardrails fire), renders + validate()s, returns the AFMAN check as the receipt with the product on
  `ToolResult.taf`; render is guarded and roundtrip skipped when findings exist (a malformed group is feedback,
  not a crash). It's a SINK — findings go back so the model RE-EMITS a fix. `scripts/test_emit_taf.py` drives the
  loop (pre-cutoff obs + meteogram -> reason -> emit -> validate -> re-emit), tool_choice='auto' so reasoning lands
  in the log.
- EMIT FINDINGS (benchmark-relevant): Together's function-calling accepts the nested $ref/$defs/anyOf pydantic
  schema AS-IS (no flattening). The model emits JSON loosely — quotes numbers (`"wind_dir":"240"`; the int|str
  union blocks pydantic's auto-coerce, so the validator coerces) and omits optional fields (gave `CloudLayer.type`
  a default). Verified end-to-end on KBLV valid 291600Z with a STRICT pre-cutoff feed: Gemma reasoned over the
  meteogram (spotted the diurnal wind dip), emitted, hit 4 AFMAN findings (BECMG>2h, missing QNH), READ them and
  self-corrected to a clean, round-trippable TAF in 2 steps.
- SKEW-T IMAGE SEAM (`soundings.py`, this session): a live upper-air client, sibling to `awc.py`/`iem.py` -- it
  FETCHES pre-rendered skew-T images and returns raw bytes. No matplotlib (we fetch pixels, we don't draw them,
  so `charts.py` stays the ONLY matplotlib file) and no SQL/DuckDB. A forecaster reads these exact products, so
  feeding the model the same image keeps the human-vs-model comparison honest. Two OBSERVED providers (RAOB,
  00Z/12Z only): SPC (spc.noaa.gov/exper/soundings -- SHARPpy-analyzed GIF with the derived indices printed ON
  the plot; CONUS/North America) and Wyoming (weather.uwyo.edu -- bare skew-T PNG, indices on a separate text
  page; GLOBAL coverage + a deep historical archive). Each names stations in its OWN id space (SPC: 3-letter
  site like MPX/OUN, or a WMO number; Wyoming: a WMO number like 72649 -- which IS MPX), so the caller passes the
  id that matches `source`. `synoptic_time()` snaps any time to the latest 00/12Z; `skewt_url()` builds the exact
  provider URL (Wyoming's wsgi page is an HTML WRAPPER -- the image itself is a stable
  /upperair/imgs/YYYYMMDDHH.<WMO>.skewt.png path, fetched directly, no HTML parse); `fetch_skewt()` returns bytes
  with an OPT-IN disk cache under `data/soundings/` (the air-gap/reproducibility path -- SuperCloud nodes have no
  internet -- but live-first while prototyping; `cache_path()` exposes where a cached image lands). Module-level
  throttle + a descriptive User-Agent, like iem/awc. `get_sounding` tool wired into `tools.py`, dispatched BEFORE
  the DB connect (it's a network fetch, needs no DB, like emit_taf) + `_image_mime()` magic-byte sniff so SPC's
  GIF and Wyoming's PNG both ride the EXISTING image-return plumbing (tool_messages was hardcoded image/png).
  Drivers: `scripts/test_sounding.py` (tool loop -- model calls get_sounding -> reads the image -> reasons;
  `--fetch-only` caches + prints the path for pre-review) and `scripts/test_sounding_ab.py` (a controlled
  SPC-vs-Wyoming A/B, same station+time, image fed directly so the source is the only variable). Conda-forge
  GRIB stack DEFERRED -- prototype off fetched images first.
- SYNOPTIC MAP SEAM (`wxmaps.py`, this session): a live surface/upper-air CHART client, sibling to `soundings.py`
  -- fetches PRE-RENDERED forecaster maps and returns raw bytes (no matplotlib, no SQL). A `CATALOG` of 14 charts
  (semantic name + the review-manifest code A1..C4), FOUR provider families: WPC (CONUS surface analysis +
  Day1/Day2 progs, GIF), OPC (Atlantic + Pacific oceanic surface analysis, PNG -- the OCONUS/maritime coverage
  WPC's CONUS view lacks), SPC mesoanalysis (hourly RAP ANALYSIS at MSLP/850/700/500/300mb, National sector s19,
  GIF), and TropicalTidbits (GFS FORECAST panels: 500mb hgt/vort, 250mb jet, MSLP+precip, 850mb temp; PNG).
  Analysis charts are "now" (no time arg); forecast charts sample GFS 6-hourly to f384 (frame = fhr//6 + 1;
  `latest_gfs_run()` picks the freshest posted cycle with a 5h post-lag; fhr must be a multiple of 6). `fetch_map
  (name, fhr=, run=, use_cache=)` returns bytes; `map_url()` gives provenance; opt-in cache under `data/maps/`;
  module throttle + descriptive UA. TT is THIRD-PARTY + hotlink-gated -- it needs a Referer header (`_REFERER`),
  and its URL scheme can change (fragile: watched, not trusted); the NOAA sources (WPC/OPC/SPC) are official and
  stable. `get_map` tool wired into `tools.py` (TOOLS now 5), dispatched BEFORE the DB connect like get_sounding;
  the enum is GENERATED from CATALOG so the tool contract can't drift; forecast `fhr` snaps to the 6h grid; rides
  the existing image plumbing (`_image_mime` handles GIF and PNG). Driver `scripts/fetch_maps.py` pulls the whole
  approved set (analysis once + forecast across a horizon, default f000-f036/6h) into `data/charts/temp/` with
  code-prefixed names for review; verified 38/38 on the GFS 12Z run.
- FORECAST SOUNDING SEAM (`fcstsounding.py` + `charts.skewt`, this session): MODEL forecast soundings, the
  forward-looking complement to the OBSERVED skew-Ts. `fcstsounding.py` is a fetch+parse client (sibling to
  soundings/wxmaps): pulls a model BUFKIT file (ISU mtarchive: `.../YYYY/MM/DD/bufkit/HH/<model>/<prefix>_<icao>.buf`)
  and parses ONE forecast-hour block into a plain `FcstProfile` (pres/tmpc/dwpc/drct/sknt/hght lists +
  CAPE/CIN/LIFT/SHOW/KINX/TOTL/PWAT/LCLP indices + lat/lon/elev/valid). No matplotlib, no SQL. 5 models
  (gfs/nam/nam4km/rap/hrrr): per-model file prefix (gfs->`gfs3_`, others match the name) + cycle/lag live in
  `_MODELS`; `latest_run()` snaps to the freshest posted cycle; `bufkit_url()` = provenance; opt-in cache under
  `data/fcstsoundings/`; module throttle. BUFKIT is TEXT, so unlike the observed soundings we RENDER the skew-T
  ourselves: `charts.skewt(profile)` (in the matplotlib seam; MetPy is PyPI -- NO conda) draws T/Td, a surface
  parcel path + CAPE/CIN shading + LCL, a height-colored AUTO-SCALED hodograph, and the indices box, with the
  right column pinned to the chart's top/bottom (SkewT under-fills its gridspec cell -- read the box after a draw,
  set_position, then a uniform tight-crop keeps the alignment). Fill rows (BUFKIT tops the profile with Td=-9999
  in the near-vacuum levels) are dropped in the parser. COVERAGE: dense over North America (US/Canada/Alaska/
  Hawaii/Mexico; ~2100 GFS stations); OCONUS SPARSE and GFS-ONLY (mesoscale models are N.America-only), specific
  AF bases hit-or-miss (Yokota yes; Kadena/Ramstein/Al Udeid no). `get_fcst_sounding` tool (station, model enum,
  fhr) wired into `tools.py` (TOOLS now 6), dispatched BEFORE the DB connect like get_sounding/get_map; a missing
  station (404) or forecast hour raises a ValueError the tool returns as FEEDBACK (with available hours), not a
  crash. Verified end-to-end (KMSP GFS f024). Added `metpy` (PyPI) via uv. FINDING: no clean pre-rendered
  forecast-sounding IMAGE exists (TT/TwisterData/COD render client-side; Pivotal 403s; rucsoundings was DOWN +
  text-only) -- BUFKIT-text -> our-render is the robust path. Spike `scripts/test_bufkit.py` kept as an
  SPC/Wyoming/BUFKIT comparison generator.
- POINT FORECAST TOOL (`get_point_forecast` + `fcstsounding.fetch_point`, this session): hourly MODEL point
  forecast TABLE from the BUFKIT SURFACE section (reuses the sounding fetch -- one `.buf` carries both profiles
  AND a surface time series). `_parse_surface` -> `PointForecast` (per forecast hour: RAW surface fields
  T2MS/TD2M/UWND/VWND/PMSL/LCLD/MCLD/HCLD/P01M + valid datetime). The tool renders a TEXT table (`tools._fmt_point`):
  rows = forecast hours, cols = variables (wind shown as dir/speed, everything else raw native units), default 48h.
  NO derived fields (RH/heat-index/gen-wx/ceiling/vis/gust/probabilities) -- DEFERRED per decision; v1 is raw
  model surface data. Modeled on `docs/TarpViewer-GenWx.csv`. TOOLS now 7. Verified (KLSV not in BUFKIT -> use
  KLAS). Spike `scripts/test_bufkit_point.py` writes the full transposed CSV for review.
- FULL-AGENT TAF TEST (`scripts/test_taf_agent.py`, this session): first end-to-end exercise of the WHOLE tool
  suite -- each of 4 models (Gemma/Qwen/Kimi/MiniMax) given all 7 read/data tools + emit_taf and asked to build a
  30h TAF for KLSV valid 2300Z. KLSV has AWC obs but NO BUFKIT output, so the prompt points at nearby KLAS for the
  model tools. Robust loop: all tool receipts appended, then images BATCHED into one follow-up user msg (avoids the
  OpenAI "tool replies must immediately follow the tool_calls msg" rule); per-model failures recorded not fatal.
  RESULTS: MiniMax + Gemma emitted AFMAN-CLEAN TAFs (both anchored on the current 25017G22KT, each with a late change group -- see CORRECTION below);
  MiniMax most efficient (4 steps/6.3k completion tok), Gemma thorough (6 steps, widest tool set incl get_map, 4
  emit->fix cycles). Qwen RUMINATED to ~35.5k tok and stalled at 3 structural findings (issued 24h not 30h,
  missing TX/TN) -- ran out of runway before fixing. Kimi FAILED distinctly: 40+ tool calls (get_map x22,
  fcst_sounding x13) across all 12 steps with ~zero reasoning (1.8k tok) and NEVER called emit_taf -- a gather-loop
  with no convergence. HEADLINE (benchmark-relevant): agentic ORCHESTRATION != chart reasoning -- Kimi/MiniMax were
  the strongest chart READERS in the map A/B, but here the discriminator is CONVERGENCE (stop gathering, emit, fix
  findings); MiniMax/Gemma converged, Kimi/Qwen did not -- the benchmark must score orchestration as its own axis.
  Infra held across 60+ calls (no crashes; KLAS-proxy worked; mixed GIF/PNG flowed).
  CORRECTION (2026-07-08 review): the clean TAFs are NOT flat 30h persistence -- both encode the diurnal WIND
  shift + a per-group QNH trend (Gemma: FM090000 23020G25KT, QNH 2975->2952; MiniMax: BECMG to 25010KT). What
  they persisted is SKY/VIS (SKC 9999), correct for a bone-dry July ridge. The REAL emit-quality gaps this run
  exposed: Qwen misread the observed DEWPOINT (-5C) as the overnight MIN temp (TN in reasoning) and then stalled
  in the emit_taf schema spiral, never emitting TX/TN (the Tier-2 emit_taf guide targets exactly this); and Gemma
  converted the SAME MSLP to different inHg at different points (1008 hPa -> 29.77 and 29.79; 1011 -> 29.85/29.86/
  29.88), a hand-conversion slip whose final QNH happened to land right. Log: `logs/taf_agent_KLSV_20260707-174247.md`.

## NEXT SESSION -- pick up here (paused 2026-07-08)
This session landed ALL 11 fixes in `review-findings-2026-07-08.md` (every box now [x]): Tier 1
(soundings 2h post-lag, get_map f0 default + single run pick, fcstsounding DRCT/SKNT + surface fill
masks -> point-table dashes, point-table valid-TIME slice, labeled batched images), Tier 2
(`tafgen.emit_taf_guide()` model-facing OUTPUT shape + TX/TN + clear-sky contract, injected into the
driver SYSTEM prompt; prescriptive emit rejections), Tier 3 (step-budget loop guard in
`test_taf_agent.py`: SYSTEM budget + one-time N-2 nudge + `TOOL_CAPS`=8; per-step token + End-ctx +
convergence reporting). All verified deterministically/stub; ruff clean; `test_tafgen.py` 9/9.

Then TWO live KLSV/072300Z runs (both this repo's target) surfaced the real frontier:
- OBS LEAKAGE (FIXED): re-running a PAST valid time leaked 16 obs from INSIDE the 30h window via the
  DB tools. Fix = `awc.load_metar(before=)` point-in-time ingest + a throwaway temp DB in
  `test_taf_agent.py` (`db_path` threaded to every `run_tool`); verified 0 leak, latest ob 072255Z.
  RESIDUAL leak (open): the network model tools (get_point_forecast/get_fcst_sounding/get_map) still
  fetch the LATEST run, which is post-start for a past valid time -- would need archived model runs to
  be airtight for historical dates. Weigh only if the benchmark must score historical dates.
- max_tokens=8000 is TOO LOW: it BROKE MiniMax (hit `length` mid-emit -> the loop's break-on-no-
  toolcall killed it) and did NOT fix Qwen (ruminates to any cap). Gemma RECOVERED once the leak was
  gone -- its earlier 30k rumination was the RETROSPECTIVE-obs confusion, not tokens.

DO NEXT, in order:
1. FIX break-on-`length` (the real convergence blocker): in `test_taf_agent.py`, when
   `finish_reason == "length"` AND no tool_calls, inject a user nudge ("you were cut off; be concise
   and call emit_taf now") and CONTINUE the loop instead of taking the final_answer/break path. Then
   set `max_tokens` ~12-16k and re-run. Expect MiniMax + Qwen to get a recovery path.
2. (Optional) Persist the vs-obs VERIFICATION below as a scoring note -- it is the first real TAFVER
   signal and seeds next-steps step 9.
3. The forecasting WORKSHEET (next-steps step 12) is the quality lever for the two systematic misses.

VERIFICATION vs observed METARs (KLSV 072255Z..081555Z = ~first 17h of validity; scored BOTH the
2026-07-07 leaked run and the 2026-07-08 leak-free run). Models get SKY/VIS right (SKC 9999 held the
whole period) and CAPTURE THE DIURNAL WIND cycle (gusty SW afternoon -> light/variable/calm overnight
-> next-afternoon pickup). Shared REAL misses, present WITH and WITHOUT leakage: overnight low +5C
(forecast TN32, actual 27C, timed too early) and the pressure RISE (obs climbed to A2990; TAFs held
~2975). Leak-free run additionally exposed an honest EVENING WIND-PEAK under-call (forecast 22G16 off
the 2255Z ob; winds actually built to 25017G22 in the first valid hours -- the leaked run had "seen"
that ob). Best converged TAFs: Gemma+Kimi (leak-free run `logs/taf_agent_KLSV_20260708-094810.md`),
Gemma+MiniMax (leaked run `..._20260707-174247.md`); Qwen incomplete both times; Kimi went from
never-emit -> clean once the loop guard/caps landed.

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
   via `load_metar` (source='awc'); TAFs are fetch+parse only (no `taf` table yet — see step 9).
5. **TAF output seam + parse checker (DONE).** `src/forecaster/tafgen.py` — `TafProduct` -> valid AF TAF
   text (`render_taf`) + the AFMAN rule checker (`validate`) + the render↔parse round-trip (`roundtrip`).
   Hazard groups, ergonomic constructors (`issue`/`amend`), and pydantic guardrails all landed; the
   `scripts/test_tafgen.py` self-test is 7/7 (byte-exact AFMAN figures + a negative case). See Status.
6. **Generate a first TAF (DONE).** `emit_taf` OUTPUT tool (schema = `TafProduct.model_json_schema()`) +
   `scripts/test_emit_taf.py`: pre-cutoff obs + meteogram → reason → emit → validate → re-emit. Verified
   end-to-end on KBLV valid 291600Z (Gemma read the meteogram, self-corrected 4 AFMAN findings to a clean,
   round-trippable TAF). See Status for the emit-schema findings.
7. **Skew-Ts + forecasting charts via FETCH (DONE).** Observed skew-Ts via `soundings.py` (SPC/Wyoming);
   surface + upper-air CHARTS via `wxmaps.py` (14-chart CATALOG: WPC/OPC surface, SPC mesoanalysis upper-air, TT
   GFS forecast panels); MODEL forecast SOUNDINGS via `fcstsounding.py` + `charts.skewt` (GFS/mesoscale BUFKIT
   text -> we render the skew-T with MetPy, uv tier). All fetch pre-rendered products or render from text, NOT
   GRIB. The conda-forge geospatial stack (eccodes/cfgrib/xarray/cartopy) stays DEFERRED -- GRIB self-render
   returns LATER only if fetched/text products prove insufficient. FINDING: no clean pre-rendered forecast-
   sounding IMAGE exists (all render client-side; rucsoundings was down + text-only), so BUFKIT-text->our-render
   is the robust path. Any matplotlib stays in the `charts.py` seam; clients stay network-only like
   `soundings.py`/`wxmaps.py`/`fcstsounding.py` (do NOT mix the two tiers).
8. **Climatology tool (BEFORE TAFVER).** A station-climatology lookup the agent can call (normals / extremes /
   frequency-by-month or -hour) so a forecast can be anchored to what is typical, not just the last 24h of obs.
   This is the deferred `climo` table in `store.py` (or a climo source) — build it when this tool needs it.
9. **TAFVER scoring (LATER — after charts + climo).** Score a generated TAF against the observed METARs
   T→T+24/30h — we already own the truth in the DB, so self-scoring needs NO TAF source. NOTE: needs a
   TIME-ALIGNED TAF (a current TAF barely overlaps PAST obs); build the deferred `taf` table (persist each TAF
   as issued → score once obs accumulate under its validity), and STRIP free-text remarks before parsing human
   TAFs (AF remarks have no delimiter). Compare vs human TAF + raw NWP (GFS/GALWEM) once those inputs exist.
10. Later: AF metric harness (OPVER/WARNVER); SuperCloud (Podman images, pre-stage weights, vLLM serve job).
11. **Model-facing REFERENCE schema for emit_taf (agent-quality; pull forward when convenient).** The KLSV
    emit runs showed the tool's `TafProduct` JSON schema does NOT surface nested-model fields to the model:
    optional nested models render as `anyOf[$ref, null]`, so the model can't see e.g. `TafTemp`'s
    `temp_c/day/hour`. It then GUESSES the shape from validation errors (burning turns) AND anxiously
    re-verifies fields it can't see -- a contributor to the step-1 rumination-to-token-cap loop. Build a
    clear model-facing reference for the OUTPUT shape (a worked TafProduct example and/or a flattened field
    guide passed in the prompt) -- NOT a replacement for the pydantic schema, which stays the validator. Goal:
    the model can SEE the contract, cutting both the schema-guessing thrash and the verification loop. See the
    KLSV reasoning-mechanics findings + the emit-schema findings in Status (model quotes numbers / omits optional
    fields).
12. **Forecasting WORKSHEET -- intermediate tasks before emit (agent-quality).** Instead of one
    "reason-then-emit" turn, give the model a structured pre-forecast worksheet (like a human forecaster's):
    explicit intermediate sub-tasks -- current trend, DIURNAL wind/gust cycle per valid day, TX/TN value+timing
    (SANITY-CHECK TX/TN against the diurnal range of OBSERVED temps, so a dewpoint isn't read as the min temp --
    the Qwen slip), pressure/QNH trend (DOUBLE-CHECK any unit conversion ONCE -- the Gemma MSLP->inHg slip),
    restriction/convective risks -- filled BEFORE calling emit_taf. The KLSV A/B showed a
    single general "diurnal recurrence" nudge already recovered gust INCLUSION and a much better TX, and REDUCED
    rumination (clearer direction -> fewer loops, not more). A worksheet generalizes that: decompose the task to
    direct convergence and improve quality (gust PLACEMENT, QNH trend, TEMPO usage -- all things the human TAF
    captured but the model missed). Pairs naturally with the climo tool (step 8) feeding the worksheet.

## Point forecast data (v1 built; enrichments deferred)
- DONE v1: `get_point_forecast` tool renders the BUFKIT surface time series as a text table (raw fields only) --
  see the Status bullet. Derived/probability fields intentionally deferred.
- A POINT hourly forecast TABLE (temp/dewpoint/RH/wind/sky/ceiling/vis/indices over time) -- the tabular cousin
  of the soundings/charts, modeled on the AF TarpViewer-GenWx product (`docs/TarpViewer-GenWx.csv`, a transposed
  hourly point forecast ~6 days out). CHOSEN SOURCE: the BUFKIT SURFACE section we ALREADY fetch in
  `fcstsounding.py` (the `.buf` file has a surface time-series block per forecast hour), so it reuses the fetch,
  keeps a KNOWN model run (the observed weakness of the alternatives), and pairs the point table with the sounding.
- FUTURE alternatives to evaluate for point model data (deliberately deferred): **Open-Meteo** (any global
  lat/lon, rich variables, trivial CSV -- but non-commercial license, PAID historical/archive API, and no clean
  model-run stamp; `gfs_seamless` blends runs) and **GRIBStream** (arbitrary-point GRIB time-series extraction).
  Both give ARBITRARY points, which BUFKIT's fixed ~2100-station list cannot; revisit if we need points off the
  BUFKIT list or want to cross-check.

## Open questions to confirm with MIT SuperCloud (supercloud@mit.edu)
- V100 variant (16 vs 32 GB) on assigned nodes.
- Max GPU job wall-time (persistent-server vs batch eval pattern).
- Recommended vLLM / container workflow, if any.

## Important caveats
- Model names and provider prices shift week to week — verify model strings against
  provider docs at build time rather than trusting any hardcoded list.
- If real (non-open-source) AF weather data is ever used, hosting may need an
  authorized DoD environment, not commercial cloud. Flag this if it comes up.

## Known problems to address
- **break-on-`length` kills a truncated emit turn (FIX FIRST -- see NEXT SESSION handoff).** In
  `test_taf_agent.py`, a turn that hits the completion cap comes back `finish_reason=length` with no
  tool_calls; the loop treats that as "model answered" and BREAKS. So a model whose emit-reasoning turn
  exceeds max_tokens is killed with no TAF. Confirmed live: at max_tokens=8000, MiniMax (a reliable
  converger) died mid-emit on step 3; Qwen dies to it at any cap. Fix = nudge-and-continue on
  length+no-toolcall, then run max_tokens ~12-16k. This, not the token value, is the real blocker.
- **Qwen agentic rumination (mitigate).** On the full-agent TAF task Qwen burned ~35.5k completion tokens and
  stalled at structural AFMAN findings without converging (issued 24h not 30h, no TX/TN). Model-specific
  (Gemma/MiniMax converge). The step-budget loop guard (below) now states the budget + nudges at turn N-2; the
  Tier-2 emit_taf guide removes the TX/TN schema thrash. Still open: the forecasting WORKSHEET (step 12) to
  direct convergence; and/or lower per-turn max_tokens with more steps. Re-test at the next live run.
- **Kimi no-emit gather-loop (loop guard BUILT 2026-07-08; live re-test pending).** Kimi called 40+ read tools
  and never emitted a TAF. `scripts/test_taf_agent.py` now has the three-layer backstop: the budget is stated in
  SYSTEM, a one-time user nudge fires at turn N-2 if no emit yet, and per-tool call caps (`TOOL_CAPS`, default 8)
  make a call past the cap return feedback instead of executing (kills the get_map x22 spam). Convergence is now
  scored per model (unprompted/nudged/never). Logic verified by a stub sim; confirm on a live Kimi re-run.
- **emit quality: sky/vis persistence + value slips (quality frontier).** The clean TAFs DO encode the diurnal
  wind cycle + per-group QNH trend; what they persist is sky/vis (SKC 9999), correct for a dry ridge. The real
  gaps (see the Status CORRECTION): a TX/TN value error (Qwen read the dewpoint as the min temp) and a repeated-
  unit-conversion inconsistency (Gemma MSLP->inHg). Addressed by the worksheet (step 12), not a code fix.

## Resolved
- MetPy wind-direction fill in forecast soundings — FIXED 2026-07-08. `fcstsounding.parse`
  masked fill rows on Td/T only, so a level with a good temp but a -9999 wind DIRECTION reached
  `charts.skewt` -> `mpcalc.wind_components` and warned `Input over 12.566 radians`. The profile
  mask now also drops levels with DRCT outside [0,360] or SKNT < 0; `_parse_surface` maps the
  -9999 surface sentinel to None (rendered as `--` in the point table). Verified: live KLAS GFS
  f012 renders with DRCT 129..277, SKNT min 4.0, and ZERO wind/radians warnings. (Tier-1 item 3.)
- TAF AMEND TX/TN clipping (tafgen.py `TafProduct.amend`) — FIXED 2026-07-07. Per the
  lead meteorologists: the AF NEVER issues a TAF (amended or not) without TX/TN, and on an
  amendment they cover the REMAINDER of the ORIGINAL 24h temp period (not "drop", which was
  the review's guess). `validate()` now HARD-REQUIRES both TX and TN, and the temp-time
  window is the uniform `[valid_from, valid_to-6]` (equals original_start+24h for routine AND
  amended, since valid_to always encodes original_start+30h — no AMD branch). `amend()`
  carries TX/TN forward (not stale-unchanged) and gained `max_temp`/`min_temp` override params
  to re-forecast the remainder; a carried temp whose hour has elapsed is flagged for re-emit.
  Requirement is gated on a new `TafProduct.military` flag (default True) so civil/NWS TAFs
  (which carry neither QNH nor TX/TN) are NOT falsely flagged. Late-amendment edge (window <24h)
  can't occur: TAFs have an 8h shelf life, so >=16h of temp window always remains. Self-test 8/8.
