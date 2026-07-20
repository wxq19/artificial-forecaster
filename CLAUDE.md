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
│   ├── climo.py          # station-climatology builder (scratch-DB ingest; NO SQL) -> store.rebuild_climo
│   ├── imagery.py        # live satellite (NESDIS/STAR) + radar (IEM radmap.php) image client (seam)
│   ├── radarsites.py     # WSR-88D site table (nearest-radar lookup; regen via scripts/build_radarsites.py)
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

- SATELLITE + RADAR IMAGERY SEAM (`imagery.py` + `get_imagery`, this session): observational imagery,
  sibling to `wxmaps.py`/`soundings.py` (fetch pre-rendered, no matplotlib/SQL). SATELLITE = NESDIS/STAR
  CDN direct sized JPEGs (`cdn.star.nesdis.noaa.gov/GOES{sat}/ABI/{CONUS|FD|SECTOR/<code>}/{GEOCOLOR|02|13|09}/
  <WxH>.jpg`); product default `geocolor` (day/night blended); GOES-East=GOES19, West=GOES18 (operator-updatable
  per bird; GOES16/17 URLs redirect). RADAR = IEM `radmap.php` for ALL modes (national `sector=conus`,
  regional/station `bbox`), which returns a COMPOSITED, labeled PNG. FINDING (design-doc correction): the raw IEM
  rasters (`4326/USCOMP`, `.../ridge/<SITE>`) are BARE reflectivity -- no basemap/legend, unusable to a VLM, same
  failure as GeoServer WMS -- so they are NOT used; `radmap.php` is the only composited path (verified by viewing
  images). Station-aware: `nearest_radar()` over `radarsites.py` (160 WSR-88D sites from IEM NEXRAD geojson,
  regenerate via `scripts/build_radarsites.py`) + a 150 km guard; beyond it, degrade to regional then national and
  SAY SO in the receipt (ETAR->national verified). Tool enums GENERATED from the catalogs (no drift, like get_map).
  Station->lat/lon via new `awc.station_latlon()` (exact ICAO; resolves major airports AND OCONUS -- climo's
  `station_meta` strips the K and KMSP collides with a TDWR sid). SATELLITE is station-aware too:
  `imagery.satellite_region_for_latlon()` routes a station's lat/lon to the covering GOES sector (17 regions
  incl. added umv/cgl/nr; tightest-sector -> CONUS-by-longitude -> full-disk; OCONUS returns None -> honest
  "no GOES coverage, Meteosat/Himawari deferred" feedback), so `get_imagery(kind=satellite, station=...)` needs
  no region guess -- Gemma verified using it for both sat+radar (lands KLSV on pacific_southwest first try).
  `scripts/fetch_imagery.py` = review/cache driver; `scripts/test_imagery_tool.py` = Gemma reasons over sat+radar.
  TOOLS now 9. Deferred: loops, OCONUS Meteosat/Himawari satellite (fast-follow), advanced products.

- CLIMATOLOGY TOOL (`climo.py` + `store` climo_* tables + `get_climo`, this session): a station-climatology
  lookup so a forecast anchors to what is TYPICAL, not just the last 24h of obs.
  KEY DESIGN: the raw multi-year history fetched to BUILD climo is THROWN AWAY -- only
  the `climo_*` PRODUCT rows persist -- so pouring 20 Julys of history into DuckDB can't leak into `obs` and
  stale-anchor the "now"-relative read tools (the obs-leakage class again). `climo.py` is the orchestrator
  (sibling to `iem.py`; NO duckdb/SQL): `station_meta` (IEM station JSON; strip the leading K for US ICAOs;
  fixed std offset = min(Jan,Jul) via zoneinfo), `ingest_history` (per-(year,month) `iem.load` into a
  `tempfile` SCRATCH DuckDB with a +/-1-day buffer + 429 backoff), `build` (one call: scratch ->
  `store.rebuild_climo` -> scratch deleted). ALL SQL stays in `store.py`: `init_climo_schema` (3 grain-matched
  wide tables climo_meta/climo_monthly/climo_hourly, SEPARATE from init_schema -- build-path only),
  `rebuild_climo` (ATTACH scratch read-only; one txn DELETE + INSERT..SELECT, idempotent per-month),
  `climo_meta`/`climo_month`/`climo_hours` readers. Month membership is LOCAL
  (`obs_time + INTERVAL offset HOUR`), hour key stays UTC; temps use ALL obs, frequencies ROUTINE-only.
  `get_climo(station, month?)` dispatched in the DB branch (read-only conn); `month` defaults to the latest
  ob's month; missing/unbuilt month -> feedback naming `scripts/build_climo.py`; NO `ToolResult.window`
  (climatology is not a time window). `_fmt_climo` renders a compact TYPICAL-conditions text product.
  `config.climo_start_year`/`climo_end_year` = 2006/2025 (last COMPLETE year = leakage guard). Built + VERIFIED
  for KLSV July (1 climo_meta + 1 climo_monthly + 24 climo_hourly rows; ruff clean); model-in-the-loop
  `scripts/test_climo.py` PASSED -- Gemma picked get_climo (not query_obs), read TX mean 40.5C (p10/p50/p90
  band), the N-morning -> S-evening wind flip, TS 0.7% peaking 04Z, fog 0% -- all inside the plan's external-
  sanity tolerances. Log: `logs/climo_KLSV_20260709-151921.md`. Drivers: `scripts/build_climo.py` (CLI build +
  `--check` recompute) and `scripts/test_climo.py`. ALL climo exit criteria now PASS (2026-07-10):
  `--check` recompute (tx_mean/pct_ts/pct_cig cells match SQL exactly; wxcodes-vs-regex TS 195/195), idempotency
  (two deterministic builds identical, stats within 1e-9), obs-untouched (persistent obs 265->265, zero KLSV
  history leaked), and all guards (read-only rejects writes, quoted/out-of-range month coerces+clamps, unknown
  station -> feedback, no ToolResult.window). ONE fix landed to reach idempotency: record dates used
  `arg_max`/`arg_min`, which pick an ARBITRARY day when several tie at the extreme (KLSV TX 47C hit multiple
  days) -> non-deterministic date across builds, undermining ship/hash reproducibility; replaced with an
  ordered-list pick `(list(local_day ORDER BY temp_c DESC/ASC, local_day ASC))[1]` = earliest tied day (record
  VALUE unchanged; dates now deterministic by construction). v1 DONE; still uncommitted (climo + imagery land
  together in the working tree) -- build any additional months the target station needs before/after commit.

- TAF WORKSHEET SEAM (`worksheet.py` + `submit_taf_worksheet`/`get_current_taf`/`check_taf` + store tables,
  this session): the pre-emit REASONING artifact (docs/taf_worksheet_design.md Milestone 1). A typed
  `TafWorksheet` the agent fills and submits through its own validation sink BEFORE emit_taf, to fix the two
  systematic KLSV misses (dewpoint-read-as-TN; MSLP->inHg re-converted inconsistently) by making those cross-
  checks first-class (`sanity_checks`) and to decompose the task so a weaker model CONVERGES instead of
  ruminating. `worksheet.py` is a sibling of tafgen: 12 pydantic sections (task/data_review/current_state/
  forecast_drivers/hazards/forecast_timeline/sanity_checks/taf_strategy/uncertainty/final_assessment/
  model_run_verification + meta) with guardrails (enum/ICAO reject impossible values; `BeforeValidator`
  lower-cases loose enums like `Moderate` per the emit-arg-quirks lesson), a semantic `validate()` returning
  SECTION-PREFIXED findings (never raises), and a `worksheet_guide()` (worked example + flattened field guide)
  modeled on `tafgen.emit_taf_guide()`. NO SQL/matplotlib/network. Key decisions: single validated sink call
  (not section-by-section; two-chunk fallback deferred until a run shows the submit turn truncating); findings
  are `blocking_findings()`-filtered so `model_run_verification:` stays ADVISORY even in required mode (it has no
  runtime backing until `get_model_run_verification`, Milestone 2); `validate()` runs the SAME comprehensive
  checks in advisory + required (mode only governs whether the driver GATES emit_taf). Config: `worksheet_mode`
  (off|advisory[default]|required), `evidence_mode` (off|key_claims[default]|strict), `persist_worksheets`.
  Three new tools in `tools.py`: `submit_taf_worksheet` (sink, schema = `TafWorksheet.model_json_schema()`,
  attaches the accepted worksheet + findings to `ToolResult`), `get_current_taf` (wraps `awc.fetch_taf` +
  `tafparse.render` for the official-TAF comparison), `check_taf` (wraps `tafgen.validate` as a dry-run;
  the emit_taf schema-error+hints block was factored into shared `_taf_schema_error`). `run_tool` gained
  `evidence_ids=` so the sink RESOLVES evidence_refs (not just presence). Persistence in `store.py`:
  `taf_worksheets` + `taf_worksheet_evidence` (all worksheet SQL here; `init_worksheet_schema` +
  `insert_worksheet` one-txn idempotent replace + readers). Driver `scripts/test_worksheet_agent.py` owns the
  agent-loop plumbing (migrates to a future agent.py): EVIDENCE THREADING (each data-tool receipt tagged
  `[evidence_id: ev_NNN]`, the id set passed to the sink), the MODE GATE (required refuses emit_taf until a
  worksheet passes), and persistence of the final worksheet+evidence+TAF. Self-test `scripts/test_worksheet.py`
  (no model/network): 19/19 -- guardrail rejects, enum coercion, empty-worksheet section coverage, MRV/blocking
  split, dangling change_group->timeline ref, evidence presence+resolution, byte-stable guide, both sinks, store
  round-trip. `tafgen` self-test still 9/9 (no regression from the shared-helper refactor); ruff clean.
  VERIFIED LIVE (MiniMax, advisory, `logs/worksheet_agent_KLSV_20260710-103022.md`): the model corrected a
  schema error -> 3 blocking semantic findings -> CLEAN worksheet -> emit_taf (AFMAN caught a 6h validity slip)
  -> CLEAN 30h TAF (proper diurnal wind cycle, SKC 9999, per-group QNH, TX/TN); 12 evidence ids threaded and
  key_claims RESOLUTION passed; worksheet persisted. CAVEATS (environmental, not worksheet bugs): the run loaded
  0 obs (valid 072300Z is now >48h past AWC's window) and get_climo errored (throwaway DB has no climo tables),
  so the model reasoned off live map/point-forecast/current-TAF and the sanity_checks TX/TN-vs-observed demo was
  thin; the `required`-mode gate is coded + stub-checked but NOT yet exercised live. NOT COMMITTED yet (lands
  with the climo+imagery working-tree changes). Two follow-ups if wanted: a live `required`-mode run, and a
  recent-valid-time run (obs + built climo) to exercise sanity_checks with real observed data.

## NEXT SESSION -- pick up here (paused 2026-07-19)

### SESSION 2026-07-19 SUMMARY (all UNCOMMITTED in the working tree -- commit as one model-data commit)
Worked the live-run health check + the GRIBStream credit-gated confirmations + a scheduling redesign.
Details in the blocks below; the headline items:
- FIRST BENCHMARK RESULTS pulled (human 85.7 > model 83.1 > persistence 81.1; MiniMax ablations; per-
  element + by-station gaps) -- see the block just below.
- GRIBStream confirmations DONE (hazard/model-state/flow-relative/OCONUS/agent-loop) + IFS confirmed
  reachable (3-hourly grid) -- see the GRIBSTREAM section.
- FIVE code changes (the working-tree diff): `store.insert_model_data` bulk pandas insert (~1000x),
  `collect._system_prompt` names the get_model_* tools (the model now calls them), `tools._fmt_hazard_scan`
  default valid-time fix, `modeldata._applicable_models` OCONUS model-drop, `scripts/probe_ifs.py` 3h-grid
  fix. Plus the RUN-CADENCE REDESIGN: new `scripts/archive_model_data.py` + `schedule.py` Phase 1b removed
  + pandas moved to a runtime dep (pyproject/uv.lock). ruff clean; test_modeldata 63/63.
- UNCOMMITTED: CLAUDE.md, pyproject.toml, uv.lock, scripts/{collect,probe_ifs,schedule}.py,
  src/forecaster/{modeldata,store,tools}.py, scripts/archive_model_data.py (new). docs/ (gitignored):
  gribstream_model_data.md + pi_setup_log.md updated. NEXT: commit these (one model-data commit); the
  model-data tier is still GATED OFF (MODEL_DATA_ENABLED=false) so nothing bills until enabled.

### FIRST BENCHMARK RESULTS -- mid-run health check + scoring signal (2026-07-19, ~3 days into the live Pi run)
Full collect->score loop VERIFIED HEALTHY on real data (sanity check PASSED). Snapshot the Pi DB via the
harvest flock pattern and query read-only; ~162 evaluations scored, ~166 pending (not-yet-elapsed windows).
Headline combined TAFVER (mean over scored evals): HUMAN 85.7% > MODEL 83.1% > PERSISTENCE 81.1% -- the
canonical human>AI>persistence ordering, a genuine first signal. Fatals: all 45 were the single 2026-07-17
402 burst; ZERO since (max-parallel 3->2 held). ~66 incomplete (~15%, all no-emit) cluster on heavy issue
hours (11Z/14Z/03Z/19Z) = cells SIGKILL'd by schedule.py's 30-min timeout when 18 cells fire/hour -- no
leakage, just lost matrix cells; RAISE the per-cell timeout or spread heavy hours before round 2 for stat power.
- MiniMax ABLATIONS (4 cells, ~30 scored each; single-point gaps directional not yet significant):
  control (ws+prior-TAF, t0) 84.4% > nows 83.4 > temp02 83.0 > NOTAF 81.9. PRIOR-TAF ACCESS is the dominant
  factor (-2.5, drops ~to persistence; wrecks ceiling continuity 86.5->75.7). temp0 beats temp0.2 (-1.4).
  Worksheet buys only ~1 TAFVER pt but HALVES output tokens (nows 8.7k vs ctl 14.5k) -- cost/benefit Q for
  round 2 (worksheet's value may be convergence/reliability, not raw score). Factor rank: prior-TAF >> temp ~ worksheet.
- MODELS (control cell): Gemma 83.3% ~= MiniMax 84.4% at similar cost = efficiency win (weak wind_dir 59.1);
  Kimi 81.4% but only 40% clean-emit, ~2x tokens, ~12 steps/20 tool-calls = the gather-loop/rumination that
  got it dropped from round 2 (n=10 scored, soft).
- PER-ELEMENT (human/model/persist): human wins 5/7, dominates altimeter (99.3) + present_weather (31.6 vs
  22.1, +9.5 = where forecaster judgment separates, though hard for all). MODEL BEATS human on VISIBILITY
  (+5.8) and ties gusts (persistence wins gusts 89.1 -- dry-ridge gusts persist). present_weather is the
  universal weak spot (~14-32%). Human has MOST busts / LOWEST in-spec (5.89 / 63.3%) yet highest TAFVER =
  aggressive-but-accurate vs persistence's do-nothing (persistence 2.26 trig / 70.9% in-spec).
- BY-STATION human-model gap: human leads 7/10 (widest KWRI +6.3, PAED +5.8, KDMA +5.0); MODEL BEATS human at
  KMIB (-3.1, solid 18/18), PABI (-4.8 coastal AK), KRCA (-3.0 thin n=5). Persistence reframes difficulty:
  at KWRI/KSSC/KFTK persistence >= human (low-skill stable regimes); at RJTY/PABI/KVBG persistence far below
  (72-74%) = high-value maritime/AK sites where the model is already competitive-to-better. That
  human-vs-model-vs-persistence per-station spread is the difficulty-mining signal for round-2 target picks.
  Watch thin pairings (KFTK/KVBG/KRCA human n=5-6). Queries: scratchpad this session; re-derive via harvest snapshot.

### GRIBSTREAM MODEL-DATA SUBSYSTEM -- BUILT this session; credit-gated confirmations remain
The whole model-data subsystem is CODE-COMPLETE, offline-tested (`scripts/test_modeldata.py` 63/63),
ruff-clean, and partially live-verified. Authoritative plan + full remaining-work list:
`docs/gribstream_model_data.md` (gitignored). Gated OFF by default so the live Pi is untouched
(`MODEL_DATA_ENABLED=false`, `MODEL_DATA_FLOW_RELATIVE=false`). ALL UNCOMMITTED (rides the working
tree with the climo/imagery/worksheet/scoring/spatial work).
- BUILT: Phase A-E (multi-coord `gribstream.fetch_points`; `store.model_data` archive; `modeldata.py`
  orchestrator + `prefetch`/`prefetch_many`; 4 tools get_model_state/get_hazard_scan/
  get_model_verification/get_nearby_model_data; collect.py/schedule.py integration) + IFS scaffold
  (disabled) + flow-relative STEERING grid + batched roster-wide prefetch.
- KEY FACTS: points are FREE <=500 (credits scale with HOURS x variables, not points); leakage guard
  for model FORECASTS is only run<=issue via `asOf` (no read-cutoff); HRRR MSLP = `MSLMA`; flow
  orientation = deep-layer 850/700/500 GFS steering mean at the ISSUE ANCHOR (not surface, not climo,
  not forecast-mean); two-pass prefetch keeps prefetch and collect's copy in lockstep.
- VERIFIED FREE (no GRIBStream credits): surface prefetch (KWRI, 494 cr, earlier) + get_model_state/
  get_nearby_model_data render; and `get_model_verification` end-to-end using the cached archive + FREE
  AWC obs (all 3 models matched 4 real hours; surfaced the GFS 12Z warm/dry bias). Offline tests cover
  archive, formatters, collect copy path, grid/flow/batch/IFS, steering math + copy-reproducibility.
- CONFIRMATIONS DONE (2026-07-19; all against a scratch DB, KWRI archive cached from killed runs
  so most reruns cost 0 credits; PAED charged 494):
  1. HAZARD scan -- CONFIRMED. get_hazard_scan renders real pressure-level icing (T/RH/CLW x GFS+HRRR)
     + turbulence (CAPE/CIN/omega/shear/SRH). BUG FOUND+FIXED: default valid-time picked the surface
     6h back-tail (04Z, no hazard data) -> empty render; now filters to hazard-bearing entries
     (cape/t650) before the default pick (`tools._fmt_hazard_scan`).
  2. AGENT-LOOP -- CONFIRMED w/ FIX. First run: Gemma emitted a clean TAF but called NONE of the
     get_model_* tools (used get_point_forecast instead). ROOT CAUSE: `collect._system_prompt` never
     NAMED the model-data tools (same failure as spatial-awareness). FIXED: prompt now names the four
     get_model_* tools + a "anchor TX/TN, QNH trend, gusts to model guidance; use hazard_scan for
     convective/icing; use model_verification to weight the less-biased model" nudge, gated on
     --model-data. RE-RUN CONFIRMED: same KWRI cached archive, Gemma went 0/4 -> called get_model_state
     (converged clean, 5 steps); did not call hazard_scan (reasonable for a dry-ridge KWRI). Prompt-
     naming is what makes the model use the tier.
  3. OCONUS degrade -- CONFIRMED w/ FIX. PAED: GFS 171/171 non-null; HRRR AND NBM 0/171 (GRIBStream's
     NBM is CONUS-only too, NOT "GFS+NBM" as the doc assumed). Degrade was clean (renders `--`) but
     SILENT + BILLED (494 cr, ~2/3 on all-null HRRR/NBM). FIXED: `modeldata._applicable_models` drops
     the CONUS-only models (hrrr,nbm) when no coord is in the CONUS bbox, with a note -> OCONUS
     stations fetch GFS-only (saves ~2/3 credits). Mixed CONUS+OCONUS batches keep all models.
  4. IFS -- CONFIRMED reachable (2026-07-19, after checking gribstream.com/models/ifsoper). The earlier
     "no data" was NOT availability: IFS steps are 3-HOURLY (0-144h every 3h), so my probe's off-grid
     valid times (19Z/04Z) returned 0 rows. On the 00/03/.../21Z grid all 11 vars return finite (KMSP,
     run 18Z, 11 cr); slug `ifsoper` + names `2t/2d/10u/10v/msl/tcc @ sfc`, `t/r/u/v/w @ 'pl <hPa>'` all
     correct; tcc is a 0-1 FRACTION (0.609). `scripts/probe_ifs.py` FIXED to snap to the 3h grid. STILL
     DISABLED -- enabling still needs: (a) a PER-MODEL time grid (IFS on 3h/6h, not the 2h surface grid,
     or it bills null rows at non-aligned even hours -- same waste class as OCONUS), (b) tcc *100 in the
     state formatter, (c) add `ifsoper` to `gribstream.MODELS` + flip `modeldata._IFS_ENABLED`. IFS is
     GLOBAL so it is correctly NOT in _CONUS_ONLY_MODELS (kept OCONUS, unlike HRRR/NBM).
  5. Flow-relative -- CONFIRMED. Steering probe computed westerly; 6 oriented upstream points
     (u251/u281/u311 @ radii 20/30) archived beside the 12x3 base grid.
  PERF FIX (found via the slow prefetch): `store.insert_model_data` did per-row INSERT ON CONFLICT
  (~13ms/row, 96s for 7182 rows, minutes for a full cycle -- CPU-bound, bad on the Pi). FIXED: register
  the batch as a pandas DataFrame + one set-based `INSERT ... SELECT ... ON CONFLICT` (~0.10s; full 48k-
  row KWRI prefetch 9min+ -> 5.45s, ~1000x). pandas moved dev-group -> runtime dep in pyproject (uv
  sync'd); executemany fallback if pandas missing. test_modeldata 63/63, ruff clean.
  STILL TO DO:
  6. Cost/budget guard, then `MODEL_DATA_ENABLED=true` on the Pi (SEPARATE credit line from the LLM
     providers; ~500 cr/station surface / ~1000 with hazards; OCONUS now GFS-only ~1/3; batching lowers
     multi-station cycles).
  7. COMMIT the subsystem + these fixes (ideally its own commit) once blessed.

- RUN-CADENCE REDESIGN (2026-07-19, IMPLEMENTED -- decouples model-data pull from forecast time).
  OLD: schedule.py Phase 1b prefetched per CYCLE-EVENT (as_of=issue), ~18/day; each event anchored
  its valid-time grid on its own issue time, so forecasts using the SAME run re-fetched on offset
  grids (full credits each, cache can't dedupe). NEW: pull on the MODEL-RUN cadence -- ONE batched
  pull of the WHOLE roster (~420 coords, free <=500 => ~1 chunk) captures each model's freshest run;
  forecasts just COPY the archive (0 cr). WHY it works with no data-layer change: `_pivot_series`
  already keeps the LATEST run per valid_time, and copy_model_data reads by coord -- so a multi-run
  archive serves "latest run <= issue" automatically (leakage-safe live; a historical-replay run<=issue
  filter in copy_model_data is a deferred follow-up). CHANGES: new `scripts/archive_model_data.py`
  (batched all-roster pull, 48h, gated on MODEL_DATA_ENABLED unless --force; ~1551 cr/fire batched =>
  ~6k/day at 4x); schedule.py Phase 1b REMOVED (cells still get --model-data and copy); Pi cron adds
  `archive_model_data.py` at 05/11/17/23Z + `--model-data` on the scheduler (docs/pi_setup_log.md).
  Verified: schedule --dry-run shows no prefetch line; archive --dry-run ~1551 cr; gated no-op; live
  KWRI-only run archived 42k rows (charged 1077 -- a fresh run, not my cache). ruff clean.
  ** FLAG FOR v2 DISCUSSION (owner will decide before implementing v2): HRRR + NBM update HOURLY but
  this job snapshots them only ~4x/day, so a forecast can read an HRRR/NBM run up to ~6h old -- fine
  for GFS/IFS (native 4x/day), a real value loss for the rapid-refresh models. Options: separate
  higher-frequency HRRR/NBM archival job, drop them from the batched roster pull and fetch at forecast
  time, or accept the staleness. NOT decided -- revisit before enabling v2. ** [[gribstream-seam]]
- DEFERRED (only if a run shows the need): multi-DIRECTION upstream extension for an evolving flow
  (fan far points along initial AND later inflow bearings -- NOT a forecast-mean, which is unsound
  across a regime change); IFS hazard bundle (icing t/r + shear u/v/w at `pl` levels); input-pinning
  the OTHER live tools (imagery/maps/soundings) via the same snapshot-and-replay pattern.

### SPATIAL AWARENESS -- built + wired (2026-07-17); PRE-WARM + AVAILABILITY CHECK ahead of v2
The spatial-awareness pair is built, VLM-verified (Gemma), and wired into the collection agent:
- `get_terrain` (`terrain.py`) -- static terrain rose + a LABEL-FREE Esri World_Shaded_Relief map
  (switched off OpenTopoMap: its rasterized town names swamped low-relief fields like KWRI) with the
  nearby airfields PLOTTED at true lat/lon: blue+labeled = fetchable (obs available), violet = context
  (orientation only). Map radius ADAPTS to include far-flung neighbors (e.g. PABI's are 60-85 mi out).
- `get_nearby_obs` (`tools.py`) -- SELECTIVE now: pass `stations=[...]` (chosen off the map) or fall
  back to nearest-n. Leakage-safe DB read (never live).
- Roster (`neighbors.py`, regen via `build_neighbors.py`): fetchable-5 now carry lat/lon + an
  `AREA_STATIONS` context catalog (all box METAR sites, positions only, capped 40) for the map.
- Collection prompt (`collect.py`) NOW names both tools + a two-step nudge (terrain -> pick upwind
  neighbors -> regional-vs-local) -- WAS the real gap: tools were in the toolset but unnamed in the
  prompt, so the model never used them. `awc._get` empty-body fix: a 204/blank response (station not
  reporting, e.g. KXVW) now returns `[]` not a JSONDecodeError.
- Verified two-step loop: `scripts/test_terrain_nearby_agent.py` (Gemma read the map, selected a
  KVGN/KSMX/KTQS/KLPC cross-section, called the marine layer correctly as REGIONAL).
BEFORE v2 (owner action items, 2026-07-17):
  1. PRE-WARM the terrain tile cache for the whole roster -- `uv run python scripts/fetch_terrain.py`
     (now renders with the tool's real markers + adaptive radius, so it caches EXACTLY the tiles
     get_terrain needs; required for the air-gapped SuperCloud + polite to Esri; drops a review JPEG
     per station).
  2. NEIGHBOR-AVAILABILITY CHECK -- audit how many of each station's 5 fetchable neighbors actually
     report live obs (KXVW/KVBG surfaced that some don't). Tells us where get_nearby_obs will be
     sparse (inland AK/Japan fields especially); the empty-body fix makes it degrade cleanly, but we
     want the coverage map before scaling.

### ROUND 2 PREP -- build `scripts/spend_report.py` BEFORE the second experiment round (owner deferred the build 2026-07-17)
Pure READ-ONLY spend rollup over the `runs` table, which ALREADY persists `prompt_tokens` + `completion_tokens`
+ `model` per run -- apply a per-`(model, provider)` price map (USD/1M input+output) and sum spend by model /
provider / station / day with a running total; runs RETROACTIVELY over round-1 rows (no schema migration needed to
report the existing data). Needs the `provider` column that lands with the multi-provider matrix change (provider is
otherwise only implicit in the base_url); until then key the price map on `model` alone. This is the ACCOUNTING layer.
The SAFETY complement is separate + simpler: a HARD per-key credit cap at each provider (OpenRouter/DeepInfra support
per-key hard limits + usage alerts) so a runaway loop or bad model id can't overrun budget -- the round-1 402 halt
(2026-07-17, Together credits exhausted ~0902Z, 37 runs fatal) was that guardrail firing bluntly.
WHY NOW: round 2 widens the model+provider spread, so per-cell spend visibility must exist before scaling. Decided
this session (2026-07-17): Kimi DROPPED from schedule.py MATRIX (worst $/clean-TAF ~16x MiniMax, 42% clean rate;
revisit with Kimi K3). Cheap serverless VISION+TOOLS candidates surveyed (per-run cost at the measured 140k-in/
12.5k-out profile): Qwen3-VL-32B @Alibaba ($0.019), Llama-4-Maverick @DigitalOcean ($0.046, the SUPPORTED successor
to deprecation-flagged Scout), GLM-4.6V @Z.AI ($0.053, chart-optimized), Gemma-4-31B @DeepInfra ($0.023, same weights
~1/3 the Together price). KEY WORKLOAD FACT: input dominates (140k in vs 12.5k out), so INPUT price sets the ranking
-- Gemma @Together ($0.067/run) is actually PRICIER than MiniMax M3 @Together ($0.057) despite being "smaller".
Provider seam options: OpenRouter (one key/one balance, provider-PINNED for reproducibility, ~5% markup) vs direct
providers (DeepInfra+Alibaba+Z.AI, cheapest but 3 balances to keep funded -- the round-1 outage failure mode). The
multi-provider matrix change also needs `Cell` to carry an optional `(base_url, key)` override, and `ping_models.py`
to smoke-test a given base_url+model for the vision AND tool-call path per provider (same weights differ by backend).

### MODEL x PROVIDER x COST REFERENCE (sampled 2026-07-17; RE-VERIFY before committing -- prices move weekly)
Serverless, OpenAI-compatible, VISION + TOOL-CALLING only (the agent's two hard requirements). Prices USD per 1M
tokens (input / output). `$/run` = cost at THIS benchmark's measured token profile (140k input + 12.5k output =
median of 90 clean round-1 runs), UNCACHED list price: `$/run = 0.140*in + 0.0125*out`. Sources: Together account
catalog + OpenRouter per-provider endpoint API (`/api/v1/models/:slug/endpoints`). Cheapest TOOLS-ENABLED + vision
route per model (lower is better):

| Model | Cheapest tools+vision route | In | Out | $/run | $/1k runs | Note |
|---|---|---|---|---|---|---|
| Qwen3-VL-32B | Alibaba (DashScope) | 0.10 | 0.42 | 0.019 | 19 | MiniMax-tier VLM, cheapest peer, diff lab |
| Llama-4-Scout | Groq | 0.11 | 0.34 | 0.020 | 20 | DEPRECATION-flagged; Maverick is successor |
| Qwen3-VL-8B | Alibaba | 0.12 | 0.45 | 0.022 | 22 | small-model floor for a size ladder |
| Gemma-4-31B | DeepInfra (tools) | 0.13 | 0.38 | 0.023 | 23 | same weights as now, ~1/3 the Together price |
| Qwen3-VL-30B-A3B | Alibaba | 0.13 | 0.52 | 0.025 | 25 | MoE variant, more active reasoning than 32B |
| Llama-4-Maverick | DigitalOcean | 0.25 | 0.87 | 0.046 | 46 | SUPPORTED Scout successor; 128-expert MoE |
| GLM-4.6V | Z.AI (or Novita, tied) | 0.30 | 0.90 | 0.053 | 53 | native tool-call VLM, ChartQAPro-benchmarked |
| MiniMax M3 | Together / DeepInfra / many | 0.30 | 1.20 | 0.057 | 57 | CURRENT ANCHOR (428B MoE, 67% clean, keep) |
| Gemma-4-31B | Together (tools endpoint) | 0.39 | 0.97 | 0.067 | 67 | CURRENT route; DeepInfra cheaper for SAME model |
| Kimi K2.7 | Together | 0.95 | 4.00 | 0.183 | 183 | DROPPED round 2 (worst $/clean, 42% clean) |

PROVIDER ALTERNATIVES (when a model has several routes): MiniMax M3 = 0.30/1.20 on ~every provider (Together,
DeepInfra, Novita, Minimax-direct, Parasail, AtlasCloud). Gemma-4-31B tools routes, cheapest-first: OpenInference
0.10/0.35, Venice 0.12/0.36, Chutes 0.12/0.37, DeepInfra 0.13/0.38, SiliconFlow 0.13/0.40, Novita 0.14/0.40, then
Together 0.39/0.97 (Together ALSO has a 0.28/0.86 Gemma endpoint but it is tools-OFF). Qwen3-VL-30B-A3B: Alibaba
0.13/0.52, DeepInfra 0.15/0.60, Novita 0.20/0.70. Llama-4-Maverick tools routes: DigitalOcean 0.25/0.87, Parasail
0.35/1.00, Google/Vertex 0.35/1.15 (Vertex needs GCP service-account auth -> BREAKS the base_url+key seam; avoid).
Llama-4-Scout tools: Groq 0.11/0.34, Google 0.25/0.70. NOTE both Llama-4 endpoints on DeepInfra are tools-OFF.

TOGETHER serverless (vision+tools) per owner's dashboard 2026-07-17: Inkling (1.00/4.05, 404s -- see ping_models),
MiniMax M3 (0.30/1.20), Kimi K2.7 (0.95/4.00), Kimi K2.6 (price unconfirmed), Gemma-4-31B (0.39/0.97), Qwen3.5-9B
(0.17/0.25), Pearl Gemma-4-31B (0.28/0.86). These 7 are the ONLY serverless vision+tools models on the Together
account -- Qwen3-VL / Llama-4 / GLM-4.6V are NOT available there serverless (dedicated-only or absent), which is
the reason to add a provider.

CAVEATS baked into these numbers: (a) INPUT dominates this workload (140k in vs 12.5k out) -> input price sets the
ranking; output rate barely moves the total (this is why Gemma@Together > MiniMax M3 despite Gemma being smaller).
(b) CACHING narrows the gaps -- each run re-sends its growing context across ~5 steps, so most of the 140k bills at
the cached-input rate (typically 25-50% of list); absolute costs drop, ordering holds; per-provider cached rates NOT
yet pulled. (c) TOOL support is PER-ENDPOINT -- some routes are vision-yes/tools-NO (the DeepInfra Llama-4s, the cheap
Together Gemma tier); only tools-enabled endpoints are tabled above. (d) Same open weights on different backends/
quantization -> benchmark scores NOT strictly comparable across providers; PIN one provider per model + record it in
run provenance. Visual version of this table published as an artifact 2026-07-17 (private; re-derive from here if the
link is lost).

### TAF ARCHIVE EXPANSION -- 53 more AF/Army human-TAF sites for wide TAFVER difficulty-mining (identified 2026-07-17)
CONCEPT: the human-TAF ARCHIVE net is SEPARATE from the model-run roster. Archiving human TAFs only needs a
fetchable AWC bulletin (NO BUFKIT gate) and is CHEAP (poll + score vs obs, ZERO LLM cost). So cast a WIDE net,
compute TAFVER per SITE and per HOUR-OF-DAY (tafamend rule-hours + tafver hourly scores already emit hourly), rank
difficulty, THEN point the billed model matrix at the hard subset. Archive-only sites NEVER enter the model matrix.
Confirmed live against AWC 2026-07-17 via the military marker QNH____INS (+ ~30h validity; civil/FAA TAFs lack it).
53 NEW sites (38 AF, 15 Army) on top of the 10-station roster, grouped by the difficulty regime to mine:
- CONVECTIVE/severe (TS timing+initiation, the hardest): KWRB KCBM KBIX KVPS KHRT KPAM KMCF KBAD (AF SE/Gulf);
  KLSF KFBG KOZR KHOP (Army SE/Gulf); KTIK KLTS KDYS KCVS KSZL KOFF KSKF KRND KDLF (AF Plains/S-TX); KFRI KGRK
  (Army Plains); KLSV KLUF KHMN (AF SW monsoon); KFHU (Army SW monsoon).
- FOG/marine layer/low cig-vis (category busts): KBAB KSUU (AF CA); EGUN EGUL ETAR (AF Europe); KLFI KDOV KADW
  (AF coastal); KFAF KGRF (Army).
- WINTER/northern (frozen precip, rapid transitions): KHIF KFFO KBLV (AF); KMUI (Army); + Offutt/Ft-Riley above.
- TERRAIN/mountain (upslope): KMUO KEDW (AF); KFCS (Army); ETIC ETOU (Army Europe).
- TROPICAL/typhoon: PGUA RODN (AF); PHHI (Army). PACIFIC/Asia monsoon: RKSO RKJK (AF); RKSG (Army).
CIVIL-FORMAT ONLY (co-located civilian bulletin, no QNH INS -- SKIP): KABQ LIPA ETHA.
NO TAF in the 17Z snapshot -- RE-CHECK (major bases issue TAFs normally; a one-shot probe misses between issuances
or in reduced-ops; the poller confirms over days like cycle_provisional): KVAD-Moody PAEI-Eielson RJSM-Misawa
PHIK-Hickam KGFA-Malmstrom KFSI-FtSill KFLV-FtLeavenworth KDAA-FtBelvoir KWSD-WhiteSands.
IMPLEMENTATION: add an archive-only station list to stations.py (SEPARATE from STATIONS so the scheduler -- which
iterates STATIONS -- structurally cannot bill them) + `poll_icaos()` = roster + archive; poll_tafs.py iterates
poll_icaos(); schedule.py UNCHANGED. COMPANION NEEDED for actual difficulty-mining: a scoring pass that scores each
archived HUMAN TAF standalone vs obs (via `--backfill iem` -- IEM serves military METARs), since archive-only sites
have NO collect.py evaluation row and today's --pending scorer only scores model-subject evaluations. See the (b)
poller sketch drafted this session (2026-07-17); NOT yet applied to the files.

### GO LIVE ON THE PI -- built + deployed + idle; 3 steps to start collecting (2026-07-16)
The whole M4 automated-collection system is BUILT, COMMITTED + PUSHED, and DEPLOYED to the Pi.
The Pi is idle and clean (no crons running, not billing). Pure tests green (tafstate 40, runlog 15,
agent 30, worksheet 19, score_pending 21); ruff clean. Full command audit: `docs/pi_setup_log.md`.

WHAT'S BUILT THIS SESSION:
- **Roster** (`stations.py`): 10 MILITARY, 8-hourly, 30h, military-format (TX/TN + meters vis + QNH INS)
  airfields, each with its `cycle` (UTC issue hours): KWRI 02/10/18, KMIB 01/09/17, KRCA 03/11/19,
  KSSC 07/15/23, KDMA 03/11/19, KVBG 06/14/22 (prov), KFTK 03/11/19 (prov), PAED 05/13/21,
  PABI 06/14/22 (prov), RJTY 05/13/21. Discovered via `scripts/probe_bufkit_stations.py` /
  `probe_stations_extended.py` (BUFKIT gate) + `probe_taf_cycles.py` (OGIMET cycle confirm). Dropped:
  6-hourly civil-format fields (no TX/TN) + BUFKIT-but-no-AWC-TAF + irregular KGRK.
- **Collector** (`scripts/collect.py`): one matrix cell/run. Obs BANKED into the benchmark DB once
  (truth accumulates as a side effect of collection), then the pre-cutoff back-window COPIED into a
  throwaway per-run DB with cutoff enforced in SQL (`store.copy_obs`); climo COPIED in (`store.copy_climo`);
  leakage-safe `get_previous_taf` (from our archive, prior-cycle only) instead of `get_current_taf`
  (dropped); mandatory worksheet (`--mode required`); ample reasoning (max_steps 24, max_tokens 16000) +
  agent truncation-recovery guard (`_LENGTH_NUDGE`). Persists a PENDING evaluation.
- **Scheduler** (`scripts/schedule.py`): one hourly cron consults `stations.cycle`, fires due stations'
  matrix in parallel (bounded `ThreadPoolExecutor`, `MAX_PARALLEL` default 2; cron uses `--max-parallel 3`
  for the Pi 5). MATRIX = 6 cells/event: 3-model CONTROL (Kimi `moonshotai/Kimi-K2.7-Code`, MiniMax
  `MiniMaxAI/MiniMax-M3`, Gemma `google/gemma-4-31B-it`; worksheet required, temp 0, prior TAF) + 3 MiniMax
  ablations (temp 0.2 / no-worksheet / no-prior-TAF). Inkling `thinkingmachines/inkling` COMMENTED OUT --
  404 on Together (verify via `scripts/ping_models.py`). ~180 runs/day.
  FUTURE: add Inkling and Kimi K3 to the tested-model set (Inkling once it's reachable on Together; Kimi K3
  as the successor to the current Kimi K2.7 control). Verify both model strings via `scripts/ping_models.py`
  before wiring them into the matrix.
- **Poller** (`scripts/poll_tafs.py`): archives new/amended human TAFs every 5 min (idempotent by content
  hash; shows bulletin type). Feeds `get_previous_taf`.
- **Provenance**: seed (config.llm_seed=1337), temperature, max_tokens, base_url, toolset_hash, served_model,
  system_fingerprint, hashed config_id on every `runs` row. Transcript paths stored RELATIVE to the DB dir
  (`runlog`) so DB + `data/runs/` are portable together (`read_transcript(path, db_path=)` resolves).
- **Scoring `--pending`** (`scripts/score_taf.py`, `test_score_pending.py` 21/21): flips elapsed pending
  evaluations, `--backfill iem` fills obs gaps.
- **climo 503 fix**: builds now retry 5xx + connection errors, not just 429. LESSON: never run two climo
  builds at once -- IEM 503s (server-side overload, not per-IP). Build single-client.
- **Harvest** (`scripts/harvest.sh`): Pi -> laptop `data/benchmark/` (flock-consistent DB snapshot + rsync
  transcripts) -> `onedrive:artificial-forecaster/` (rclone; verified round-trip) -> optional prune of Pi
  transcripts >N days (PRUNE=1). Pull results back a few times/week; code+`.env` never pulled.

PI STATE (`pi@192.168.0.21`, hostname `wx-collector`, Pi 5 / 4-core / 8GB / 109GB free), verified
live 2026-07-16 end of session: UTC timezone; git+uv installed; repo at `7d03f60` -- NEEDS `git pull`
to reach the pushed `1e2c394` ("Bug fixes for the automated runs" = the collect rework + copy_obs +
--pending scorer + relative transcript paths); `uv sync` done; `.env` transferred; models ping 3/3;
crontab EMPTY (confirmed `no crontab for pi`). JULY climo VERIFIED COMPLETE in the DB for all 10
stations (1 climo_monthly + 24 climo_hourly rows each) -- the build log shows only 5 "OK" lines
because it was restarted mid-build after the `Climo fix'` pull; that is a LOG artifact, the DB is
complete. Do NOT re-run the July build.

GO-LIVE STEPS (next session):
1. On Pi: `cd ~/artificial-forecaster && git pull` (gets reworked collect.py / store.copy_obs / relative
   paths / scorer). Confirm HEAD == origin.
2. ONE test run: `~/.local/bin/uv run python scripts/collect.py --station KWRI --model google/gemma-4-31B-it
   --mode required` -- confirm climo_months>0, get_previous_taf served, emit clean, relative transcript path.
3. Install the 3 UTC cron entries from `docs/pi_setup_log.md` (poller `*/5`, scheduler `:02 --max-parallel 3`,
   scorer `35 */6 --pending --backfill iem`). -> LIVE. GET EXPLICIT GO-LIVE OK FIRST (billed + recurring).
4. FIRST-SCORE SANITY CHECK (~31h after the first collection cycle = 30h validity + 1h grace): eyeball
   `logs/score.log` and one `logs/tafscore_*.md` on the Pi -- this is the first time the whole
   collect->score loop runs on REAL data (the scoring self-tests only prove internal consistency).
   Expect: status=scored, coverage ~100% (obs banked by collect's dual-write), subject + human +
   persistence rows in the comparison table.
5. (Optional) `rclone config` on the Pi for continuous Pi->cloud backup (the laptop harvest leaves a
   loss window if the SD card dies between pulls).

CAVEATS: climo is JULY-ONLY (this test ~1 week per owner) -- if collection runs into August, build
August BEFORE Aug 1 (`build_climo.py --station <icao> --months 8`, one station at a time -- concurrent
IEM builds 503). Satellite images ~1MB base64; transcripts ~1-3MB/run -> ~1.3-3.8GB/week (prune keeps
the Pi lean).

### (historical 2026-07-13) TAF SCORING SUBSYSTEM built (M0-M3 of docs/taf_score.md)
Implemented the whole scoring ENGINE from `docs/taf_score.md` (three orthogonal scorers on shared
primitives). All PURE modules (no duckdb/SQL/network/matplotlib/LLM); naive-UTC throughout; ruff clean.
- **M0 shared foundation** -- `src/forecaster/tafstate.py`: absolutizer + forecast-state resolver +
  `opportunities()` + two-view (conservative/union + per-field predominant) hourly truth builder +
  category classifiers (TAFVER ladders AND DAF A4.1 lower-of) + present-weather normalizer +
  `predominant_state`/`conservative_state` view reconstructions. Plus `tafparse.explicit_fields` (which
  fields a group RESTATED vs inherited), the half-open `store.scoring_window` reader, the `tafs` archive +
  `evaluations` spine (`store.init_scoring_schema`), and `scripts/archive_taf.py`. `test_tafstate.py` 37/37.
- **M1 amendment busts** -- `src/forecaster/tafamend.py`: 6 build-now doctrine rules (category lower-of,
  Rule 1 wind, Rule 5 altimeter, Rule 7 TS, Rule 8 TEMPO, Rule 9 BECMG/FM timing) + 3-layer aggregation
  (hourly -> rule episodes -> deduped amendment triggers) + amd-service remark exclusion + persistence
  baseline (`tafstate.persistence_taf`). `test_tafamend.py` 20/20.
- **M2 skill** -- `src/forecaster/tafskill.py`: axis 1 continuous element errors (predominant view,
  prevailing only; wind/cig/vis + per-group QNH + per-TAF TX/TN), axis 2 event contingency
  (`EVENT_CATALOG_V1`, POD/FAR/CSI/HSS via `contingency_scores`, min-cost episode timing), axis 3 ordinal
  MACE; `skill_deltas` benchmark deltas on MATCHED hours only. `test_tafskill.py` 21/21.
- **M3 TAFVER** -- `src/forecaster/tafver.py`: 7 Table A7.1 MOPs (0/1 + fractional PW CSI), anti-averaging
  combined = sum(earned)/sum(available), INITIAL/BECMG diagnostic buckets, A7.2 category accuracy+bias,
  provenance hashes (policy/profile/obs), `fitl_value_added` (refused on any hash mismatch). Provisional-
  policy labeled. `test_tafver.py` 22/22.
- **Driver** `scripts/score_taf.py` (`--scorers tafver,amend,skill` + persistence baseline; markdown report)
  and **`scripts/grade_taf.py`** (NEW): exhaustive HOUR-BY-HOUR audit log -- per-element/per-rule tables that
  sum explicitly to each headline; handles epoch-prefixed METAR lists + a malformed-TX/TN repair. Non-LLM.

### First real grading (KLSV 112300Z TAF vs supplied 33-ob METAR sequence) + 3 bugs it surfaced
Ran `grade_taf.py` on a user-supplied military 30h TAF (3 BECMGs, per-group QNH, turbulence group) + its
METARs. Log: `logs/grade_KLSV_112300Z.md`. Headline: **TAFVER 83.0% (146/176), 5 amend triggers, in-spec
0.57, MACE 0.0** (dry desert -> cig/vis all cat E). The real TAF exposed 3 genuine bugs (all FIXED + each has
a regression test):
1. **TAFVER didn't evolve through BECMGs** -- the baseline opportunity scored the ORIGINAL prevailing for all
   30h. Fixed: baseline resolves to the EVOLVED prevailing (BECMG folded in once complete).
2. **Visibility scored `unresolved`** -- the truth-view reconstructions kept vis in METERS but dropped
   `vis_sm`, and the TAFVER vis classifier keys off statute miles. Fixed `_vis_from_m` in
   `predominant_state`/`conservative_state` (9999 m -> unlimited; else convert via the Table 8.1 seam).
3. **BECMG attribution + double-count** (owner-DECIDED policy change): (a) post-BECMG hours are now
   ATTRIBUTED to the BECMG group, not INITIAL -- `_baseline_segments` splits the timeline at each BECMG's
   END and labels segments by the governing group (AFMAN: post-valid-time the BECMG prevails); (b) the BECMG
   transition window is now **BEST-OF** (the hour is correct if obs matches the OLD prevailing OR the
   "becoming" state -- one row, no double-count) instead of the old DECIDED double-count. `Opportunity` gained
   `role` + `alternate_indices`; TAFVER folds best-of via `_best_of`/`_score_all`. TEMPO/PROB still emit their
   own overlay rows (unchanged; align to best-of later if wanted).

### STORE STATUS: results persistence + --pending BUILT (M4 step 3 DONE, 2026-07-16)
The per-scorer RESULT tables now exist (`store.init_results_schema`: tafver_runs/hourly/summary,
tafamend_runs/rule_hours/events, tafskill_runs/element_rows/event_hours/episodes) with append-only
idempotency (deterministic `scorer_run_id` over evaluation+taf+subject+policy_hash+scorer_version;
identical reruns are no-ops, changed inputs are NEW runs) plus the batch aggregators
(`tafver_points`/`skill_errors`/`skill_cells` -- summed cells/points; division + contingency scores
stay in the scorer modules). `score_taf.py --pending` is the post-validity pass: selects elapsed
pending evaluations (grace default 1h), checks truth coverage (default >=90% of window hours;
`--backfill iem` fills gaps -- IEM DOES serve the military fields for METARs, it is TAFs it lacks),
scores subject + persistence + the paired HUMAN routine TAF (`store.human_taf_for_window`), persists
under one write_lock hold per evaluation, and flips pending->scored (or ->partial with
`--allow-partial`; failed-required-coverage stays PENDING -- the two are never conflated). Evaluation
rows now carry `taf_id` + `scored_at` (migrations bring old DBs forward); provenance (obs_hash via
tafver.obs_hash over the exact scoring_window rows, truth-policy/profile hash via the new shared
`tafstate.stable_hash`, coverage manifest) lands on the spine row. Each scorer module has a
`SCORER_VERSION` const (bump on scored-output-changing fixes). Self-test
`scripts/test_score_pending.py` 21/21 (offline; scored/partial/skip/future/idempotency/aggregators/
taf_id-fallback-via-runs). Known limitation: human TAFs are parsed from raw (parse_body preferred
when present, but no remark-stripper exists yet -- same as the validated KLSV grading path).

### M4 PLAN -- paired live collection (decisions locked with owner 2026-07-13)
Goal: schedule the TAF AGENT to run in parallel with each HUMAN TAF so both are archived FROZEN together
(leakage-proof: at issue time truth doesn't exist yet). Locked decisions:
- **(a) run the agent LIVE** at issue time now; snapshot-and-replay of inputs -> (b) later for Batch API/
  multi-model/reproducibility.
- **MANUAL fires first** (build the collector as a hand-run command); revisit the trigger (poll AWC for a
  new official TAF vs fixed clock) after a couple days of watching.
- **Stations** (all issue live 30h MILITARY TAFs; BUFKIT only exists for civil-co-located ICAOs):
  `KWRI` (McGuire NJ), `KMIB` (Minot ND), `KSSC` (Shaw SC) are self-covered (GFS+NAM+HRRR); `KBAB` (Beale CA)
  has NO BUFKIT -> proxy the model-data tools to **`KSMF`** (Sacramento, ~30nm, same valley) while still
  archiving KBAB's own human TAF + obs. (Coverage probes in this session; KLSV/KLAS pattern.)
- **Storage:** DuckDB single-writer enforced with a `flock` lockfile (archive-write at issue time vs
  `--pending` scoring-write later never overlap).
- **Pi:** unattended collection on an always-on Raspberry Pi (64-bit Pi OS Lite, `uv sync`, `.env`); two
  cron jobs under `flock`; pull data off via nightly `rsync` (one `.duckdb` file + archived raw + logs).
  Owner asked for a full Pi runbook when we automate.
- **Build order:** (1) human-TAF archive path (`awc.fetch_taf` -> `tafs`, canonical=true, producer_kind=human)
  DONE; (2) `scripts/collect.py` -- archive human + run agent + write a PENDING evaluation, NO scoring --
  DONE; (3) `score_taf.py --pending` -- score elapsed windows, flip to scored, BUILD the deferred result
  tables -- DONE 2026-07-16 (see STORE STATUS above); (4) Pi/cron + retrieval runbook -- deployed 2026-07-16
  (docs/pi_setup_log.md; crons PENDING install until the climo build finishes; a third cron entry runs
  `score_taf.py --pending --backfill iem` every 6h).

### COLLECTION HARDENING (review fixes, 2026-07-16 -- this session)
- **Truth-obs banking via dual-write:** `collect.py` now ingests obs ONCE into the BENCHMARK DB (no
  cutoff -- truth wants everything; the model never reads that DB) and copies the pre-cutoff back-window
  into the per-run DB via new `store.copy_obs` (cutoff enforced in SQL). Successive 8-hourly cycles tile
  the timeline, so verification obs accumulate as a side effect of collection; IEM is only the backfill.
- **get_previous_taf leakage guard hardened:** `store.previous_human_taf` gained `valid_before`; the
  collector passes the run's valid_from, so the current cycle's bulletin can NEVER qualify however early
  it posts (KBLV was observed posting 30 min early; the 15-min issue buffer alone would break silently).
  Roster routine TAFs verified live to stamp exactly on the valid hour; mid-cycle AMDs still pass the
  filter by design (a human forecaster has them in hand too; the notaf ablation measures the anchoring).
- **Stub run row:** collect.py inserts a `stop_reason='incomplete'` runs row BEFORE the agent runs (under
  the lock); the final persist_run replaces it by run_id. A cell killed by schedule.py's 30-min timeout
  now leaves a record instead of vanishing (a silent missing matrix cell).
- **Temp-dir leak fixed:** the per-run DB dir is rmtree'd in a finally (Debian 13 /tmp is tmpfs -- leaked
  dirs accumulated in RAM on the Pi). `--ingest-hours` default 12->24 (matches get_trend's look-back).
- **`store.copy_climo` no longer swallows errors:** only CatalogException (source has no climo tables)
  reads as not-built -> 0; a missing/locked/corrupt source now raises.
- Verified end-to-end live (KWRI, bogus model id -> fatal captured, scratch DB): 24h obs banked, run-DB
  copy cutoff correct, stub->final row replacement, human 1800Z routine TAF archived, temp dir cleaned.
  All self-tests green (incl. new test_score_pending 21/21); ruff clean. NOT committed (owner reviews).

### UNCOMMITTED: everything is still in the working tree
The entire scoring subsystem (this session) PLUS the earlier climo + imagery + worksheet work are all
uncommitted. Commit before/at the start of M4. `docs/taf_score.md` is the authoritative scoring blueprint
(gitignored -- the tracked CODE + persisted policy hashes are the versioned record).

### EARLIER CONTEXT (2026-07-09, predates + is superseded by the scoring subsystem above)
An earlier session landed ALL 11 fixes from the 2026-07-08 review (that findings doc has since been
removed now that every item shipped): Tier 1
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

DO NEXT: the two reinforcing pieces were the climatology tool + the forecasting worksheet.
1. CLIMATOLOGY TOOL (next-steps step 8) -- DONE. `get_climo` built + verified for KLSV July; ALL
   climo exit criteria PASS as of 2026-07-10 (--check, idempotency after a record-date
   tie-break fix, obs-untouched, guards; see the Status bullet + `logs/climo_KLSV_20260709-151921.md`).
   Remaining housekeeping: commit the uncommitted climo + imagery work, and build any additional months
   the target station needs.
2. FORECASTING WORKSHEET (next-steps step 12) -- v1 DONE (2026-07-10; see the Status bullet + design
   doc + `logs/worksheet_agent_KLSV_20260710-103022.md`). The typed `TafWorksheet` + sink + evidence
   threading + mode gate + persistence are built, self-tested 19/19, and verified live (MiniMax advisory:
   schema-error -> semantic findings -> clean worksheet -> clean 30h TAF). Follow-ups if wanted: a live
   `required`-mode run (gate coded but unexercised), a recent-valid-time run (real obs + built climo) to
   exercise sanity_checks against observed data, and Milestone 2 (`get_model_run_verification`, which
   promotes model_run_verification from advisory to gated).
(Optional) Persist the vs-obs VERIFICATION below as a scoring note -- the first real TAFVER signal, seeds
next-steps step 9.

DEFERRED (was step 1, deprioritized 2026-07-09): the break-on-`length` convergence fix (nudge-and-continue
on length+no-toolcall + max_tokens ~12-16k). Skipped for now because the worksheet may make it unnecessary;
revisit only if convergence is still failing AFTER the worksheet lands. Details in Known problems below.

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
8. **Climatology tool (DONE).** `get_climo(station, month?)` reads the `climo_*` product tables built by
   `climo.py` (`scripts/build_climo.py`). The raw multi-year build history
   is thrown away (leakage guard) — only the product rows persist. Built + verified for KLSV July (model
   selected get_climo, values inside tolerance). See the Status bullet. Feeds the worksheet (step 12).
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
12. **Forecasting WORKSHEET -- intermediate tasks before emit (v1 DONE 2026-07-10).** `worksheet.py`
    (typed `TafWorksheet` + guardrails + semantic `validate()` + `worksheet_guide()`), the
    `submit_taf_worksheet` sink + `get_current_taf`/`check_taf` wrappers, config modes
    (`worksheet_mode`/`evidence_mode`), `store` persistence tables, and the `test_worksheet_agent.py`
    driver (evidence-id threading + mode gate + persistence) all landed. Self-test 19/19; verified live
    with MiniMax (advisory): schema-error -> semantic findings -> clean worksheet -> clean 30h TAF. See the
    Status bullet. Milestone 2 (`get_model_run_verification`) + continuity tools (`get_previous_tafs`) are
    the deferred later milestones. ORIGINAL RATIONALE (retained): instead of one
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

## Open questions to confirm with the lead meteorologists (SME)
TAF scoring design lives in `docs/taf_score.md` (sec 15 has the full SME list). Standing
follow-ups:
- HAND-SCORED GOLDEN FIXTURE (gates "official" TAFVER): get a lead meteorologist to score
  ONE TAF by hand (initial + FM + BECMG + TEMPO groups) against a real METAR/SPECI sequence,
  so `tafver.py` can be matched BYTE-FOR-BYTE against a human expert's per-element + combined
  numbers. This proves our DOCTRINE INTERPRETATION is correct; the synthetic self-tests only
  prove internal consistency. Until it passes, TAFVER output is labeled provisional/benchmark,
  NOT official. Deliverable is the SME's; we supply the TAF + ob sequence.
- Installation cig/vis category tables + official published landing minima for the target
  station(s). v1 uses a FIXED default (200 ft ceiling, 1/2 SM vis) for all stations.
- Confirm the A7.2 "Table A2.1" = "Table A7.1" typo reading via observed BIFROST behavior.

## Important caveats
- Model names and provider prices shift week to week — verify model strings against
  provider docs at build time rather than trusting any hardcoded list.
- If real (non-open-source) AF weather data is ever used, hosting may need an
  authorized DoD environment, not commercial cloud. Flag this if it comes up.

## Known problems to address
- **Live network tools drift with wall clock (FIX ACTION for the next large experiment; flagged
  2026-07-16 at go-live).** The DB-side leakage guards are airtight (obs + prior-TAF cutoffs are
  pinned to the ISSUE time and enforced in SQL, so scheduler fire time / heavy-hour spillover
  cannot leak truth). But the live network tools (get_imagery, get_map, get_sounding,
  get_fcst_sounding, get_point_forecast) fetch whatever is posted at WALL-CLOCK time: a cell that
  executes minutes-to-an-hour after the pinned issue time sees slightly fresher imagery, and
  possibly a newer model cycle, than the human forecaster had at issue. Input-fidelity drift, not
  truth leakage (the verifying METARs are never exposed) -- acceptable for this ~1-week run, but
  before the NEXT big collection, pin those inputs to issue time: snapshot-and-archive the network
  products at (or just before) the issue hour and serve the agent the frozen copies (the M4
  "snapshot-and-replay (b)" path; also what makes historical valid times airtight and enables
  Batch API / multi-model replay off identical inputs).
- **break-on-`length` kills a truncated emit turn (DEFERRED 2026-07-09 -- pending the worksheet).** In
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
