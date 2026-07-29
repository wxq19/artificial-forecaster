# CLAUDE.md — Artificial Forecaster

Guidance for Claude Code working in this repo. Read this fully before acting.

## How to communicate

- Write every user-facing reply in ASD-STE100 Simplified Technical English.

- Use one idea in each sentence.

- Use a maximum of 20 words in an instruction sentence.

- Use a maximum of 25 words in a descriptive sentence.

- Use a maximum of six sentences in a procedural paragraph.

- Use the active voice.

- Use the simple present tense when possible.

- Keep the articles "the" and "a".

- Use one word for one meaning.

- Do not replace a word with a synonym for variety.

- Do not use idioms, slang, or figures of speech.

- Keep technical names unchanged. This includes files, commands, functions, classes, variables, and error text.

- Use plain language.

- Explain an unavoidable technical term with a short definition.

- Lead with the action or the outcome.

- Start a completed task with:
  "Done: <outcome>"

- Do not add a conversational preamble.

- Do not start with phrases such as:
  "Let me..."
  "Great question..."
  "I would be happy to..."
  "Based on your request..."

- Put the context and reasoning after the action.

- Use numbered steps for a sequence.

- Put one bounded action in each step.

- Use a maximum of five items in one list.

- Split a longer list into:
  "Do now"
  and
  "Do later"

- Restate the task state during every turn of a multi-step task.

- Use this format:
  "Step 3 of 5 done: schema updated. Next: backfill."

- Do not assume that the user remembers the previous message.

- Give a concrete time estimate when the task requires user work.

- Do not use vague estimates such as:
  "This will take some work."

- Keep normal answers to six sentences or fewer unless the user asks for depth.

- Answer only the requested topic.

- Do not include unrequested alternatives, comparisons, or tangents.

- End with one concrete next action when work remains.

- Do not end with:
  "Let me know."
  "Tell me what you think."
  "I can help with that."

- State assumptions before you act.

- Ask a question only when a requirement is genuinely ambiguous.

- Otherwise, select the sensible default and state the selected default.

- If a second issue appears, finish the first issue.

- Offer the second issue as a separate task.

- Do not combine the second issue with the current task.

- After a change, summarize:
  - What changed
  - Where it changed
  - Why it changed

- Include exact file paths when files change.

- After a feature change, add a short manual test checklist.

- The checklist must state what to open, click, enter, and confirm.

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

### VERSION CONTROL IS MINE — HARD RULE, NO EXCEPTIONS
**NEVER run a git or GitHub command that changes state. I am the ONLY one who commits.**
- FORBIDDEN, always, even when I asked for the underlying work and even when tests pass:
  `git commit`, `git push`, `git merge`, `git rebase`, `git reset`, `git tag`,
  `git checkout`/`switch`/`restore` that discards work, `git stash`, `git branch -d`,
  and every `gh` write (PR create/merge, release, issue edit).
- ALLOWED: read-only inspection — `git status`, `git log`, `git diff`, `git show`,
  `git fetch`, `git rev-parse`, `gh` read commands.
- Finishing a task means leaving the work **in the working tree, uncommitted**, and
  telling me what changed. Do not stage it, do not commit it "for review", do not offer
  to commit as a follow-up. A green test suite is NOT authorization to commit.
- Deploying to the Pi is a git action too: **do not `git pull` on the Pi.** Tell me what
  needs deploying and I will pull.
- If you think something should be committed, say so in one sentence and stop.

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
- **Cloud (real inference):** used because the laptop has no GPU. Round 1 ran on
  Together AI (`https://api.together.ai/v1`) with Gemma-4-31B, MiniMax-M3 and Kimi.
  Round 2 moves to OpenRouter, which reaches models Together does not serve. A router
  picks its own backend and quantization, so an unpinned round silently mixes providers
  and the scores stop comparing -- `config.llm_provider_order` PINS the backend and
  `llm_provider_allow_fallbacks=false` makes a reroute a hard failure, not quiet
  contamination. Non-routing endpoints (Together, Ollama, vLLM) see no extra body.
- **HPC (final target):** MIT SuperCloud. Slurm scheduler, Volta V100 GPUs
  (`--gres=gpu:volta:1`), Podman containers (GPU via `--device nvidia.com/gpu=all`),
  vLLM serving the weights. Compute nodes have NO internet — weights/images pre-staged
  via the download partition (`-p download`), loaded from local paths.

## Model choices
- Local dev: `qwen3-vl:2b` / `:4b` (small, just for plumbing). This is the config
  default, so a run with no `.env` talks to Ollama, never to a billed endpoint.
- Cloud: stay serverless (per-token), not dedicated. Use a Batch API for a big eval run.
  Round-2 candidates are the screened set below; prices are in `docs/model_costs.md`.
- SuperCloud: 8B-32B class VLM (fits V100s).
- KEEP THE SAME MODEL TIER across environments so benchmark numbers are comparable.
- PIN one provider per model and record it in run provenance. The same open weights on
  a different backend or quantization do NOT produce comparable scores.

### ROUND-2 MODEL CANDIDATE SET (screened 2026-07-24 on OpenRouter; `scripts/screen_models.py`)
Screened 11 vision+tools models on 3 HARD frozen round-1 fixtures (KFTK/RJTY/KWRI) for
$/AFMAN-clean TAF + chart-reading + reasoning ablation. Full report:
`logs/model_screen_20260724-141933.md`; details in memory `model-screen-resume`. TAFVER here is
on hard cases so runs below round-1's 83.5 pooled. Candidates (best clean arm; model -> provider):
- **Inkling (reason-high) 85.9** @ Together -- best forecaster; 1 of 3 miss was a provider
  JSONDecodeError, scored 87.6/84.3 where it ran. ~$0.093/clean.
- **Gemma-4-31B (base) 82.8** @ Together -- best value+reliability, vision 9/9, $0.016.
- **Grok-4.5 (reason-low) 80.6** @ xAI -- most reliable (3/3), $0.074, vision 9/9.
- **Gemini-3.1-flash-lite (reason-low) 76.8** @ Google -- cheap+reliable (3/3), $0.014.
- **Mistral-small-3.2 (base) 74.6** @ Mistral -- CHEAPEST $0.007, vision 7/9.
- MiniMax-M3 (reason-on) 74.5 @ Together (round-1 anchor), $0.026; Kimi-K3 75.4 @ Moonshot
  but priciest ($0.151, int4); Qwen3-VL-235B 66.1 @ Alibaba/DeepInfra (lowest emitter).
- **OUT: MiMo-v2.5 and Qwen3-VL-32B cannot emit a valid TAF on any provider** (MiMo ruminates
  10-40k tok then the provider empty-responds; re-run at 16 steps/40k tok did not help).
KEY LESSONS: (1) MORE reasoning mostly HURT -- default LOW/OFF (Gemini-high broke emit entirely;
only Inkling rewarded high). (2) Vision is near-universal; EMIT reliability (valid nested TAF JSON)
is the discriminator, same as round-1 orchestration. (3) OpenRouter routes to third-party backends
of wildly varying tool-call/image fidelity -- BAD: Novita (mangles nested args), SiliconFlow (leaks
special tokens), GMICloud (empty-responds); a hard provider pin OVERRIDES account presets, so put
the exclusion in the request (`ignore` list). xAI rejects GIF (transcode charts to PNG). Validate on
the REAL nested-emit task, not a small probe. Before a round: set an OpenRouter hard per-key credit
cap, and wire the validated (model->provider) pins + IGNORE_PROVIDERS into `schedule.py`.
NOTE: all screen code is UNCOMMITTED (OpenRouter seam in src/forecaster/{config,llm,agent,store,
runlog}.py + ping_models.py + .env.example; new scripts/screen_models.py).

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
- **Tools run live in the agent loop:** model emits a tool call → our code fetches or
  renders → returns a chart (PNG) → fed back to the VLM as an image. NOTE what actually
  shipped: we FETCH pre-rendered forecaster products (wxmaps, imagery, soundings) and
  render from TEXT or from the GRIBStream archive (charts.skewt, charts.meteogram). We
  do NOT decode GRIB ourselves, so cfgrib/cartopy stayed deferred.
- **The model is stateless.** It only knows what's in the `messages` array on each call.
  WE own context — building, trimming, and managing the `messages` list is our code's job.
- **GRIB/imagery do NOT go in a relational DB.** Keep arrays/images as files
  (GRIB/NetCDF/Zarr, PNGs); a DB stores structured records + file references only.
- **The relational DB is for:** run/experiment tracking, parsed METAR/TAF observations,
  and verification scoring. DuckDB was chosen over Postgres: single-tenant, embedded
  (works on the air-gapped SuperCloud node), and a `.duckdb` file you can ship and hash.

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
│   ├── charts.py         # the ONLY file that imports matplotlib/metpy (meteogram/wx_timeline/skewt/filmstrip/loop_mp4)
│   ├── soundings.py      # OBSERVED upper-air client (seam): SPC/Wyoming skew-T images, plus a
│   │                     #   `bufr` source that returns a typed ObsProfile for charts.skewt
│   ├── wxmaps.py         # live synoptic map client (WPC/OPC/SPC-meso/TT GFS); fetch charts (seam)
│   ├── upper_air_sites.py # nearest-3 radiosonde table per station (regen via scripts/build_upper_air_sites.py)
│   ├── climo.py          # station-climatology builder (scratch-DB ingest; NO SQL) -> store.rebuild_climo
│   ├── imagery.py        # live satellite + radar image client (seam): GOES via NESDIS/STAR, Himawari
│   │                     #   via SLIDER/OSPO, Meteosat via EUMETSAT WMS; radar via IEM radmap.php.
│   │                     #   Imports PIL (NOT matplotlib) for ONE job: compositing SLIDER's
│   │                     #   map layers onto its raw tiles -- see _slider_overlay
│   ├── radarsites.py     # WSR-88D site table (nearest-radar lookup; regen via scripts/build_radarsites.py)
│   ├── terrain.py        # static terrain/coastline client (shaded-relief map + terrain rose) (seam)
│   ├── geo.py            # shared spatial primitives (great-circle distance/bearing/nearest-N); stdlib only
│   ├── neighbors.py      # static nearest-neighbor roster (regen via scripts/build_neighbors.py)
│   ├── gribstream.py     # GRIBStream point-forecast client -- model time-series fetch (seam)
│   ├── modeldata.py      # GRIBStream ORCHESTRATOR (prefetch -> store archive; no SQL/network of its own)
│   ├── worksheet.py      # typed TafWorksheet pre-emit reasoning artifact + validate() + guide (seam)
│   ├── tafstate.py       # SHARED scoring primitives (absolutizer, truth views, baselines); PURE
│   ├── tafver.py         # scorer 1 -- TAFVER (ACCI15-120 Att 7 percent-correct); PURE
│   ├── tafamend.py       # scorer 2 -- amendment-implied busts (DAFI 15-129 rules); PURE
│   ├── tafskill.py       # scorer 3 -- skill (element error / event contingency / MACE); PURE
│   ├── tafarchive.py     # raw TAF bulletin -> immutable `tafs` archive row (pure; no DB)
│   ├── stations.py       # roster of military aerodromes + their issue cycles (+ archive-only list)
│   ├── runlog.py         # freeze an agent RunResult as durable provenance (runs row + transcript)
│   ├── agent.py          # the agent LOOP -- the ONLY file driving the model's tool-calling turns
│   └── tools.py          # agent read tools + OUTPUT sinks (emit_taf/submit_taf_worksheet/check_taf)
├── scripts/              # dev + end-to-end test drivers (markdown logs -> logs/)
├── docs/                 # GITIGNORED. References (FMH-1 wx table, AFMAN 15-124, AFH 15-101)
│                         #   + the design docs and the offloaded record -- see the pointer table below
├── data/                 # GITIGNORED throwaway/cache: forecaster.duckdb, charts/, soundings/, imagery/,
│                         #   terrain/, gribstream/, metars/ -- PLUS benchmark/ + runs/ (the harvest)
└── logs/                 # run transcripts (markdown)
```

## Secrets hygiene (hard rule)
- API keys live in `.env` ONLY. Never in code, never in `.env.example`, never committed.
- `.gitignore` excludes `.env`, `.venv/`, `models/`, `*.tar`, `data/`.
- If you ever notice a key in tracked content, stop and flag it immediately.


## Status (current state, 2026-07-28)

Every seam named in the Project structure tree above is BUILT, self-tested and
ruff-clean. The tree is the authoritative map of which file owns which seam.
For WHY a seam is shaped the way it is, and the bug each step surfaced, read
`docs/build_log.md` -- do not re-derive it here.

- **Input seams:** metar.py, tafparse.py (both parse to typed objects; the library
  object never escapes; raw text always retained).
- **Output seams:** tafgen.py (typed product -> AFMAN TAF text + validate + roundtrip),
  worksheet.py (pre-emit reasoning artifact).
- **Data seams:** store.py (the ONLY SQL/DuckDB), awc.py + iem.py (obs), climo.py
  (station climatology), gribstream.py + modeldata.py (model point data),
  soundings.py / wxmaps.py / imagery.py / terrain.py (fetch pre-rendered products),
  charts.py (the ONLY matplotlib).
- **Agent:** agent.py owns the tool-calling loop; tools.py exposes 19 read tools in
  `TOOLS` plus the sinks emit_taf / submit_taf_worksheet / check_taf, and
  get_previous_taf for leakage-safe continuity.
- **Scoring:** tafstate.py (shared primitives) + tafver.py / tafamend.py / tafskill.py
  (three orthogonal scorers, all PURE), with four baselines: persistence,
  climatology, human, human_composite.
- **Collection:** scripts/collect.py (one matrix cell), scripts/schedule.py (hourly
  dispatch), scripts/poll_tafs.py (human TAF archive), scripts/score_taf.py --pending,
  scripts/results_report.py (the ONLY correct way to derive a headline number).

### Repository state -- READ THIS FIRST
- **Everything since commit 8b7f107 is UNCOMMITTED** (20 modified + 4 untracked as of
  2026-07-28), covering the model screen, the BUFKIT removal, the nowcast work, the whole
  archive build and the imagery honesty pass. `pyproject.toml` and `uv.lock` joined the
  modified set on 07-28d (the pillow declaration); no NEW untracked file has appeared since
  07-28b, so every other edit landed in a file already on the list.
- **AND FOUR COMMITS ARE UNPUSHED.** `origin/main` is at **50823b1**; local `main` is at
  **8b7f107**. The Pi pulls from the REMOTE, so it can see none of this -- which is why it
  still polls 63 stations instead of 71. Verified read-only 2026-07-28.
- **`.env` has `llm_model` EMPTY** and points at OpenRouter, so any code path that
  falls back to `settings.llm_model` returns a 400 ("No models provided"). Pass an
  explicit `--model`.
- **`MODEL_DATA_ENABLED` is still false**, so the GRIBStream tier bills nothing until
  it is flipped. The Pi has its OWN `.env`. See PICK UP HERE item 0 for the detail --
  do not restate it here, because a second copy drifts.
- The Pi runs the poller and two scorers. No LLM billing since 2026-07-20.

## Where the rest of the record lives (read on demand, NOT by default)

CLAUDE.md keeps the standing rules, the current state, the open decisions and ONE
live session block. Everything else moved to `docs/` on 2026-07-28 to cut context
cost. `docs/` is GITIGNORED, so these files have no git history -- copy before you
delete, never after.

| File | What is in it | Read it when |
|---|---|---|
| `docs/build_log.md` | The full build narrative for every shipped seam, 2026-06 to 2026-07-28: what each module does, why it is shaped that way, and the bug each step surfaced. | You need the REASON behind a design, or the history of a bug class. |
| `docs/history_2026-07.md` | Closed session blocks (07-19 through 07-23), superseded context, finished next-steps, resolved problems. | You need the numbers or reasoning behind a decision CLAUDE.md now states in one line. |
| `docs/model_costs.md` | The 2026-07-17 model x provider x price table and the per-provider routing notes. | You are choosing a provider. Prices move weekly -- re-verify. |
| `docs/archive/CLAUDE_pre_trim_20260728.md` | The 2,057-line CLAUDE.md exactly as it stood before the 2026-07-28 trim, byte for byte. | Something reads as missing after the trim, and you want to confirm the original wording. |
| `docs/archive/` (rest) | Retired design docs, superseded by shipped code. | Rarely. Prefer the code. |
| `docs/archive.md` | The round-2 archive design. Its "Build plan" section is the AUTHORITATIVE tracker for archive work. | Any archive or replay task. |
| `docs/artifact_store.md` | The STORAGE contract for frozen bytes (archive prep item 6): content-addressed blobs + the three index tables, the identity-not-station keying, capture-time `run_manifest`, and the measured sizing. NOT the tool contract. | Building `archive_run.py`, or any question about how a frozen artifact is addressed. |
| `docs/taf_score.md` | The scoring blueprint (M0-M3) and the full SME question list. | Any scoring change. |
| `docs/gribstream_model_data.md` | The model-data subsystem plan and remaining work. | Any GRIBStream or model-data task. |
| `docs/taf_worksheet_design.md` | The worksheet design. Milestone 2 in it is SUPERSEDED -- see the 07-22 block in the history file. | Worksheet keep/drop/redesign. |
| `docs/pi_setup_log.md` | The Pi runbook and the exact cron lines. | Deploying to the Pi. |
| `docs/field_ideas.md` | Literature triage and costed lesson list. | Planning a new experiment. |
| `docs/nowcast.md` | The nowcast workstream design. Read section 10 FIRST. | Nowcast work (deferred until after round 2). |

Two things live in CODE, not in prose. Do not restate them here, because a copy drifts:
- The station rosters are `src/forecaster/stations.py` (`STATIONS`, `ARCHIVE_STATIONS`,
  `poll_icaos()`).
- The tool contracts are `src/forecaster/tools.py` (`TOOLS` plus the sinks).

## NEXT SESSION -- pick up here (paused 2026-07-28)

### SESSION 2026-07-28e -- ARCHIVER BUILT (Phase 3.1), item 7 DECIDED
UNCOMMITTED, now **21 modified + 7 untracked**. NEW: `src/forecaster/artifacts.py`,
`scripts/archive_run.py`, `scripts/test_archive_run.py`. MODIFIED: `src/forecaster/store.py`
(the three archive tables), `docs/archive.md`, CLAUDE.md. ruff clean. Self-tests:
test_archive_run 43/43 (new), test_modeldata 124/124, test_runlog 25/25, test_worksheet 19/19,
test_score_pending 29/29, test_tafstate 40/40, test_results_report 49/49, test_tool_stamps 5/5,
test_tool_fallbacks 5/5.

**ITEM 7 IS DECIDED (owner, 2026-07-28): round 2 serves tools ONLY from the archive, never
live, with no live fallback.** That is stronger than the design `docs/archive.md` carried, and
two consequences are now load-bearing. First, a capture must cover the WHOLE product space a
station can request, not the calls one agent happened to make -- which is why `archive_run.py`
ENUMERATES per station rather than recording a run. Second, a request outside the archived set
must snap to the nearest artifact and SAY SO, or refuse; `store.nearest_artifact_key` returns
`snap_minutes` for that receipt. The archive.md row is updated in place.

**Phase 3.1 is BUILT to the `docs/artifact_store.md` contract** -- content-addressed blobs in
`artifacts.py`, three index tables in `store.py` (store.py stays the only file touching
DuckDB; images never enter the relational DB), keyed on RESOLVED product identity and NEVER on
station, with `run_manifest` written at capture time. It lives in its own
`data/archive/index.duckdb` so a round can be harvested and pruned as a file move.
- **MEASURED, live:** KWRI cold = 68 artifacts / 15.7 MB. A second station in the same sector
  and domain = **8 new, 60 reused, 1.6 MB** -- the per-product keying doing its job. A re-run
  at the same pinned cycle = 66 reused, 0 bytes.
- **What is frozen: provider BYTES plus provenance, not tool receipts.** A receipt is a pure
  function of the bytes and the index row, so re-generating it at serve time lets a wording fix
  reach an already-collected round. Same argument as archiving loop FRAMES, and it is what lets
  any (frames, step_min) be subsampled from one capture.
- **Every resolver is the function the tool calls**, so the archive cannot drift from what the
  agent would be served. Where `tools.py` applies a presentation rule on top (OSPO has no
  geocolor, Meteosat publishes no water vapour, the radar in-network test), it is mirrored AND
  pinned by a self-test check -- under serve-from-archive a drift there is a hole nobody
  notices until a replay is short an image.
- **TWO DEFECTS FOUND BY RUNNING IT, both mine, both fixed before landing.** (a) The radar
  group was gated on the regional bbox, so **KMIB and KRCA would have archived NO radar** --
  they sit in a gap between the curated regions while having a WSR-88D 35 km and 23 km out, and
  the tool serves them a station view. Now mirrors `_radar_for_station`'s in-network test
  exactly. (b) `--dry-run` hit the network, because the sounding planner resolved a launch time
  eagerly; it is deferred behind `expand` like the loop frames.
- **RJTY's nearest radiosonde has NO record in the Wyoming BUFR provider** (47646 TATENO,
  78 km; `last_known_time` is None, and the 07-28d fix is what made that distinguishable from a
  quiet site). The planner now walks the nearest THREE sites instead of only the first.
- NOT captured here, deliberately: GRIBStream rows (asOf rebuilds them), obs and TAFs already
  in the scoring DB. The live TAF bulletin IS captured -- AWC serves only the current one.

**THEN THREE CAPTURE-SIDE FIXES, all owner-prompted, all landed the same session.** These
matter because capture is irreversible: 3.2 can be rebuilt against frozen bytes, but an hour
captured without them is gone.
1. **JAPAN HAS SOUNDINGS AFTER ALL -- the provider has TWO FEEDS and we only read one.**
   `src=BUFR` is the ~1 s ascent; **`src=FM35` is the traditional TEMP bulletin**, and both
   answer TEXT:CSV with the SAME columns, so one parser reads either. 47646 TATENO (RJTY's
   nearest, 78 km) returns **HTTP 400 under BUFR and 462 launches for 2026 under FM35**, 3 of
   them off-cycle. So "RJTY has no upper-air data" was OUR bug, not a coverage gap -- the
   earlier note in this file saying otherwise was wrong. New `soundings.resolve_source()`
   tries BUFR then FM35 and returns the feed WITH the time; `fetch_profile(..., src=)` and the
   cache key carry it; `last_known_time` searches all feeds, else a site living only in FM35
   is declared nonexistent. `get_sounding` now renders Tokyo (verified by eye: 1001 hPa
   28.0/23.3 C, correct hodograph) and its receipt names the FEED, since 27-of-53 levels means
   something different from 71-of-6,527. The archiver keys `47646/fm35` vs `72501/bufr`.
2. **LOOP CADENCE REBUILT: capture the finest STEP, let SPAN come from accumulation.** Frames
   key on their REAL time, so consecutive hourly captures concatenate into one continuous
   series per (region, product) -- which means a capture only has to get the STEP right and
   bridge to the next hour. Span = (frames-1) x step, so **7 frames x 10 min = 60 min exactly
   closes the gap**; 6 would leave a 10-minute hole every hour. Default was 10 x 30. This is
   **CHEAPER** (7 frames not 10, and loop frames are ~60% of the archive: KWRI 68 -> 59
   artifacts, 15.7 -> 13.4 MB) and strictly MORE capable -- a 30-min capture can never answer a
   10-min request, a 10-min capture subsamples to any coarser one. Verified all three providers
   return DISTINCT frames at a 10-minute step.
3. **`--all-regions`, default OFF.** `get_imagery` takes an explicit `region`, and only **16 of
   23** regions are reachable from the 71 stations. The 7 unreachable (both full disks,
   conus_west, puerto_rico, caribbean, middle_east, africa) would have no bytes at all under
   an archive-only round. ~16 extra stills/hour, ~9% on the round. These get NO manifest row on
   purpose: `run_manifest` answers "what was this STATION entitled to", and a named far region
   is outside that, so the serve side resolves it through `artifact_keys` directly.

**DEFERRED (owner, 2026-07-28): terrain pre-warm, to be done with the climo build.**
`terrain.sample` still does a live elevation fetch, so it is the one live call left under an
archive-only round. Static, so there is no fidelity risk and no capture deadline.

**Next: Phase 3.2**, the serve side, and it is the only piece that touches `tools.py`.

### SESSION 2026-07-28d -- second code review: 8 findings fixed, pillow declared
UNCOMMITTED, now **20 modified + 4 untracked** (`pyproject.toml` + `uv.lock` are the two new
entries; `uv.lock` gained pillow as a DIRECT dependency). Touched
`src/forecaster/{tools,imagery,soundings}.py`, `scripts/{build_upper_air_sites,
audit_neighbors}.py`, `pyproject.toml`, CLAUDE.md. ruff clean. Self-tests: test_tool_fallbacks
5/5, test_tool_stamps 5/5, test_modeldata 124/124, test_geo 29/29, test_worksheet 19/19,
test_runlog 25/25, plus `build_upper_air_sites.py --check` CHECK PASS (all 71 still rebuild
byte-identically) and live checks of every changed path.

A second `/code-review` pass over the working-tree diff found 8 defects; all are fixed. The two
that mattered:
1. **The 07-28c site-only hazard guard never fired for the case it was written for.** It tested
   `if ref is None`, true only when the point has NO rows at all -- but a neighbour or grid
   point HAS surface rows under config B, so `ref` was not None and
   `get_hazard_scan(station="KWRI", location="KNEL")` still rendered a confident
   "diagnosed from GFS + HRRR" header over an empty ICING block and `CAPE=--, CIN=--`. The
   fix picks the nearest LEVEL-BEARING entry and tests THAT, which also fixes a second case:
   a valid time landing in the surface 6h back-tail now snaps to the nearest level-bearing
   time (and the header names it) instead of rendering blank. **A guard written against the
   wrong condition reads exactly like a guard that works.**
2. **An unknown `bufr` sounding id was reported as a real site that is not reporting.**
   `soundings.inventory()` scrapes the page and returns `[]` for an unparseable id AND for a
   quiet site, so `latest_time` alone cannot tell them apart -- and `tools.py:120` teaches the
   model that `OUN`/`MPX` are valid ids for the `spc` source, so carrying one over to `bufr`
   is a likely model action. Verified live: `OUN` and `99999` both claimed "the site is simply
   not reporting". New `soundings.last_known_time()` reads the whole record (this year, then
   last, so early January is not a false dead site) and the two cases now get different
   replies; the quiet-site reply names the most recent ascent on record.
The other 6, briefly: **pillow is now DECLARED** in `pyproject.toml` (07-28c flagged this and
left it -- `imagery.py` imports PIL at MODULE scope and `tools.py` imports `imagery`, so a
missing Pillow breaks the whole agent import, not just the SLIDER overlay); a failed
`awc.station_latlon` in `_get_map` told the model "no catalogued chart source covers this
station at all", fabricating a geographic fact about a CONUS station whose lookup merely 5xx'd
(a `located` flag now separates lookup failure from real no-coverage, and withholding is
unchanged in both); `_eumetsat_loop_frames` labelled frames with the RAW time while the URL
snapped down to the 10-minute grid, so `step_min=15` captioned the 11:40Z scan "11:45Z" (snap
now happens once, before both label and URL -- the same label-over-wrong-content class as the
two 07-28c imagery bugs, and the third instance of it in two sessions);
`build_upper_air_sites.build()` printed SKIPPED and CONTINUED on a network blip, silently
omitting the station from the table and ending in defect 2's false "not reporting" (it now
refuses, like `build_neighbors.build()`); `audit_neighbors`' write guard compared list LENGTH
to the roster, so 71 wrong ICAOs passed and would write a table measured over the wrong set
(membership now); and `_SYNOPTIC_WV` widened `puerto_rico`/`caribbean` into `conus_east`,
whose southern edge does not reliably reach 18 N -- they now widen to `full_disk_east`, which
is the right synoptic scope for a tropical site anyway. No roster station resolves there, so
that one is latent.

**Still open, carried from 07-28c and NOT fixed:** `llm.routing()` checks only
`llm_provider_order`, not the base URL, so with `LLM_PROVIDER_ORDER` set in `.env` a Together
run would send an OpenRouter `provider` body. LLM-only -- it cannot affect the archiver.

### SESSION 2026-07-28c -- code review: 10 findings fixed, SLIDER imagery given a map
Still UNCOMMITTED and still **18 modified + 4 untracked** -- every edit landed in a file the
earlier blocks already list, so the counts and the push story below are unchanged. Touched
`src/forecaster/{imagery,tools,awc,modeldata,soundings,neighbor_rates}.py`,
`scripts/{build_neighbors,audit_neighbors}.py`, CLAUDE.md, docs/artifact_store.md. ruff clean.
Self-tests: test_tool_fallbacks 5/5, test_tool_stamps 5/5, test_modeldata 124/124, test_geo
29/29, test_worksheet 19/19, test_runlog 25/25, plus 8 direct checks for the paths no self-test
covers. Verification page (private): https://claude.ai/code/artifact/ffbb8f1b-e89f-4f19-8e9d-928821cab8ae

**A /code-review pass over the working-tree diff found 10 defects; all are fixed.** The two that
mattered were both in the imagery seam and both reproduced live before fixing:
1. **Every Meteosat LOOP frame was the same image.** `_eumetsat_loop_frames` appended its own
   `&time=` to a URL that already carried one (the fix that added `at=` to
   `meteosat_point_url` missed this caller), and the WMS honours the FIRST value. Three
   requested frames returned one sha256 under three labels. **Same class as the round-1
   wrong-ocean bug: a confident label over content that is not what it says.** Verified fixed
   at ETAR -- 6 distinct frames, and Gemma-4-31B read the west-to-east motion correctly.
2. **water_vapor receipts named the wrong sector AND the wrong provider.** `fetch_satellite`
   widens WV to its synoptic scope internally, but `_imagery_satellite` built the label and
   `satellite_source()` from the UN-widened region -- RJTY read "Himawari -- Japan (enhanced
   IR) / NOAA/OSPO" over a SLIDER full-disk image. The receipt is now built from the resolved
   region. Geocolor and infrared are unchanged (KWRI Northeast, RJTY Japan, ETAR Meteosat).
The other 8, briefly: the EUMETSAT disk cache cited the requested slot after a step-back
served an earlier one (now stores the served URL in a sidecar); `GET_LOOP`'s `product`
description still advertised the removed `visible`; `hazard_coords`' site-only justification
was wrong about `get_hazard_scan`, which DOES forward `location` (docstring corrected, and the
off-site path now names the site-only limit instead of listing locations); `_radar_degrade`'s
new `near` argument was never passed; the "no neighbour reported" reply counted the REQUESTED
subset as the roster size; `awc.station_latlon` was uncached under three geographic gates on
the busiest tool (now `lru_cache`); a short BUFR CSV row could `IndexError` past the guard;
and `build_neighbors.build()` would emit an EMPTY fetchable roster for a station added since
the last audit, because `reports()` returns 0 for silent AND unmeasured alike -- it now
refuses via the new `neighbor_rates.measured()`. `build_neighbors.py --check` PASSES: all 71
stations rebuild byte-identically.

**SLIDER IMAGERY HAD NO GEOGRAPHY ON IT AT ALL (owner-reported, then fixed).** A full-disk
water-vapour image with no coastline is not a readable product -- nothing tells a VLM which
moisture plume is over which country. **Cause: SLIDER serves RAW pixels.** The coastlines,
borders and graticule on its website are SEPARATE tile layers its browser app composites
client-side. GOES/STAR burns boundaries and a graticule into the JPEG and Meteosat IR
composites one server-side, so **SLIDER was the only provider serving a bare raster** -- the
same trap as the raw IEM radar rasters the module docstring already warns about. Fixed in
`imagery._slider_overlay`: fetch `lat` + `coastlines` + `countries` (draw order) from
`/data/maps/{sat}/{sector}/{layer}/{color}/{ts}/00/000_000.png` and alpha-composite. Applies
to **every SLIDER image, still and loop frame, all three products** -- IR was equally bare and
a night geocolor loop was ten near-identical dark tiles. Details that matter:
- The overlay layers are **STATIC** (`latest_times_all.json` reports one epoch stamp,
  19700101010000) and are memoized per process, so a 10-frame loop costs 3 extra requests.
- **Composite BEFORE the cache write**, so a cache hit serves what a live fetch would.
- **Any overlay failure returns the bare tile**, verified by forcing the fetch to raise -- this
  runs in the agent loop, where a degraded picture beats a dead tool call.
- **imagery.py now imports PIL.** Owner's call, over putting it in charts.py: the composite is
  inseparable from the fetch (it needs the sector and the served tile), and imagery.py already
  decides what the product IS. charts.py remains the only matplotlib file; PIL is not
  matplotlib. **`pyproject.toml` does NOT list pillow** -- it resolves transitively through
  matplotlib. Declare it.
- **`LoopFrame.data` no longer equals what `url` returns** for SLIDER (url gives the bare
  tile). Deliberate: the archive should hold what the model saw, and static layers keep the
  composite reproducible either way. Noted in the dataclass docstring.
- SIZING: **+9% on Himawari bytes, so the 1.54 GB/day archive figure stands.** SLIDER ships IR
  and WV as 8-bit PALETTE PNGs and writing them back as true colour TRIPLED them (185 -> 563
  KB) until the composite was requantized to 256 colours (222 KB, indistinguishable). Geocolor
  is already RGB and is deliberately not quantized. Full table in `docs/artifact_store.md`.

**Still open from this session:** `llm.routing()`'s docstring says it returns an empty body off
OpenRouter, but it only checks `llm_provider_order`, not the base URL -- with
`LLM_PROVIDER_ORDER=DeepInfra` in `.env`, a Together run would send an OpenRouter `provider`
body. Not fixed; the Gemma runs above cleared the variable per-command instead.

### SESSION 2026-07-28b -- Phase 1 CLOSED, imagery honesty pass, archive sizing settled
All UNCOMMITTED (18 modified + 4 untracked). New files: `scripts/audit_neighbors.py`,
`src/forecaster/neighbor_rates.py`, `docs/artifact_store.md`, `data/pi_export/tafs_20260729.parquet`.
Modified: `src/forecaster/{imagery,tools,awc,neighbors}.py`, `scripts/{build_neighbors,test_loop,
test_tool_fallbacks}.py`, CLAUDE.md, docs/archive.md. Self-tests: test_tool_fallbacks 5/5
(was 4/5), test_tool_stamps 5/5. ruff clean. Nothing reached the Pi.

1. **Neighbour audit (1.3) DONE -> Phase 1 complete.** Detail in the archive.md Build plan row.
2. **`visible` REMOVED as a product** (owner's call). Nothing gated it on solar elevation, so a
   night request returned a black image and a night LOOP returned ten of them -- ~7-8% of the
   archive in black pixels. `SAT_PRODUCTS` now drives both tool enums, so removing the key was
   the whole change; a stale `visible` argument coerces to geocolor via the new
   `imagery._goes_band` (GOES subscripted the dict and raised KeyError while every other
   provider already fell back).
3. **Water vapour PINNED TO SYNOPTIC SCOPE** (`imagery.synoptic_region`). WV is a jet/dry-slot
   product and was being served at the station's tight sector. **16 sector views -> 6 synoptic
   scopes, all 48 CONUS stations sharing ONE image.** geocolor/infrared keep the tight sector,
   verified unchanged. Widening happens BEFORE the cache key, which is what makes the sharing
   real. Judgment call flagged in code: the two Pacific-coast sectors map to `conus_east`
   because GOES-West's "CONUS" is PACUS; a west-coast forecaster may want `full_disk_west`.
4. **THREE METEOSAT DEFECTS, all found by RENDERING the images, not by reading code.**
   (a) `_EUMETSAT_LAYERS` held ONLY geocolor, so infrared and water_vapor fell through to it --
   SHA-256 confirmed all three were ONE image at europe/middle_east/africa. Same class as the
   KMIB/KRCA wrong-ocean bug: confident label, wrong content. Fixed by adding a real IR layer
   (`mtg_fd:ir105_hrfi` + `style_01`, which colours cold tops) and by REFUSING water vapour.
   (b) **The boundary overlay never worked.** `osmgray:all_boundaries_light` returns 0 KB alone
   and `geocolour+overlay` is byte-identical to geocolour; ruled out tile caching and WMS
   version. Only `ir105_hrfi` composites one, so IR is the only European product with borders.
   (c) **Omitting TIME does not return the latest scan** -- it returns a server default of
   unknown vintage. A no-TIME fetch at 00:07Z returned a BROAD DAYLIGHT Europe; an explicit
   23:20Z returned the correct night image. Every Meteosat image ever served carried an unknown
   valid time. Fixed: TIME is now always explicit.
5. **METEOSAT WATER VAPOUR REMOVED** (owner's call): refuses like radar via
   `imagery.meteosat_has_product` + `tools._meteosat_no_product`, text and NO image, gated in
   BOTH get_imagery and get_loop. MTG publishes no WV at all; the only candidate was MSG
   `wv062`, which is 6.2 um upper-level (not the 6.9 um the rest of the roster serves), raw
   greyscale, unlabelable, and on the retiring MSG leg.
6. **`awc._get` now retries HTTP 304.** We send no conditional header, so "Not Modified" is the
   edge cache misbehaving under load -- it killed a `build_neighbors` run on its FIRST call
   minutes after the 800-request audit, while curl to the same URL returned 200.
7. **`test_tool_fallbacks.py` 4/5 -> 5/5.** Its stubs declared `(name, *, fhr, run)` while
   `wxmaps.map_url` had gained `domain`. The FAIL was the small half: T8a was a FALSE PASS,
   "passing" because the stub's TypeError triggered the fallback rather than the simulated 403.
   Stubs now take `**kw`.
8. **Archive sizing + the artifact store contract** -- see ARCHIVER PREP items 5 and 6.

### SESSION 2026-07-28a -- archive.md Phase 0 + Phase 1 built (all but the neighbour audit)
All UNCOMMITTED. Files touched: CLAUDE.md, src/forecaster/{imagery,soundings,modeldata,tools}.py,
scripts/{archive_model_data,collect,schedule,test_modeldata,test_tool_stamps}.py, plus
docs/{archive,pi_setup_log}.md (gitignored). Self-tests: test_modeldata 124/124 (20 new checks),
test_tool_stamps 5/5, worksheet 19, tafgen 9, score_pending 29, results_report 49, tafstate 40,
agent 31, runlog 25. ruff clean throughout. NOTHING reached the Pi -- it pulls from git and this
is all working-tree.

Worked the "what must happen before the archiver" list in `docs/archive.md` (its Build plan
section is the authoritative tracker; every item below is marked DONE there too).

**0.4 `get_sounding` was mislabelling its product by 24h.** `soundings.synoptic_time()`
subtracted the post-lag THEN snapped, so it was not idempotent -- and the call path applied it
three times (tools -> fetch_skewt -> skewt_url/cache_path), so the receipt, the cited URL and
the delivered image were three different soundings. FIXED: an already-synoptic time returns
unchanged. Deliberate trade: passing exactly 12:00Z now MEANS 12Z (honest 404 if unposted),
which is what an archiver wants. This was the only defect that would corrupt an archive
silently, with nothing downstream to catch it.

**0.5 HRRR/NBM pinned to the synoptic cycles.** `modeldata.archive_run_and_as_of` snaps hourly
models back to the newest 00/06/12/18Z cycle that has had an hour to post, then pins just
before that cycle's HOURLY successor (a 6h pin on an hourly model selects the wrong run --
the same trap `prefetch_verification` already knew about). GFS/IFS untouched. ZERO credit
impact; trades up to ~6h of freshness for a cycle that NAMES a run, so two forecasts using the
same guidance are comparable. **This closes the standing "HRRR/NBM cadence" v2 question** in
favour of option 1, but explicitly rather than by accident of cron timing.

**0.1 `--no-model-data` retired.** BUFKIT *was* model data, so the arm described a state no
forecaster ever had. The get_model_* tools are ALWAYS granted and always named in the prompt
(naming is what makes models call them -- the 07-19 finding); an empty archive answers "not
pre-fetched" on its own, which is a truthful degrade. The credit-SPENDING prefetch survives as
opt-in `--prefetch-model-data`, since that was a cost control wearing an experiment's name.
`get_ensemble_prob` was in the drop-set but had never been named in the prompt -- now is.

**1.2 `get_loop` REBUILT off NASA GIBS onto each provider's own timestamped index** (STAR CDN
for GOES, RAMMB/CIRA SLIDER for Himawari, EUMETSAT unchanged). GIBS had two defects that
together made a loop unfreezable: it could only be probed BACKWARDS from now, so no loop could
be built for a past instant; and its NRT geocolor is too intermittent to loop, so every product
silently collapsed to clean-IR -- four products advertised, one served. Now frames are SELECTED
from a published index (a named frame is a frame that exists), `product` is honoured (verified
by eye, not byte hash), and `satellite_loop(..., at=T)` reconstructs a loop for any instant
still in the window. MEASURED provider windows: STAR ~10 days at 5-min cadence, SLIDER ~17h
(its index is the last 100 scans), EUMETSAT >=24h but it 500s intermittently on VALID times --
so frames get one retry and a dead frame is skipped rather than losing the whole loop. Trade
accepted: a GOES loop is now SECTOR-wide rather than station-cropped.

**ARCHIVE RAW FRAMES, NOT COMPOSED LOOPS (owner decision).** Measured at the widest case (10
frames, all four products, one station): **9,780 KB raw vs 9,755 KB filmstrip+mp4 -- a wash**,
so storage does not decide it. Capability does: only frames let replay subsample any
(frames, step_min) from one capture; re-composition is a pure function (offline, milliseconds);
it decouples the archive from `charts.py` so a rendering improvement reaches already-collected
rounds; and frames DE-DUPLICATE across stations sharing a sector (verified: PAED and PABI both
resolve to `alaska` and return identical frame URLs). `imagery.satellite_loop` now returns
`LoopFrame(time, label, url, data)` -- `label` is a display string with no year/month and the
REQUESTED time is not the SERVED time once `_select_times` snaps, so the store keys on
time+url. `tools._get_loop` composes. Generalises: get_map/get_imagery/get_sounding fetch
pre-rendered products so raw IS the product; only get_loop and get_fcst_sounding ever needed
the distinction, and both now archive inputs.

**KMIB AND KRCA WERE GETTING SATELLITE IMAGERY OF THE WRONG OCEAN (round-1 caveat).** Both
resolved to GOES-West CONUS, which is **PACUS** -- Hawaii and the eastern Pacific; neither
station is in frame. Found only by rendering the examples. NOT a loop bug: it lived in the
shared `satellite_region_for_latlon`, so `get_imagery` did the same throughout round 1 (called
in 38.7% of runs). Two stacked causes, both FIXED: (a) the `northern_rockies` bbox stopped at
-104 while the real `nr` sector reaches the upper Mississippi (Iowa visible at its right edge,
confirmed by fetching it), leaving the Dakotas in a gap before `upper_mississippi` at -98 --
east edge widened to -99, nearest-center rule arbitrates the overlap; (b) the CONUS fallback
split birds at -100, but only GOES-EAST's CONUS product covers the continental US -- it now
always returns GOES-East. `conus_west` is retained but RELABELLED "Pacific/West (GOES-West
PACUS)" so it cannot read as a drop-in again. **Do not analyse per-station satellite behaviour
at KMIB/KRCA from round 1.** RJTY's Himawari `japan` tile is 51.5% black padding (full_disk is
23.6% but hemisphere-wide) -- owner DECIDED to keep it; regional content beats a clean
hemisphere for a TAF.

**0.2 the credit estimator under-reported 5.4x, not the ~3x the doc guessed.** Replaced by
`modeldata.estimate_prefetch_many`, built from the SAME helpers the fetch uses (`_time_grid`,
`_applicable_models`, `_surface_vars`/`_profile_vars`/`_hazard_vars`, `_chunk`) rather than
restating the cost model -- the restatement is what drifted. Four independent errors: it
omitted `_profile_vars` ENTIRELY (5 vars at every pressure level, the dominant term since the
sounding tier moved off BUFKIT), used `hours//step + 1` and so ignored the 6h pre-anchor tail,
called `_hazard_vars` without `profiles=` (double-counting the T/RH the merged bundle already
pays for), and iterated MODELS instead of `_applicable_models`.
**REAL COSTS (48h/2h/3h, one 500-coord chunk): ~5,882 credits/pull, not ~1,701.** Per model --
gfs 252 surface + 2,016 levels (112 vars), hrrr 252 + 1,890 (105), nbm 224 + 0, ifsoper 168 +
1,080 (60). **The level ladder is 85% of the bill** (4,986 of 5,882), so `--hazard-step` and
the level list are the ONLY levers that matter; halving the level cadence to 6h saves ~2,500.
At 4 pulls/day that is ~23,500/day, ~706k per 30-day round for the deterministic tier alone.
**STATION-COUNT CLIFF (matters for round-2 selection): ~42 surface + ~37 level coords per
station and points are free below 500, so credits are FLAT from 1 to ~11 stations** (verified
2 through 10, all ~5,882). A second SURFACE chunk starts ~12 stations (+896); a second LEVEL
chunk ~14 (+4,986, roughly double). An 11th station is free; a 14th doubles the tier.
archive.md's whole credit table was computed from the broken formula and is marked superseded.

**0.3 budget cap.** `--max-credits` in archive_model_data.py, **default 12,000**, `0` disables.
REFUSES with exit 2 and spends nothing rather than truncating -- a half-archived cycle looks
complete while silently omitting models or hours. Sized to pass the INTENDED config (~10,625 =
5,882 deterministic + 4,743 full GEFS) and trip on the ~14-station doubling (~16,500). A first
cut at 8,000 refused the intended config; a ceiling that blocks what you meant to run is a
papercut, not a guardrail. **STILL MISSING and not settable from here: the per-key hard cap at
GRIBStream and at each LLM provider** -- round 1's 402 was exactly that absence.

**1.1 `prefetch_ensemble_many`.** Unions ONE point per station into a single bundle. Members
bill linearly but coordinates are free below 500 and the ensemble is site-only, so the bill is
identical for 1 station or 400 -- verified flat 1 to 40. Full 31 members at 48h/3h = **4,743
credits**, matching the earlier projection. `prefetch_ensemble` is now a thin wrapper over it
(same return shape; consensus_experiment.py unaffected). **Also fixed the GEFS grid-snap bug**
(consensus-experiment bug (c), still open until today): it built its own unsnapped grid, so an
odd issue hour like 22Z gave 22/01/04Z, matched NONE of GEFS's 00Z-anchored times, and returned
zero rows, zero credits and NO error. Both the level bundle and the ensemble now share
`modeldata._snapped_grid`, and an empty ensemble return carries an explicit "do not treat this
cycle as captured" note instead of a clean zero. Wired in behind `--ensemble` /
`--ensemble-members` (default all 31: members not fetched cannot be recovered later).

**PI RUNBOOK LANDMINE DEFUSED.** `docs/pi_setup_log.md` instructed adding `--model-data` to the
scheduler cron when enabling the tier. That flag is gone, so following the runbook would have
made `schedule.py` exit "unrecognized arguments" and killed the whole hourly dispatch. Corrected,
and noted there that the archive cron has no `--ensemble` (so GEFS is not pulled) and that
`--max-credits` must be re-checked if the station list grows.

**CORRECTION TO A ROUND-1 CLAIM.** The "get_loop called ZERO times in 592 runs" statistic
recorded below is NOT evidence models avoid the tool -- owner clarified get_loop postdates the
start of round 1. Treat it as unmeasured. `scripts/test_loop.py` is a COMPREHENSION test (the
only tool offered); Gemma called it and read the motion correctly on 2026-07-28. **No SELECTION
test exists** -- whether a model reaches for a loop unprompted with the full toolset is still
unknown, and worth settling before committing to archive four products per station.

### ARCHIVER PREP -- 8-item list, items 1-3 settled 2026-07-28
The owner and I worked a pre-archiver checklist one item at a time. State of each item:

1. **Credit ceiling -- CLOSED, no code change.** GEFS is OUT of the archive (owner's call).
   A pull is then ~10,362 credits, under the existing 12,000 `--max-credits`. WITH the
   ensemble it is ~15,107 and the cap refuses. `prefetch_ensemble_many` and
   `get_ensemble_prob` stay in the code, unused; the cron just never passes `--ensemble`.
   Steering probe (`MODEL_DATA_FLOW_RELATIVE`) stays OFF: it adds 6 upstream points per
   station (220/330 km, vs the ring's 165 km) for +1,322/pull, but model data is the one
   layer `asOf` can rebuild later, so it buys nothing that cannot be bought later.
2. **`MODEL_DATA_ENABLED` -- DECIDED: stays `false` for now.** See item 0 below.
3. **Per-key cap at GRIBStream -- CONFIRMED IMPOSSIBLE (owner checked 2026-07-28).**
   GRIBStream has NO API-key ceiling. It has a dashboard showing credits used per day, and
   that is NOT a hard stop. **So there is no provider-side brake at all, and the repo is the
   only brake.** Today the repo has a PER-PULL ceiling only. `credits_charged` is printed at
   the end of a pull and then discarded -- nothing in `store.py` persists it. So four
   scheduled pulls (~41k/day) and eight pulls (~83k/day) look identical to our code: each is
   individually under 12,000. A cron misfire, a manual `--force` on top of the schedule, or a
   retry after a partial failure all re-bill in full and nothing notices. After a 30-day
   unattended run nobody could answer "what did this cost" from the DB.
   A `model_data_pulls` spend table was PROPOSED and **REJECTED by the owner 2026-07-28**.
   Do not re-propose it without new evidence. The consequence is accepted deliberately:
   **there is no daily brake, on either side of the wire.** The per-pull `--max-credits`
   ceiling is the whole control. Watch the GRIBStream dashboard by hand.

  4. **Neighbour audit (Phase 1.3) -- DONE 2026-07-28.** Full detail in the archive.md Build
     plan row; do not restate it here. Phase 1 is now complete.
  5. **Storage + retention -- SETTLED 2026-07-28. The old "~2-4 GB/day, 50-130 GB" estimate
     was a guess and is superseded.** Derived from the Pi's own bulletin archive (2,364 TAFs,
     63 stations, pulled 2026-07-29) plus the SH 8 at their documented 6-hourly cycle: **all
     24 hours are some station's issue hour**, so the archiver captures hourly -- but only the
     regions in play at that hour, which is what per-product keying buys. **1.54 GB/day, peak
     hour 11Z at 108.6 MB (22 stations).** Owner capped a round at **30 days wall clock**, so a
     round is **at most 46 GB** against 107 GB free. Retention is the one piece still open, and
     it is no longer urgent: one round fits easily, but TWO resident rounds is 92 GB of 107, so
     the working assumption is harvest-and-prune BETWEEN rounds. Decide before a SECOND round,
     not before the first. Numbers live in `docs/artifact_store.md`.
  6. **Artifact store shape -- DRAFTED 2026-07-28 in `docs/artifact_store.md`.** Content-
     addressed blobs + three index tables. The load-bearing decision is **key on resolved
     product identity + time, NEVER on station** (71 stations collapse to 16 satellite regions,
     5 WV scopes, 7 chart sets, 52 sounding sites). `run_manifest` is written at **CAPTURE
     TIME**, not derived at replay -- the KMIB case settles it, and the reasoning is in the doc.
  7. Serve-tools-from-archive vs push-as-context. **THE LAST DECISION BEFORE BUILDING**, and
     the only one of these that touches `tools.py`.
  8. Build Phase 3: `archive_run.py`, then serve from the archive, then dry-run + diff.
  9. Commit and deploy to the Pi. **LAST** (owner, 2026-07-28): the archiver is built and
     tested on the laptop, so deployment is a consequence of the work, not a prerequisite.
     An earlier draft of this list put deploy at item 4; that was wrong.
     **EXCEPTION, added 2026-07-28: the TAF poller is NOT part of that "last".** The Pi polls
     only 63 stations, so the 8 SH sites are archiving NOTHING, and a TAF is not backfillable
     -- AWC serves only the current bulletin. See PICK UP HERE item 5.

RECOVERABILITY, the rule that orders all of this: imagery, maps and loops CANNOT be
re-fetched for a past instant, and analysis charts have no time parameter at all. GRIBStream
serves past runs via `asOf`. So spend the irreversible effort on pixels; model data can be
backfilled. **ONE EXCEPTION, found 2026-07-28: EUMETSAT is time-queryable back to 2020** (MTG
from 2024-09-23) at a 10-minute cadence, so Meteosat frames ARE recoverable -- the only
provider that is. GOES STAR holds ~10 days, SLIDER ~17 hours.

### PICK UP HERE (2026-07-28)
0. **`MODEL_DATA_ENABLED` MUST BE FLIPPED TO `true` ON THE PI before the archiver can bill.**
   The Pi is the machine that will run `archive_model_data.py`. The flag is deliberately still
   `false` (owner's call 2026-07-28) so the cron cannot spend by accident while Phase 3 is
   being built; use `--force` for deliberate one-off test pulls until then. It is read in
   exactly ONE place (`archive_model_data.py:81`) and by nothing in `collect.py` or the agent
   toolset, so flipping it starts GRIBStream spend and CANNOT start LLM spend. NOTE the Pi has
   its OWN hand-transferred `.env` -- changing the laptop's does nothing to it.
   GEFS is OUT of the archive (owner, 2026-07-28): the code stays, the archive cron just never
   passes `--ensemble`. That keeps a pull at ~10,362 credits, under the 12,000 `--max-credits`
   ceiling, so no cap change is needed. WITH the ensemble it is ~15,107 and the cap refuses.
1. **COMMIT *AND PUSH* the working tree, then pull on the Pi.** 20 modified + 4 untracked, on
   top of the model-screen, BUFKIT-removal and nowcast work. **THE PUSH IS THE PART THAT KEEPS
   GETTING MISSED:** verified 2026-07-28, `origin/main` is at **50823b1** while local `main` is
   at **8b7f107** -- FOUR commits exist only on the laptop. The Pi cannot receive ANY of this
   until they are pushed.
2. **`.env` has `llm_model` EMPTY** and points at OpenRouter, so anything defaulting to
   `settings.llm_model` 400s ("No models provided"). Today's runs used explicit `--model`.
3. **Per-key hard credit caps at GRIBStream and each LLM provider** -- the half of 0.3 that
   cannot be done from the repo.
4. **THE PI IS ARCHIVING TAFs FOR ONLY 63 OF 71 STATIONS -- the 8 SH sites capture NOTHING,
   and a TAF cannot be backfilled** (AWC serves only the current bulletin, so every hour of
   delay is permanently lost data at exactly the sites the winter comparison needs). Verified
   2026-07-28 end to end: the Pi's `poll_icaos()` returns **63**; its TAF archive holds 2,364
   bulletins across **63** stations; the SH 8 exist ONLY in the uncommitted working tree
   (`git show HEAD:src/forecaster/stations.py` has no trace of them).
   **NO CODE CHANGE IS NEEDED** -- `poll_tafs.py:32` already calls `stations.poll_icaos()`, so
   the pull alone fixes it. Both upstream risks were cleared: AWC serves live TAFs for all 8,
   and all 8 survive the EXACT poller path (`awc.fetch_taf_rows` -> `store.insert_taf`, 8/8,
   zero errors, typed `routine`) including their negative-temperature forms (`TNM07/2900Z`,
   `TXM01`) and `CAVOK`. So this is purely a delivery problem: commit, push, pull.
   **CORRECTION to the PI STATE block below: the Pi is NOT on an orphaned commit any more.**
   Checked read-only 2026-07-28 -- HEAD `50823b1` IS an ancestor of `origin/main`, 0 behind,
   working tree clean. **A plain pull will fast-forward; the `git reset --hard` that block
   recommends is no longer needed.**
   SSH NOTE: the Pi **blocks ICMP**, so ping fails and a first ssh can report "No route to
   host" while port 22 is open and healthy. Do not read a failed ping as a dead Pi.
5. **Round-2 station list -- SETTLED 2026-07-28: 71 stations** = the 63 already polled + the 8
   Southern Hemisphere (Patagonia) winter sites, now in `stations.ARCHIVE_STATIONS`
   (`poll_icaos()` -> 71 unique; all 8 verified live on the poller path, SAWC already reporting
   `2000 SN BKN005`). The decision is **archive wide, run narrow**: freeze inputs at all 71
   (irreversible), choose the billed VLM subset later off the frozen bytes (reversible).
   **No VLM runs are happening now -- archiver only, model runs batched later.**
   COST CONSEQUENCE + THE FIX (traced 2026-07-28): 71 stations cross into 6 coordinate chunks
   on BOTH tiers, so GRIBStream goes ~5,605 -> **~33,630 credits/pull, ~134.5k/day, ~153k with
   GEFS**. (An earlier note here said 35,292/141k -- that assumed 18 level valid times; it is
   **17** unless the issue hour misses the 00Z-anchored 3h grid. Per chunk: surface 28 times x
   32 vars = 896; levels 17 x 277 = 4,709.)
   **A station is NOT a coordinate -- it is 42 of them**: 1 site + 5 neighbours + a **36-point
   ring grid** (`GRID_RADII_DEG` 0.5/1.0/1.5 deg x 12 bearings every 30 deg). 10 stations = 420
   coords = under the free 500 cap; 71 = 2,982 = 6 chunks. The grid is what turns stations into
   credits.
   **AND THE GRID IS FETCHED WITH THE FULL 112-VARIABLE PRESSURE LADDER, which no tool reads.**
   `get_nearby_model_data` reads ONE SURFACE alias at a time; `get_fcst_sounding` and
   `get_hazard_scan` both default to the SITE column via `_resolve_md_location(con, station,
   None)`. So ~29,900 of the 33,630 credits/pull buy level data at points nothing looks at.
   Three configurations: (A) today 33,630/pull; (B) **levels at the site column only ->
   10,085/pull, 40k/day, loses NOTHING any tool reads** -- the recommended default, and it fits
   the existing 48k/day tier; (C) B plus stripping the surface ring grid off-run -> 6,501/pull,
   26k/day, but `get_nearby_model_data`'s gradient view stops working at trimmed stations.
   **CONFIG B ADOPTED 2026-07-28**: `modeldata.hazard_coords` is now SITE-ONLY
   (`include_grid=True` restores the grid). `archive_model_data.py` also now iterates
   `poll_icaos()` (71) rather than `icaos()` (10) -- it would otherwise have left 61 stations
   with NO model data at all. Verified by `--dry-run`: **~10,085 credits/pull**, 2,921 surface
   coords (6 chunks, deduped across stations) + 71 level coords (1 chunk). ~40k/day, inside the
   48k/day tier.
   **THE PULL COST MOVES WITH THE ISSUE HOUR, and that is the whole gap between the 10,085
   here and the 10,362 in ARCHIVER PREP item 1 -- they are the same config, not two configs.**
   The level grid is 00Z-anchored at 3h, so an hour that misses it gets 18 valid times instead
   of 17, and one extra level valid time costs exactly 277 credits (112 gfs + 105 hrrr + 0 nbm
   + 60 ifsoper). Re-measured 2026-07-28d: `--dry-run` printed 10,362 with 18 level times and
   2,904 surface coords. Both figures are correct, both are under the 12,000 cap, and the
   worst case is the one to plan against. Each station's 5 fetchable NEIGHBOURS are in the surface set (KWRI ->
   KNEL/KVAY/KMJX/KTTN/KPNE), so nearby-site model data is covered.
   **A vs B is INVISIBLE to the agent** (both render the site column, byte for byte). **B vs C
   is visible in exactly one tool**: get_nearby_model_data goes from 44 lines to 8. The 36 ring
   points are the only samples reaching past the airfield cluster -- at KWRI every neighbour is
   within ~40 km and reads 23-24 C while the 1.5-deg NW grid point (~165 km) reads 20.3 C, so
   under C an advecting front is invisible until it is already in the neighbour obs.
   Levels stay recoverable later via `asOf`, unlike the live-fetched imagery/maps/loops.
   Storage: the "~50-130 GB per 30-day round" figure once here was a GUESS and is SUPERSEDED --
   measured 2026-07-28 at **1.54 GB/day, at most 46 GB per round** (see ARCHIVER PREP item 5
   and `docs/artifact_store.md`). It fits, so harvest-and-prune matters only between rounds.
   Full reasoning + the cycle-hour data in docs/archive.md "Station list --
   SETTLED". It unblocks, in this order:
   `build_neighbors.py` for any NEW station (**`neighbors.py` currently covers ONLY the 10
   current roster stations** -- the archive.md plan omits this prerequisite; it is now 61
   stations short, and it degrades SILENTLY to an empty list), then 1.3 the
   neighbour audit, then 2.2 `upper_air_sites.py`, then 2.3 climo months.
   CYCLE HOURS: `ArchiveStation` has no `cycle` field and does not need one hand-built --
   measured off the 2,009-bulletin archive, **all 63 issue routine TAFs (1.4-2.9/day) and 55
   of 63 have a clean 8-hourly cycle**; the ragged 8 are ETAD/ETAR/KBAD/KDMA/KDYS/KFHU/KRND/
   KSUU. Peak load is **14 stations at 02/10/18Z**. The SH 8 are DENSER and unaligned:
   24h validity on **6-hourly** issue (valid 00/06/12/18Z, issued ~1h prior) = 32
   station-cycles/day, not 24.
   **RE-DERIVED 2026-07-28 off a fresher 2,364-bulletin export** (`data/pi_export/
   tafs_20260729.parquet`, pulled from the Pi): counting a station's cycle hours as those
   seen >=2 times, **all 24 hours are some station's issue hour** and the peak is **22
   stations at 11Z**, then 19 at 17Z. The "14 at 02/10/18Z" figure above counts something
   narrower; for ARCHIVER CAPTURE PLANNING use 24 hourly captures and the 11Z peak.
   TOOL FIXES FOR THE NON-US SITES -- **satellite / radar / maps / nearby-obs all DONE
   2026-07-28**; only soundings remain:
   - **Satellite**: `SatRegion("south_america_south", ... "SECTOR/ssa", "1800x1080")`. All 8 SH
     sites had been resolving to `full_disk_east` -- not an error, a silent collapse to a
     useless zoom. Verified by eye; no NH station moved.
   - **Radar**: `tools._radar_no_coverage` -- OUT-OF-NETWORK stations now get a message and NO
     IMAGE, for every product incl. an explicit `national_mosaic`. **This also fixes RJTY**,
     whose "radar" was a CONUS composite all through round 1.
   - **Maps**: `wxmaps.charts_for_latlon` + a station gate in `tools._get_map`. Station is
     threaded from `AgentConfig.station` through `run_tool`, NOT read from model args, so the
     gate cannot be bypassed by omitting a field. SPC fallback gated to `domain == "us"`; TT
     domain added to the CACHE KEY (else a cached CONUS panel serves a South American request).
     **TT domain codes are SHORT -- `eu`, `me`, `aus`, `ak`, `ea`, `samer`; enumerate them from
     the region menu, never guess.** A guess-based pass wrongly concluded Europe had no
     coverage. **A 404 on our field name does not mean a domain is sparse**: `wpac`/`cpac` serve
     23-24 packages under different names (`uv200` not `uv250`, `mslp_pcpn` not
     `mslp_pcpn_frzn`) -- see `_TT_FIELD_OVERRIDES`. All 71 stations now resolve to charts.
   - **`get_nearby_obs`**: says "no obs in the local region" instead of returning blank, and
     distinguishes no-neighbours-exist from neighbours-reported-nothing.
   - **`get_sounding` -- DONE 2026-07-28, new `bufr` source.** `soundings.inventory()` reads the
     provider's own INVENTORY index (ONE request per station-year; every cell link embeds a full
     `datetime=`), `latest_time()` takes the newest launch at or before the cutoff, and
     `fetch_profile()` parses TEXT:CSV (wind in **m/s** -> kt) and thins ~3,300-4,500 one-second
     levels to ~70 by PRESSURE BIN (not every-Nth, so it does not depend on ascent rate), the
     lowest 150 hPa ~4x finer. `soundings.py` still owns no matplotlib -- it returns an
     `ObsProfile` and `tools._sounding_bufr` renders via `charts.skewt` (which gained an optional
     `title` so an observed ascent is not captioned "forecast"). **It NEVER snaps to 00/12Z**:
     87155's 2026 inventory has 211 launches, **14 off-cycle (all 15Z)**, and an off-cycle ascent
     is released BECAUSE something is happening -- a synoptic snap systematically discards the
     most informative soundings (same for CONUS 06Z/18Z special ascents). **This SUPERSEDES
     Phase 0.4** -- 0.4 makes the snap self-consistent, availability makes it unnecessary.
     Endpoint MOVED: `cgi-bin/sounding` 404s, use `/wsgi/sounding`. `PNG:SKEWT` under BUFR is an
     HTML wrapper, not an image -- that is why we render.
   - **`upper_air_sites.py` -- BUILT 2026-07-28 (Phase 2.2 DONE).**
     `scripts/build_upper_air_sites.py` freezes the nearest 3 radiosondes for all 71 stations
     from the **NOAA IGRA v2 station list** (stable public text, no auth; network `M` ids carry
     the WMO number in their last 5 digits; 916 sites active since 2024). `get_sounding` now
     takes a station ICAO and says which site it used and how far ("KWRI ... nearest, 72501
     UPTON NY, 174 km ENE"). Distances independently reproduce the hand-derived ones in
     archive.md (SCCI 1 km, SAWG 187 km, SCBA 328 km). GOTCHA: IGRA `[72:76]` is the FIRST year,
     `[77:81]` the LAST -- reading the wrong one yields "(none)" everywhere and looks like a
     coverage failure, not a parse bug.
   - **TT substitution policy (owner, 2026-07-28): serve the BEST AVAILABLE panel, do not
     withhold** because there is no perfect 1-for-1 match. wpac/cpac have NO temperature field at
     any level (T850/T850a/T925/T700/T2m/Td2m all 404), so the 850 mb slot serves `mslp_uv850`
     (height+wind) RELABELLED -- the receipt says what it actually shows and flags the
     substitution. All 4 C-charts now resolve at every one of the 71 stations.
   Design note for 1.3: audit a reporting RATE over a window, not a single probe -- KNEL
   returning nothing once is not proof it never reports, and a one-shot miss would permanently
   mislabel the terrain map that is the model's menu for get_nearby_obs.
5. Then Phase 3: `scripts/archive_run.py`, serve-tools-from-the-archive, dry-run + diff.
6. Optional and cheap: a get_loop SELECTION test (full toolset, motion-flavoured task) to decide
   whether the tool earns its archive slot.

## Round 1 result and standing state

### ROUND 1 IS OVER. Live collection STOPPED 2026-07-20 ~14:20Z (owner's call, cost).
The scheduler cron is removed from the Pi and the last in-flight event drained, so NOTHING
bills any more. Round 1 ran 2026-07-16 20Z -> 2026-07-20 14Z (3.7 days, 10 stations,
587 model runs, **$38.32** list-price LLM spend total).

**HEADLINE RESULT -- SUPERSEDED 2026-07-20 evening; see the corrected figures below.**
The numbers first recorded here (human 85.8 > model 83.8 > persistence 82.2, gaps -1.98 and
+1.55) were computed at scorer_version **1** AND by AVERAGING per-evaluation percentages,
which violates the anti-averaging rule. Both are now fixed; regenerate any figure with
`scripts/results_report.py` rather than re-deriving it. Report artifact (private, live,
NOT yet refreshed for v2):
https://claude.ai/code/artifact/d306bcc0-9a76-4a01-a3e5-97a274fcaba1
Site map artifact: https://claude.ai/code/artifact/cbac15fa-58cb-45a0-bb30-538de481c3f1

**CORRECTED HEADLINE (scorer_version 2, POOLED points, n=231 human-paired):**
**human 85.84 > model 83.23 > persistence 82.23**, and over all 298 scored evaluations
human 85.84 > human_composite 83.96 > model 82.61 > persistence 80.63 > **climatology 78.13**.
Paired per-evaluation gaps: model-vs-human **-2.72** (+/-0.50 SE), model-vs-human_composite
-2.20 (+/-0.52), model-vs-persistence **+1.53** (+/-0.57). All three resolved at >=2 SE.
The model-vs-persistence margin is unchanged from the original read; the model-vs-human gap
WIDENED from -1.98 to -2.72 because the v2 rescore (remark stripping + WND...AFT overlay)
raised human scores 85.15 -> 85.84 while the model did not move -- exactly the effect the
rescore was predicted to have. Per element (model vs human, paired): the human leads present
weather (+6.1), wind direction (+4.7), altimeter (+4.3), ceiling (+3.9) and wind speed (+3.1);
the model leads visibility (+0.8) and ties gusts (persistence beats both at 88.2). Present
weather is everyone's weak spot (33-40%). Model BEATS human at KMIB.

**ABLATIONS SHRANK vs the 07-19 read** -- with ~2x the sample, prior-TAF access is worth
only ~0.6 pt (not 2.5), and NO cell-to-cell difference clears its ~1 pt SE. Settling them
needs 4-10x more data, which is why round 1 was ended.

### WORKSHEET DECISION -- a THIRD side arrived 2026-07-20 (owner to decide for round 2)
The worksheet COSTS ~0.8 TAFVER points AND DOUBLES cost per run ($0.036 no-worksheet vs
$0.062 control) while halving output tokens. Its remaining defensible value is convergence
and reliability, not score or cost. Options: drop it, make it advisory, or redesign. This is
the cheapest single round-2 win available.
**NEW EVIDENCE (lead-time decay, this session):** the no-worksheet cell has the HIGHEST
short-range score (87.5 at 0-3h) and one of the STEEPEST decay rates (-1.84 TAFVER pts per
10h of lead), while the control has by far the FLATTEST curve (-0.29/10h; Gemma -0.91,
temp0.2 -1.48, no-priorTAF -1.13, Kimi -3.16). So the aggregate "-0.8 points" may be
averaging away a real trade: worse nowcast, better long range. A 30h TAF is mostly long
range, so this is worth settling before dropping the worksheet. CAVEAT: single-cell slopes,
no error bars yet -- suggestive, not resolved. Cheap to settle on existing round-1 data.

### PI STATE (verified 2026-07-20 ~17Z) -- 4 crons, ZERO billing
`poll_tafs.py` */5 (archives ALL bulletin types for 63 stations), `score_taf.py --pending
--backfill iem` 35 */6 (goes IDLE ~7/22 when the 122 flush), `score_taf.py
--archive-difficulty --backfill iem` 50 4 (NEW this session; VERIFIED working on KDMA --
8 TAFs, 100% coverage), `cloudsync.sh` 15 8. Model-data tier unset in .env -> OFF.
~~Pi may still need `git fetch && git reset --hard origin/main`~~ **RESOLVED -- do not run
that reset.** Checked read-only 2026-07-28: the Pi is on `50823b1`, which IS an ancestor of
`origin/main`, 0 commits behind, working tree clean. It is no longer on the orphaned c213d05,
so a plain pull fast-forwards. The blocker is on the LAPTOP side: 4 commits are unpushed.
Also: the Pi blocks ICMP, so a failed ping does not mean a dead Pi -- see `docs/pi_setup_log.md`.

### CLIMO FOR ROUND 2 -- NOT a deadline (corrected 2026-07-27)
This block previously read "HARD DEADLINE -- August climo before 2026-08-01". **That was
wrong and it misdirected planning three times before anyone checked the code.** There is no
calendar constraint: `climo.build` iterates `range(climo_start_year, climo_end_year + 1)` =
2006..2025 (`climo_end_year` is a FIXED config value), and the raw history lands in a scratch
DuckDB that is discarded -- it never touches the runtime `obs` table. Building August climo
DURING August 2026 pulls Augusts 2006-2025, the identical row set, and the build is
deterministic + idempotent, so the output is byte-identical whenever it runs. The likely
origin of the false deadline is a conflation with the obs-leakage problem, which the
scratch-DB design already solved on 2026-07-09. A date would only matter if `climo_end_year`
were bumped to 2026 -- a config decision, not a calendar one.
THE ACTUAL CONSTRAINT: build the right MONTHS for the right STATIONS before round 2 needs
them, ONE STATION AT A TIME (concurrent IEM builds 503 -- see [[iem-single-client-builds]]).
So it is GATED ON ROUND-2 STATION SELECTION (see above), not on the calendar; building early
for a guessed station list is wasted work. Cost is ~40 IEM requests per station-month
(20 years x 2 report types) = ~5-10 min per station at the throttle, so a 10-station build is
an hour or two of unattended time whenever it is wanted.

### SME GOLDEN FIXTURE -- SUBMITTED 2026-07-20, expect 1-2 weeks
Sent blind (our 88.7 withheld): KDMA 181100Z routine 30h TAF (TEMPO + 4 BECMG, monsoon
convection) + 47 obs, `logs/sme_fixture_KDMA_181100Z.md`, with a 33-row hourly worksheet
(`sme_worksheet_KDMA_181100Z.csv` -- 30 hourly rows + 3 TEMPO overlay rows, because A7.1's
combined formula sums pcf/ap PER GROUP) and `sme_summary_KDMA_181100Z.csv`.
CONFIRMED FROM PRIMARY SOURCE this session (ACCI15-120 Att 7, rendered from docs/TAFVER.pdf):
**TAFVER is HOUR-BY-HOUR, not ob-by-ob** -- Table A7.1's column is headed "Hourly Score",
every element reads "the hourly score is one point...", and the altimeter row ("if the lowest
altimeter observed during a given hour...") only makes sense if multiple obs reduce to ONE
score per hour. The unit is (group x hour): Combined TAF Accuracy = points "for every hour in
the TAF for all groups", `(BECMG pcf + TEMPO pcf + FM pcf)/(BECMG ap + TEMPO ap + FM ap)`.
Our implementation matches (33 opportunities over 30 hours for this TAF). Six doctrine
questions were asked; the ones that most affect the score are BECMG attribution, the
best-of BECMG transition window, and within-hour ob selection (A7.1 specifies it ONLY for
altimeter -- our pessimistic reading for cig/vis/wind/wx is a documented GUESS).
NOTE: no FM group exists in ANY archived TAF (these forecasters favour BECMG), so the fixture
validates INITIAL/TEMPO/BECMG only; a second FM fixture was offered.


## Open work (consolidated 2026-07-28)

This list replaces the old "Likely next steps" and "STILL OPEN FOR ROUND 2" blocks.
Steps 1-9, 11 and 12 of the old list are all DONE; they are in
`docs/history_2026-07.md` if you need the record. Items here are open.

### Blocking round 2
1. **COMMIT *AND PUSH*, then pull on the Pi.** 20 modified + 4 untracked, on top of the
   model-screen, BUFKIT-removal and nowcast work. `origin/main` is 4 commits behind
   local `main` -- the push is the step that keeps being missed. This now BLOCKS DATA
   CAPTURE, not just deployment: see PICK UP HERE item 4 (the 8 SH sites archive no
   TAFs until the Pi pulls, and a TAF is not backfillable).
2. **Per-key HARD credit caps at GRIBStream and at each LLM provider.** The repo
   cannot set these. Round 1's 402 halt was exactly this guardrail missing.
   [[together-credit-limit-402]]
3. **Scheduler timeout fix.** 121 round-1 cells died on the 30-min per-cell timeout,
   clustered on heavy issue hours. Raise the timeout, stagger cycle hours, or restore
   parallelism on a provider whose tier allows it.
4. **Serve-tools-from-archive vs push-as-context** (archiver prep item 7). The last
   decision before Phase 3 can be built, and the only one that touches `tools.py`.
5. **Round-2 climo months.** Gated on station selection, NOT on the calendar (see the
   CLIMO FOR ROUND 2 block). Build ONE station at a time -- concurrent IEM builds 503.
   [[iem-single-client-builds]]

### Round-2 decisions the owner still owns
1. **Worksheet: keep, make advisory, or redesign.** It costs ~0.8 TAFVER points and
   DOUBLES cost per run, but has the flattest lead-time decay curve. Cheapest single
   round-2 win available. See the WORKSHEET DECISION block.
2. **Provider migration + `ping_models` re-verify.** Prices move weekly; `Cell` needs
   an optional (base_url, key) override and a `provider` column. See
   `docs/model_costs.md` and the ROUND-2 MODEL CANDIDATE SET block.
3. **`get_sounding` and `get_loop`.** get_sounding was selected in 0.8% of round-1
   runs -- drop it or rewrite its description. get_loop is UNMEASURED (it postdates
   round 1); run a SELECTION test with the full toolset before giving it four archive
   slots per station.
4. **Envelope consensus.** The consensus experiment showed a MEAN smooths hazard
   extremes downward. Build an arm that shows MAX / high-percentile for gust and wind
   while keeping the mean for smooth scalars, and re-run the same 3 cases. Cheap --
   the model data is already archived under `data/consensus_experiment/`.
5. **Gust bias.** Gusts are the model's worst element (persistence beats it), and
   GFS over-forecasts them by 8-22 kt. GRIBStream has a gust field, so feeding it raw
   without the bias could make gusts WORSE. Independent of everything else here.

### Deferred, not scheduled
- **Raw-NWP baseline leg** (the open half of the old step 9). Comparing against
  GFS/GALWEM needs archived model forecasts; round 1 stored none. Reconstruction via
  `gribstream.fetch_points` with `asOf` costs ~500 credits per station-window, so
  scope ONE high-signal station (KVBG) first.
- **AF metric harness** for OPVER and WARNVER.
- **SuperCloud**: Podman images, pre-staged weights, vLLM serve job.
- **Live `MODEL_DATA_ENABLED` run** to confirm the agent actually READS
  get_model_verification in the loop. It has never been in a live agent's toolset.
- **Archive Phase 3**: `scripts/archive_run.py`, serve-tools-from-the-archive,
  dry-run + diff. See the archive.md Build plan.

## Process note (2026-07-20)
I committed AND pushed unasked this session; the owner reset it and force-pushed their own
commit. **See the VERSION CONTROL IS MINE hard rule at the top of this file** -- never run a
state-changing git/gh command, including `git pull` on the Pi. Leave work uncommitted in the
working tree and say what changed. [[never-run-git-write-commands]]

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
  cannot leak truth). But the live network tools fetch whatever is posted at WALL-CLOCK time: a
  cell that executes minutes-to-an-hour after the pinned issue time sees slightly fresher imagery
  than the human forecaster had at issue. Input-fidelity drift, not
  truth leakage (the verifying METARs are never exposed) -- acceptable for this ~1-week run, but
  before the NEXT big collection, pin those inputs to issue time: snapshot-and-archive the network
  products at (or just before) the issue hour and serve the agent the frozen copies (the M4
  "snapshot-and-replay (b)" path; also what makes historical valid times airtight and enables
  Batch API / multi-model replay off identical inputs).
  **NARROWED 2026-07-28.** Exactly six tools still fetch live: get_current_taf, get_sounding,
  get_map, get_imagery, get_loop, get_terrain. Each is stamped with its fetch time by
  `tools._stamp_fetched`. `get_point_forecast` and `get_fcst_sounding` came OFF this list when
  BUFKIT was removed -- both now read the asOf-pinned GRIBStream archive, so they are
  leakage-safe by construction, like the other get_model_* tools. Closing the remaining six is
  what archive.md Phase 3 (serve-tools-from-the-archive) is for.
  **PARTLY CLOSED 2026-07-28 for Meteosat only.** `imagery.fetch_satellite` and
  `fetch_meteosat_point` now take `at=` and send an EXPLICIT `&time=`, so European get_imagery
  and get_loop can be pinned to an instant instead of drifting with wall clock. The other
  providers cannot: GOES STAR serves an unstamped "latest" URL and SLIDER holds only ~17 h.
  This ALSO fixed a live defect, not just a drift risk -- omitting TIME never returned the
  latest scan, it returned a server default of unknown vintage (a 00:07Z fetch came back as
  broad daylight over a dark Europe). NOTE the requested instant is not exactly the served one:
  EUMETSAT snaps to the nearest held scan, bounded by one 10-minute step, which is why
  `docs/artifact_store.md` keeps `requested_utc` and `served_utc` as separate columns.
- **Qwen agentic rumination (mitigate).** On the full-agent TAF task Qwen burned ~35.5k completion tokens and
  stalled at structural AFMAN findings without converging (issued 24h not 30h, no TX/TN). Model-specific
  (Gemma/MiniMax converge). The step-budget loop guard (below) now states the budget + nudges at turn N-2; the
  Tier-2 emit_taf guide removes the TX/TN schema thrash. Still open: the WORKSHEET decision (see Open work) to
  direct convergence; and/or lower per-turn max_tokens with more steps. Re-test at the next live run.
- **Kimi no-emit gather-loop (loop guard BUILT 2026-07-08; live re-test pending).** Kimi called 40+ read tools
  and never emitted a TAF. `scripts/test_taf_agent.py` now has the three-layer backstop: the budget is stated in
  SYSTEM, a one-time user nudge fires at turn N-2 if no emit yet, and per-tool call caps (`TOOL_CAPS`, default 8)
  make a call past the cap return feedback instead of executing (kills the get_map x22 spam). Convergence is now
  scored per model (unprompted/nudged/never). Logic verified by a stub sim; confirm on a live Kimi re-run.
- **emit quality: sky/vis persistence + value slips (quality frontier).** The clean TAFs DO encode the diurnal
  wind cycle + per-group QNH trend; what they persist is sky/vis (SKC 9999), correct for a dry ridge. The real
  gaps (full write-up in `docs/build_log.md`): a TX/TN value error (Qwen read the dewpoint as the min temp) and a repeated-
  unit-conversion inconsistency (Gemma MSLP->inHg). Addressed by the worksheet, not a code fix.

