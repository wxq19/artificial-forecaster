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
  `https://api.together.xyz/v1`. Current dev model: `Qwen/Qwen3.5-9B` (vision, supports
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

## Likely next steps
1. **v2 — METAR store + first tool (NEXT UP).** Build `src/forecaster/store.py` as the
   ONLY file that touches DuckDB (same seam idea as `llm.py` for the client). Chosen over
   Postgres deliberately: single-tenant, reproducible (a `.duckdb` file you can ship/hash),
   embedded (works on the air-gapped SuperCloud node, zero ops), analytics-first, and it
   queries Parquet/Arrow in place — fits the "arrays-as-files, records-in-DB" rule.
   - `MetarObs` rows → an `obs` table; attach the real month/year HERE (a METAR alone
     only carries day + time — that's why `MetarObs` stores `day`/`time`, not a datetime).
   - Add numeric visibility parsing (statute miles, per DAF Table 8.1) so tools can do math.
   - Then the first agent TOOL: a trend query (ceiling/vis/wind over the last N hours) the
     model calls in the loop. Keep all SQL inside `store.py`; tools never see markdown.
   - Tables to plan for: `obs`, `climo`, `runs`, `scores`, `assets` (file refs).
2. Build the agent loop + first GRIB tool (skew-T or 200mb winds) — Weeks 1-5.
3. Build the AF metric harness (TAFVER / OPVER / WARNVER) — Weeks 6-8.
4. Stand up SuperCloud: Podman images, pre-stage weights, vLLM serve job.

## Open questions to confirm with MIT SuperCloud (supercloud@mit.edu)
- V100 variant (16 vs 32 GB) on assigned nodes.
- Max GPU job wall-time (persistent-server vs batch eval pattern).
- Recommended vLLM / container workflow, if any.

## Important caveats
- Model names and provider prices shift week to week — verify model strings against
  provider docs at build time rather than trusting any hardcoded list.
- If real (non-open-source) AF weather data is ever used, hosting may need an
  authorized DoD environment, not commercial cloud. Flag this if it comes up.
