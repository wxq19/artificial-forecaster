# Code review findings — 2026-07-06

Full-codebase review of `src/forecaster/` (all modules read; config, .gitignore,
and secrets hygiene checked — hygiene is CLEAN, nothing tracked, no keys in code).
Every item in Tier 1 was CONFIRMED BY EXECUTION, not just read; the repro code is
in the appendix at the bottom so any session can re-verify. Line numbers are as of
commit 68c76ad.

Suggested order: 2 + 4 (validate gaps + guard roundtrip), 3 (required wind), and
5 (run_tool hardening) first — they are all "the agent loop dies or lies instead
of self-correcting," the failure mode the emit/re-emit design exists to prevent.
Then 1 (month-end), then 7 (config anchoring) before any HPC work.

PROGRESS 2026-07-06: ALL findings resolved (1-9 + all of Tier 3, incl. d.5 via Option 1).
3 fixed via the corrected approach, not the doc's original "make wind required" idea; 5
also closed the `n`-coerce half of 9; 8 done as policy B; d.5 parameterized. Nothing open.
FOLLOW-UP (new, not in original review): amend() carries max_temp/min_temp forward
unchanged, but TX/TN times are absolute and can fall outside the clipped validity after an
amendment (AFMAN: TX/TN cover the first 24h). Separate from d.5; track if it matters.
  RESOLVED 2026-07-07: per the lead meteorologists, the AF never issues a TAF without TX/TN,
  and on amendment they cover the REMAINDER of the ORIGINAL 24h period. validate() now
  hard-requires both (gated on a new TafProduct.military flag so civil/NWS TAFs are exempt)
  and checks the uniform window [valid_from, valid_to-6]; amend() carries TX/TN forward with
  optional max_temp/min_temp overrides. Self-test 8/8. See CLAUDE.md "Resolved".

---

## Tier 1 — confirmed bugs (each reproduced; see appendix)

### 1. [x] `TafProduct.issue()` crashes when validity crosses a month end
FIXED 2026-07-06: `_end_plus_hours` + `issue()` + `validate()` take an optional
days_in_month (28-31); the end day WRAPS (32 -> 1) instead of overflowing, and the
span check sizes a month-crossing validity (or SKIPS it when the month is unknown,
instead of the old bogus negative). Within-month behavior unchanged. Verified.

- Where: `tafgen.py:49` (`_end_plus_hours`), `tafgen.py:126` (`valid_to_day` le=31)
- What: `_end_plus_hours` computes day 32 for a TAF issued late on the last day
  of a 31-day month; the pydantic `le=31` constraint then rejects it, so a
  routine TAF is UNBUILDABLE on those days (~2 days of every month).
  Confirmed: `TafProduct.issue(... valid_from_day=31, valid_from_hour=10 ...)`
  -> `ValidationError: valid_to_day ... input_value=32`.
- Related: `validate()`'s 30h span check works in absolute day*24+hour, so a
  month-crossing TAF (e.g. valid 3106/0112) false-flags with a negative span.
  That caveat IS documented in the docstring; the `issue()` crash is not.
- Fix idea: the caller of `issue()` always knows the real date, so add an
  optional `days_in_month` parameter (or accept a `date`) and have
  `_end_plus_hours` wrap the day number. Consider the same for `validate()`'s
  span math so a month-crossing TAF stops false-flagging.

### 2. [x] Change group with no timing passes `validate()` clean, then roundtrip crashes the loop
FIXED 2026-07-06: validate() flags any FM/BECMG/TEMPO missing from_day/from_hour;
roundtrip() wrapped in try/except in _emit_taf. Verified.

- Where: `tafgen.py:396` (validate checks BECMG/TEMPO *end* only, never any
  group's *start*), `tafgen.py:249` (`_hh` comment falsely claims "validate()
  reports the real finding"), `tools.py:250` (`roundtrip()` is the ONLY step in
  `_emit_taf` not wrapped in try/except).
- What: an FM group with `from_day`/`from_hour` omitted (the documented model
  habit of omitting optional fields) gives: validate() -> CLEAN, render ->
  `FM????00`, roundtrip() -> uncaught `ValueError: invalid literal for int()
  with base 10: '??'` — which escapes `run_tool` and kills the agent loop
  instead of becoming re-emit feedback. Confirmed end-to-end.
- Fix idea (two independent pieces, do both):
  - `validate()`: add a finding when any change group (FM/BECMG/TEMPO) has
    `from_day` or `from_hour` None.
  - `_emit_taf`: wrap `tafgen.roundtrip(product)` in try/except like render,
    reporting the exception as correctable feedback.

### 3. [x] Omitted wind silently becomes a calm forecast (benchmark-corrupting)
FIXED 2026-07-06: NOT via "make wind_speed required" (that idea was wrong -- AFMAN
1.3.3.1/.2 + NWS: BECMG/TEMPO include only CHANGED elements, so wind is legitimately
absent there). Instead: calm is now EXPLICIT (dir 0 + speed 0); omitted wind (dir None,
speed 0) renders NO wind element in change groups and is flagged by validate() on the
prevailing + FM periods only (mandatory per Fig 1.1 / 1.3.3.3). Render bug fixed too:
_wind returns None -> _body omits it, no bogus 00000KT. Verified.

- Where: `tafgen.py:78` (`wind_speed: int = Field(0, ge=0)`), `tafgen.py:192`
  (calm branch renders `00000KT`).
- What: a model that skips the wind fields entirely gets `00000KT` rendered
  with ZERO findings — a plausible-looking, validated, wrong TAF. Silent, so it
  corrupts scoring without a trace. Confirmed: group with only vis+QNH renders
  `00000KT`.
- Fix idea: make `wind_speed` required (drop the default), or add a
  `validate()` finding when the prevailing or an FM group carries no wind
  (FM is self-contained per 1.3.3.3, so wind is mandatory there anyway).

### 4. [x] `wind_dir=None` with `wind_speed>0` raises TypeError with misleading feedback
FIXED 2026-07-06: _wind renders a visible '///' placeholder (no crash); validate() adds
the accurate "wind speed given without a direction" finding. VRB (a string) is unaffected.
Verified.

- Where: `tafgen.py:195` (`int(g.wind_dir)` on None), `tools.py:241-244` (the
  render guard catches it but blames "missing its day/hour fields").
- What: confirmed `TypeError: int() argument must be ... not 'NoneType'`. The
  model is then told to fix day/hour fields — the wrong thing — so it thrashes.
- Fix idea: `validate()` check "wind_dir required when wind_speed > 0" (turns
  this into accurate feedback before render is even attempted).

### 5. [x] `run_tool` crashes on malformed read-tool args instead of returning feedback
FIXED 2026-07-06: run_tool guards missing station, coerces+clamps hours to [1,48] and n
via new _int_arg(), guards empty rows before charting, and wraps the whole read-tool body
in try/except -> ToolResult("error: ..."). All three confirmed crashes now return feedback.
(Also fixes the un-coerced `n` half of #9.) Verified.

- Where: `tools.py:269` (`args["station"]` -> KeyError), `tools.py:286`
  (`int(args["hours"])` -> ValueError on `"24h"`; `min(..., 48)` caps the top
  but not the bottom), `charts.py:148` (`times[0]` IndexError on empty rows).
- What: three confirmed crashes: missing `station` -> KeyError; `hours: "24h"`
  -> ValueError; negative `hours` -> start>end -> empty window -> IndexError in
  `meteogram`. All three kill the agent loop. This is asymmetric with
  `_emit_taf`, which carefully converts every failure into correctable text.
- Fix idea: try/except around the read-tool body in `run_tool` returning
  `ToolResult("error: ...")`; clamp `hours` to [1, 48]; guard empty rows in
  the `get_trend` branch (or in `meteogram` itself).

### 6. [x] A `groups` entry with `change=None` passes validation and renders a rogue line
FIXED 2026-07-06: one-line `validate()` check -- every entry in `groups` must set
change (FM/BECMG/TEMPO); only the prevailing period has none. Verified.

- Where: `tafgen.py:68` (Literal allows None because the prevailing needs it;
  nothing forbids None inside `groups`), `validate()` has no check.
- What: confirmed — `groups=[TafProductGroup(change=None, ...)]` validates
  CLEAN and renders a bare `18008KT 9999 SKC QNH2990INS` continuation line
  (no FM/BECMG/TEMPO head), which no parser will attribute correctly.
- Fix idea: one `validate()` line — every group in `groups` must have
  `change` set.

---

## Tier 2 — design-level issues (not executed, but solid)

### 7. [x] `config.py` is cwd-dependent (the "infra bug masquerading as model error" class)
FIXED 2026-07-06: both .env and db_path anchored to the repo root via
Path(__file__).resolve().parents[2]; env vars still override. Verified identical from
repo root and from /tmp. (Deeper HPC/scratch-path handling deferred to HPC setup.)

- Where: `config.py:4` (`env_file=".env"`), `config.py:8`
  (`db_path="data/forecaster.duckdb"`).
- What: both paths are relative, so running from any directory other than the
  repo root silently falls back to the Ollama defaults and/or creates a fresh
  EMPTY DuckDB. Doubly relevant for Slurm on SuperCloud, where cwd is whatever
  the submission script says.
- Fix idea: anchor both to the repo root via `Path(__file__).resolve()`
  parents. Keeps the seam intact (still the only config source).

### 8. [x] `ON CONFLICT DO NOTHING` means corrected obs never land
FIXED 2026-07-06 (policy B, correction-wins): new `corrected` BOOLEAN column (a COR in
the raw), migrated onto existing DBs via ALTER ADD COLUMN IF NOT EXISTS. ON CONFLICT now
DO UPDATE ... WHERE excluded.corrected AND NOT COALESCE(obs.corrected, FALSE): keep-first
stays the default, a COR overwrites a stored non-COR regardless of arrival order, and a
plain report can never downgrade a stored correction. Verified: COR wins, re-served
original preserved, idempotent re-ingest, METAR/SPECI-same-minute keeps-first.
NOTE: two corrections for the same minute keep the first COR (rare; relax at TAFVER if needed).

- Where: `store.py:57`.
- What: a COR METAR (or IEM re-serving a corrected ob) shares
  `(station, obs_time)` with the original, so the FIRST-inserted value wins
  forever — verification truth data can freeze a pre-correction value.
- Decide (a policy call, not a one-liner): keep-first (current), upsert, or at
  least count/surface conflicts in the insert summary so they are visible.
  Related edge: a METAR and SPECI in the same minute also collide.

### 9. [x] Misleading no-data error in `_resolve_window` + un-coerced `n`
FIXED 2026-07-06: `_resolve_window` returns (start, end, reason); the reason
distinguishes "no observations stored for STATION" from "give either hours or start+end".
The `n`-coerce half was already closed with #5 (_int_arg). Verified both messages.

- Where: `tools.py:205-210`, `tools.py:283`.
- What: a station with zero obs returns `(None, None)` and the model is told
  "give either hours or both start and end" — its args were fine; there is no
  data. It will thrash re-formatting arguments. Also `get_latest_obs` passes
  `args.get("n", 1)` straight through — models quote numbers (`"n": "3"`), so
  coerce with `int()`.
- Fix idea: distinguish "no observations for STATION" from "bad arguments" in
  the receipt text; `int()` the `n` arg.

---

## Tier 3 — minor / cosmetic
(All FIXED 2026-07-06 except the amend()-remarks item, which is d.5 -- a decision left
for discussion. Self-test 7/7, ruff clean, behaviors verified.)

- [x] `tools.py:185` — `_fmt` prints `None/None` for missing temp/dewpoint
  (AUTO stations do this); `metar.render` uses an em dash for the same case.
  FIXED: per-value em dash -> `—/—`.
- [x] `metar.py:128` — pressure precedence: `A####` is searched across the
  WHOLE raw including RMK, before `Q####`. Many intl stations (RJ..., etc.)
  carry `A####` in remarks, so hPa ends up derived from inHg instead of exact
  from the Q token — quietly contradicting the "exact from raw" intent. Search
  the pre-RMK body, or prefer Q for hPa when both exist.
  FIXED: search only the pre-`RMK` body. Verified Q1008(body)+A2977(rmk) -> hpa 1008.0.
- [x] `metar.py:252` — `render()` sorts by `(day, time)`, which mis-orders a
  batch spanning a month boundary (day 31 sorts after day 1). The store path is
  immune (real `obs_time`); this affects only the file/paste path.
  FIXED (documented): MetarObs has no month, so nothing to sort on; docstring now
  states the within-month assumption and points month-spanning reads at the store path.
- [~] `tafgen.py` small gaps (4 of 5 FIXED; amend()-remarks = d.5, open):
  - [x] wind dir 0 with speed>0 renders `000` (convention is 360);
    FIXED: `_wind` renders a real north wind as 360.
  - [x] icing/turbulence thickness silently floored by `// 1000` (1500 -> 1000ft)
    with no multiple-of-1000 finding (Table 1.6 values are whole thousands);
    FIXED: validate() flags a non-whole-thousands thickness.
  - [x] wind-shear height not checked <=2000ft AGL; WS direction not checked
    <=360 / %10;
    FIXED: validate() checks WS height (0<h<=2000) + direction (0-360, /10).
  - [x] `valid_from_hour` permits 24 (only an END should use 24);
    FIXED: field tightened to le=23 (a START is 00-23; only an END uses 24).
  - [x] (d.5) `amend()` drops TAF-level remarks — possibly intentional (duty status
    changes), but if a `LAST NO AMDS` should survive an amendment, it currently does not.
    FIXED 2026-07-06 (Option 1, parameterize): `amend()` gains `remarks: list[str] | None`;
    default still drops (a duty remark can be stale/self-contradicting on an AMD), but the
    caller re-supplies the ones that should survive -- tafgen does not guess duty policy.
    Verified: default -> no remarks; passing LIMITED METWATCH -> it survives.

---

## Appendix — repro probe (Tier 1 evidence)

Run with `uv run python <file>`. Output observed 2026-07-06 at commit 68c76ad:
case 1 ValidationError (day 32); case 2 validate CLEAN then roundtrip
ValueError '??'; case 3 TypeError int(None); case 4 renders 00000KT; case 5
validate CLEAN + rogue line; case 6 KeyError / ValueError / IndexError.

```python
"""Probe suspected issues without touching the repo or DB."""
from forecaster import tafgen
from forecaster.tafgen import TafProduct, TafProductGroup

def case(label, fn):
    try:
        r = fn()
        print(f"{label}: OK -> {r}")
    except Exception as e:
        print(f"{label}: {type(e).__name__}: {e}")

prev = TafProductGroup(wind_dir=240, wind_speed=10, vis_m=9999, qnh_inhg=29.92)

# 1. issue() on the last day of a 31-day month (validity crosses month end)
case("issue day31", lambda: TafProduct.issue(
    station="KBLV", issue_day=31, issue_hour=10, issue_minute=0,
    valid_from_day=31, valid_from_hour=10, prevailing=prev).station)

# 2. FM group with no from_day/from_hour: does validate() flag it?
p = TafProduct(
    station="KBLV", issue_day=10, issue_hour=10, issue_minute=0,
    valid_from_day=10, valid_from_hour=10, valid_to_day=11, valid_to_hour=16,
    prevailing=prev,
    groups=[TafProductGroup(change="FM", wind_dir=180, wind_speed=8,
                            vis_m=9999, qnh_inhg=29.90)],
)
print("2. validate findings:", tafgen.validate(p) or "CLEAN")
case("2. render", lambda: tafgen.render_taf(p))
case("2. roundtrip", lambda: tafgen.roundtrip(p))

# 3. wind_dir omitted but speed > 0 (model omits optional field)
g = TafProductGroup(wind_speed=12, vis_m=9999, qnh_inhg=29.92)
case("3. render wind_dir=None speed=12", lambda: tafgen._wind(g))

# 4. omitted wind entirely -> silent calm?
g2 = TafProductGroup(vis_m=9999, qnh_inhg=29.92)
print("4. omitted wind renders as:", tafgen._wind(g2))

# 5. groups may contain change=None (a second 'prevailing')
p5 = TafProduct(
    station="KBLV", issue_day=10, issue_hour=10, issue_minute=0,
    valid_from_day=10, valid_from_hour=10, valid_to_day=11, valid_to_hour=16,
    prevailing=prev, groups=[TafProductGroup(wind_dir=180, wind_speed=8,
                                             vis_m=9999, qnh_inhg=29.90)],
)
print("5. change=None group findings:", [f for f in tafgen.validate(p5)] or "CLEAN")
case("5. render", lambda: tafgen.render_taf(p5))

# 6. run_tool robustness: missing station / bad hours
from forecaster import tools
case("6a. run_tool no station", lambda: tools.run_tool("query_obs", {}).text[:60])
case("6b. run_tool hours='24h'", lambda: tools.run_tool(
    "query_obs", {"station": "KORD", "hours": "24h"}).text[:60])
case("6c. run_tool negative hours (meteogram)", lambda: tools.run_tool(
    "get_trend", {"station": "KORD", "hours": -4}).text[:60])
```
