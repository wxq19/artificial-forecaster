# Review findings — 2026-07-08 (ALL OPEN)

Review of the new fetch/render modules (`soundings.py`, `wxmaps.py`,
`fcstsounding.py`, `charts.skewt`, the point-forecast tool) plus the first
full-agent TAF run (`logs/taf_agent_KLSV_20260707-174247.md`, 4 models, all 8
tools). Line numbers are as of commit 2ca76c4. Evidence for the Tier 2/3 items
is IN THAT LOG (cited per item) — the models already ran the repro for us.

Suggested order: Tier 2 first (6, 7, 8 are mostly prompt/description text —
cheap, and they remove the failure fuel for the next multi-model run), then 9
(harness fairness, needed before re-judging Kimi), then Tier 1 (1-5, small code
fixes). Tick each checkbox and append a short FIXED note (date + what was done
+ how verified) as items land, matching review-findings-2026-07-06.md.

House rules for whoever fixes these: advisor-built repo — keep changes minimal
and explained; no emojis anywhere; matplotlib only in charts.py; SQL only in
store.py; clients (soundings/wxmaps/fcstsounding) stay network-only; README.md
is the user's personal tracker, do not touch it.

---

## Tier 1 — code fixes (feedback quality / data hygiene)

### 1. [x] `soundings.synoptic_time` has no post lag — 404s near 00Z/12Z with misleading feedback
FIXED 2026-07-08. Added `_POST_LAG_H = 2.0` and back it off before snapping to 00/12Z
(user chose 2h to be safe), same pattern as wxmaps/fcstsounding. Reworded the
`_get_sounding` error: a fetch failure now reads "no launch at this synoptic time, or
wrong id for this provider (SPC: 3-letter or WMO; Wyoming: WMO)", not "check the site
id". Verified against monkeypatched clocks: 00:30Z/01:45Z -> prior 12Z; 12:30Z/13:45Z
-> 00Z; the 12Z image only goes current at 14:00Z (2h post-launch). Did NOT add
404-retry-previous-cycle (the lag covers the reported failure; kept minimal).

- Where: `soundings.py:40` (`synoptic_time`), `tools.py:413` (the except path
  in `_get_sounding`).
- What: radiosondes post ~60-90 min after launch, but `synoptic_time` snaps to
  the current 00Z/12Z the moment the hour ticks over. A `get_sounding` call
  between ~00:00Z and ~01:15Z (or 12:00-13:15Z) builds a URL for a sounding
  that is not on the server yet -> HTTP 404 -> the tool tells the model to
  "check the site id for this provider". The site id is fine; the model gets
  steered at the wrong knob (the "misleading feedback" failure class from the
  07-06 review, item 4).
- Fix idea: back off a post lag (~90 min) before snapping, the same pattern as
  `wxmaps.latest_gfs_run` (`_GFS_POST_LAG_H`) and `fcstsounding.latest_run`.
  Alternatively (or additionally) catch the 404 in `_get_sounding` and retry
  the PREVIOUS synoptic time once, saying so in the receipt. Also reword the
  error: a 404 is "not posted yet / not launched at this site", not
  necessarily a bad id.
- Verify: monkeypatch "now" to 00:30Z and confirm the URL targets the prior
  12Z; live-fetch one site each from SPC + Wyoming.

### 2. [x] `_get_map` default `fhr=0` is invalid for `gfs_mslp_precip` (hit live by Kimi)
FIXED 2026-07-08. `_get_map` now reads `f0 = spec.params.get("f0", 0)` and both
DEFAULTS and FLOORS fhr at f0 (`_int_arg(args.get("fhr"), f0, lo=f0, hi=...)`), written
generically so any future averaged-field chart inherits it. Added a clause to the
GET_MAP description: "averaged-field charts (gfs_mslp_precip) start at f006, not f000."
Verified: gfs_mslp_precip with fhr omitted / 0 / 3 / 9 all return f006 (no error), 12 ->
f012; gfs_500mb (f0=0) still defaults to f000.

- Where: `tools.py:437` (`_int_arg(args.get("fhr"), 0, lo=0, ...)`),
  `wxmaps.py:90` (the chart's `f0: 6` — its first frame is f006 because the
  precip field is 6h-averaged and has no f000).
- What: a model that omits `fhr` (or passes 0) for `gfs_mslp_precip` gets
  `ValueError: fhr must be a multiple of 6 in 6..384 ... got 0`. Confirmed in
  the KLSV log — Kimi step 6, first call. It burned a correction turn on a
  default WE chose.
- Fix idea: in `_get_map`, default and floor `fhr` at
  `spec.params.get("f0", 0)` instead of 0 (i.e. `_int_arg(args.get("fhr"),
  f0, lo=f0, hi=...)`), and add one clause to the GET_MAP description:
  "gfs_mslp_precip starts at f006". Write it generically off `f0` so any
  future averaged-field chart inherits the behavior.
- Verify: `run_tool("get_map", {"chart": "gfs_mslp_precip"})` (fhr omitted)
  and with `fhr: 0` and `fhr: 3` all return an f006 image, no error; other
  gfs_* charts still default to f000.

### 3. [x] `fcstsounding` fill sentinels: surface rows unmasked + the tracked DRCT/SKNT gap
FIXED 2026-07-08 (both halves in one pass). `parse` profile mask now also drops a level
whose DRCT is outside [0,360] or SKNT < 0 (the CLAUDE.md "Known problems" item), so a
good-temp level with a -9999 wind no longer reaches MetPy. `_parse_surface` maps any
surface field <= -9000 to None, and `tools._fmt_point` prints "--" for a None value
(wind dashes unless BOTH u/v present; the cloud triple dashes unless all three present).
Verified: live KLAS GFS f012 renders with DRCT 129..277, SKNT min 4.0, and ZERO
wind/radians warnings (the "input over 12.566 radians" warning is gone); a synthetic
surface row with fill renders as dashes with columns still aligned. Removes the
CLAUDE.md "Known problems" MetPy-fill entry.

- Where: `fcstsounding.py:201` (`_parse_surface` — no fill handling at all),
  `fcstsounding.py:155` (profile mask covers T/Td only; the DRCT/SKNT half is
  already tracked in CLAUDE.md "Known problems" — fix both in one pass).
- What: BUFKIT uses -9999 as its missing sentinel. `_parse_surface` passes
  values straight through, so a missing surface field would print `-9999` (or
  a garbage wind via `_uv_to_dirspd`) in the point-forecast table the model
  reads as truth. Same bug class as the profile-level DRCT issue that produces
  the MetPy "input over 12.566 radians" warning (surfaced on KLAS).
- Fix idea: in `_parse_surface`, map any field <= -9000 to None and have
  `tools._fmt_point` print a dash for None (wind needs BOTH u/v present; the
  cloud triple needs all three). In `parse`, extend the fill mask: also drop
  levels with DRCT outside [0, 360] or SKNT < 0 (the CLAUDE.md item,
  verbatim).
- Verify: KLAS GFS skew-T renders with ZERO MetPy warnings (reproduce the
  warning first on the unfixed code); a synthetic surface row with -9999
  renders as dashes, not numbers.

### 4. [x] Batched tool images are unlabeled — the model can't tie image N to call N
FIXED 2026-07-08. In `test_taf_agent.py` the batched-image follow-up now carries the
images as `(receipt-first-line, png)` pairs and interleaves a `[image for: <receipt>]`
text item before each image in the content array, so a multi-image turn (e.g. several
forecast soundings) keeps each image tied to its call. Ruff clean + driver parses;
live confirmation that the endpoint accepts the interleaved shape rides the next agent
run (it is the standard OpenAI text/image content array). `tools.tool_messages` (the
single-call path) left as-is; label it too when this migrates to agent.py.

- Where: `scripts/test_taf_agent.py:141-146` (the batched follow-up user
  message). `tools.tool_messages` (`tools.py:599`) has the same shape but is
  single-call, so ordering is unambiguous there; still worth labeling for
  consistency when this migrates to `agent.py`.
- What: when one turn makes several image-returning calls, the images arrive
  as ONE user message: a single "Images from the tool calls above:" header
  then N anonymous images. Kimi's step 2 fetched SEVEN forecast soundings
  (f000..f030) in one turn — ordering is the only cue tying image k to
  forecast hour k. A model that loses confidence in that mapping re-fetches
  (Kimi's get_map x22 / get_fcst_sounding x13 spam is consistent with this)
  or, worse, silently reasons over the wrong valid time — the image-vs-text
  desync failure class again.
- Fix idea: interleave content items — before each image, a text item carrying
  that tool's receipt first line (already available as
  `res.text.splitlines()[0]`), e.g. `[image for: GFS forecast skew-T for
  KLAS, f012 valid ...]`. The OpenAI content array supports alternating
  text/image_url items.
- Verify: inspect the built messages array for a 2+ image turn — alternating
  text/image items, labels matching receipts; one live tool-loop run to
  confirm the endpoint accepts the shape.

### 5. [x] Minor: point-forecast `hours` slices rows, not hours; `_get_map` double run pick
FIXED 2026-07-08. (a) `_fmt_point` now slices by valid TIME -- keeps rows while
`row["valid"] <= rows[0]["valid"] + timedelta(hours=n)` -- so hours=N covers N wall-clock
hours even once the BUFKIT series goes 3-hourly. Verified on synthetic mixed-cadence
rows: hours=6 stops at the +6h valid time (5 rows), not row 6. (b) `_get_map` resolves
`latest_gfs_run()` ONCE and passes that `run` to both `map_url` and `fetch_map`, and the
receipt now cites the run -- so provenance and image can't straddle a cycle-post
boundary.

- Where: `tools.py:484` (`rows = pf.rows[:n]` in `_fmt_point`),
  `tools.py:440-441` (`map_url` then `fetch_map`, each calling
  `latest_gfs_run()` internally).
- What: (a) `hours=48` takes the first 48 ROWS. Equivalent only while the
  BUFKIT surface series is hourly; GFS goes 3-hourly at longer ranges, so a
  big `hours` silently covers more wall-clock time than the label claims.
  (b) the provenance URL and the fetched image resolve `latest_gfs_run()`
  independently; across a cycle-post boundary the receipt could cite a
  different run than the image shown.
- Fix idea: (a) slice by valid time — keep rows while
  `row["valid"] <= rows[0]["valid"] + timedelta(hours=n)`. (b) compute `run`
  once in `_get_map` and pass it to both calls; cite the run in the receipt.
- Verify: (a) synthetic rows with mixed cadence (hourly then 3-hourly) —
  hours=48 stops at the 48h valid time, not row 48; live KLAS table unchanged
  (hourly file). (b) code inspection is enough.

---

## Tier 2 — model-facing contract (what we hand the model)

Highest-leverage items: all four models' KLSV failure modes trace to contract
VISIBILITY, not meteorology.

### 6. [x] emit_taf reference guide — the nested schema is invisible (Qwen's 30k-token death spiral)
FIXED 2026-07-08. `tafgen.emit_taf_guide()` returns a model-facing reference: a
worked VALID TafProduct (rendered from a live `_example_product()` as JSON + the TAF
text it produces) plus a flattened field guide that names the fields the schema hides
behind `anyOf[$ref,null]` (TafTemp temp_c/day/hour, cloud-layer shape). Delivered via
the drivers' SYSTEM prompt (`test_taf_agent.py`, `test_emit_taf.py` append it). Drift
guard: `scripts/test_tafgen.py` now validates + round-trips the same `_example_product()`
(case "emit_taf guide example", 9/9 PASS), so the guide cannot diverge from the
validator. Verified the guide text renders. Live multi-model KLSV re-run (Qwen step
count / convergence) still to run — costs tokens; deferred to the next agent run.

- Where: `tools.py:259` (EMIT_TAF description), `tafgen.py` (TafProduct /
  TafTemp / CloudLayer shapes), CLAUDE.md next-steps item 11 (this formalizes
  it).
- What: optional nested models render as `anyOf[$ref, null]` in
  `model_json_schema()`, so the model cannot see `TafTemp`'s fields. In the
  KLSV log, Qwen step 6 burned its entire 30k-token budget cycling THREE
  guesses about `day`/`hour` semantics ("relative to issue time?
  day-of-month? must be 1-31?") and never emitted again — it stalled at 3
  structural findings it knew how to fix meteorologically. Gemma
  independently paused on the same shape in its step 6.
- Fix idea: a worked, VALID TafProduct JSON example plus a flattened field
  guide (one line per field: name, type, meaning, example — explicitly "day =
  UTC day-of-month 1-31; hour = UTC whole hour 0-23") passed in the system
  prompt or tool description. NOT a schema replacement — pydantic stays the
  validator. Keep it in ONE place (e.g. a `tafgen.emit_taf_guide()` the
  driver imports) and have a self-test render/validate/roundtrip the example
  so the guide physically cannot drift from the validator.
- Verify: self-test proves the worked example builds clean, renders, and
  round-trips; re-run the Qwen KLSV scenario and compare step count /
  completion tokens / whether it converges past the TX/TN findings.

### 7. [x] Schema/check contract mismatch: TX/TN optional in the schema, required by the AFMAN check
FIXED 2026-07-08. Pydantic fields stay optional (civil TAFs omit them). Aligned the
VISIBLE contract: the emit_taf description now states TX/TN are required on every AF
TAF with the exact TafTemp shape (`{"temp_c","day","hour"}`), and the #6 guide spells
day/hour semantics. `_emit_taf` also appends a `note:` naming the three required
integers when a `max_temp`/`min_temp` union error fires (verified: malformed TafTemp ->
feedback names temp_c/day/hour). The AFMAN finding and the description now agree.

- Where: `tafgen.py` (`max_temp`/`min_temp` default None on TafProduct;
  `validate()` hard-requires both when `military=True` — the 2026-07-07
  resolution), `tools.py:259` (the description says nothing about TX/TN).
- What: the model reads the JSON schema, sees `max_temp` is optional, omits
  it, then gets an AFMAN finding demanding it — a contradiction between the
  two contracts we handed it. Qwen NOTICED the contradiction in-log ("the
  schema says optional but the check requires them") and it visibly fed the
  rumination loop: schema says one thing, validator says another, so it
  distrusted everything.
- Fix idea: keep the pydantic fields optional (civil TAFs legitimately omit
  them; the class must still build without them). Align the VISIBLE contract
  instead: state in the emit_taf description (and the #6 guide) that TX/TN
  are required on every AF TAF, with the exact shape
  (`"max_temp": {"temp_c": 43, "day": 7, "hour": 23}`). Optionally
  post-process the schema dict in EMIT_TAF to inject that into the field
  descriptions — it is already a plain dict at tool-build time. Also worth
  improving: when TafTemp construction fails inside the None|TafTemp union,
  pydantic's error is a terse two-branch pair; `_emit_taf` could rewrite that
  specific case to spell out the three required integer fields.
- Verify: emit with max_temp omitted -> the AFMAN finding text and the
  description now AGREE; emit with a malformed TafTemp -> feedback names
  temp_c/day/hour explicitly.

### 8. [x] Clear-sky encoding cost 3 of 4 models a turn each (CLR/SKC vs empty clouds list)
FIXED 2026-07-08. Both halves done: (a) the emit_taf description + #6 guide say clear
skies = an empty clouds list (renders SKC); SKC/CLR are not valid covers. (b) Two
prescriptive rejections that NAME the fix -- `TafProductGroup._check_covers` catches an
SKC/CLR/NSC/NCD cover with the "pass an empty clouds list" message, and `_emit_taf`
scans args for a clear-sky token and attaches the same hint even when CloudLayer's
required height_ft trips first (which was masking the cover error). Verified: SKC with
and without a height both feed back the fix; `clouds: []` renders SKC and passes clean;
FEW/SCT/BKN/OVC unchanged.

- Where: `tafgen.py` (CloudLayer `cover` Literal FEW/SCT/BKN/OVC; `render_taf`
  emits `SKC` for an empty list), `tools.py:259` (description silent on it).
- What: in the KLSV log — Qwen step 4 (invented `FEW100` after `CLR` was
  rejected, then reasoned in circles about whether omission was legal),
  MiniMax step 3 (tried `SKC`, got rejected, self-corrected to `[]`), Gemma
  (one of its 4 emit cycles). The renderer HAPPILY prints SKC; only the input
  side refuses the token. That asymmetry is invisible until you trip on it.
- Fix idea: two independent halves, do both: (a) one sentence in the emit_taf
  description — "for clear skies pass an empty clouds list (it renders as
  SKC); SKC/CLR are not valid layer values"; (b) improve the rejection: a
  validator on `cover` (or a rewrite in `_emit_taf`) so an SKC/CLR/NSC layer
  fails with a message that NAMES the fix ("pass an empty clouds list")
  rather than just listing the legal values. Prefer prescriptive rejection
  over silent coercion — a group with SKC+FEW250 should not silently become
  two layers.
- Verify: emit with `clouds: [{"cover": "SKC"}]` -> feedback says "use an
  empty list"; emit with `clouds: []` -> renders SKC and passes; FEW/SCT/BKN/
  OVC unchanged.

---

## Tier 3 — harness / benchmark fairness (future runs)

### 9. [x] Models are never told the step budget (Kimi judged under a rule it couldn't see)
FIXED 2026-07-08 (all three layers, per decision). In `test_taf_agent.py`: (a) SYSTEM now
states "at most N turns; call emit_taf by turn N-2"; (b) a one-time user-role nudge fires
at step N-2 if no emit attempt yet ("you have enough data; call emit_taf now"); (c)
`TOOL_CAPS` (get_map/get_sounding/get_fcst_sounding/get_point_forecast = 8) makes a call
past the cap return feedback instead of executing. Convergence is scored per model
(unprompted/nudged/never) in the summary + transcript. Logic verified by a stub sim: a
gather-forever model -> nudge fires exactly once at step 10, get_map capped on calls 9-12,
convergence=never; emit@3 -> unprompted; emit@11-after-nudge -> nudged. Live Kimi re-run
(which layer gets it to emit) rides the next agent run.

- Where: `scripts/test_taf_agent.py:32` (`--max-steps 12`, harness-only),
  `SYSTEM`/`TASK` (no mention of any budget), CLAUDE.md "Known problems" (the
  Kimi loop-guard item — this extends it).
- What: `max_steps` exists only in our loop. From the model's side the
  conversation just STOPS. Kimi's step 1-2 reasoning shows a coherent plan
  (it reasoned about issue-time conventions, knew the Vegas-area RAOB site);
  it was executing a breadth-first gather with no convergence pressure and no
  idea a clock existed. "No TAF emitted" is then partly a harness artifact.
  Its 22x get_map fhr sweeps and a step-12 switch to NAM4KM soundings still
  argue weak convergence — but the benchmark cannot separate "never
  converges" from "converges when told the budget" until the budget is
  visible.
- Fix idea: three layers, cheapest first: (a) state the budget in SYSTEM
  ("you have at most N turns; emit by turn N-2"); (b) inject a user-role
  nudge at step N-2 ("you have enough data; call emit_taf now" — the
  CLAUDE.md backstop; user role, NOT system — Together 400s on
  mid-conversation system messages); (c) per-tool call caps (e.g. get_map <=
  8/run) where a capped call returns feedback ("cap reached; you have enough
  data") instead of executing. Then score convergence as its own axis:
  unprompted / nudged / never.
- Verify: a stub-client dry run that gathers forever — nudge fires exactly
  once at step N-2, caps bite and log as capped; then a live Kimi re-run to
  see which layer (if any) gets it to emit.

### 10. [x] Cost accounting: prompt tokens dominate and the summary hides it
FIXED 2026-07-08 (token-only, per decision -- no hardcoded $ rates, which CLAUDE.md warns
shift weekly). Each step dict now stores `ptok`/`ctok` (from `r.usage`), printed per step in
the transcript. The summary gained an "End ctx (ptok)" column (final turn's prompt size = the
conversation peak) alongside the p+c totals, plus a Cost note explaining the resend mechanics
(every turn re-sends the whole conversation incl. all images as fresh prompt tokens, so a wide
gather loop can out-cost a ruminator -- read both columns). Dollar estimates intentionally
deferred. Verified via ruff + parse; live token numbers populate on the next run.

- Where: `scripts/test_taf_agent.py:99-100` (only running totals kept), the
  summary table (one p+c column).
- What: every turn re-sends the whole conversation — including every image so
  far — as fresh prompt tokens. KLSV: Kimi produced 1.8k completion tokens
  but 347,922 prompt tokens (7-8x Gemma's 45,828). "Kimi barely used tokens"
  reads exactly backwards off the current table; it was likely the most
  expensive run of the four. Rumination (Qwen, 35.5k completion) and
  gather-loops (Kimi, 348k prompt) are BOTH cost failures, on different axes.
- Fix idea: record per-step prompt/completion tokens (already in `r.usage`,
  just store them on each step dict and print per step in the transcript);
  add final-step prompt size (= end context) and/or a per-model cost estimate
  at current provider rates to the summary; a one-line "cost note" under the
  table explaining the resend mechanics. Verify whether Together discounts
  cached prefixes before assuming any relief.
- Verify: per-step columns sum to the totals; image-heavy steps visibly
  inflate subsequent prompt counts.

### 11. [x] CLAUDE.md status correction: the clean KLSV TAFs were NOT flat persistence
FIXED 2026-07-08 (doc-only; claims re-verified against the log first). The persistence claim
was corrected in BOTH the Status FULL-AGENT bullet (added a CORRECTION: both clean TAFs encode
the diurnal wind shift + per-group QNH trend -- Gemma FM090000 23020G25KT QNH 2975->2952,
MiniMax BECMG to 25010KT -- and persist only sky/vis, correct for a dry ridge) and the
Known-problems entry. Real gaps cited with values checked against the log: Qwen read the
observed dewpoint (-5C) as the overnight min temp in reasoning and stalled in the emit_taf
schema spiral (never emitting TX/TN); Gemma converted the same MSLP to different inHg at
different points (1008 -> 29.77 AND 29.79; 1011 -> 29.85/29.86/29.88). NOTE: the review's
"identical TAFs" and "1008 -> 29.76 and 29.86" specifics were imprecise -- corrected to what
the log actually shows. Both gaps folded into the step-12 worksheet sub-tasks (sanity-check
TX/TN vs observed diurnal range; double-check any unit conversion once).

- Where: CLAUDE.md Status, the FULL-AGENT TAF TEST bullet ("Even the CLEAN
  TAFs are PERSISTENCE (gusty wind + SKC held flat 30h, ignoring overnight
  diurnal easing)") and the matching "Known problems" entry.
- What: the log contradicts this. Gemma's final TAF: `FM080700 VRB03KT` ->
  `FM082000 21015G25KT` -> `FM090000 23020G25KT`, QNH trended group-by-group
  from model MSLP. MiniMax: BECMG to VRB03KT overnight, BECMG back to
  22015G25KT the next afternoon. Both encode the diurnal wind cycle straight
  off the BUFKIT point table; what they persisted was sky/vis — correctly,
  for a bone-dry July ridge. The REAL quality gaps this run exposed are
  subtler: Qwen forecast TN = -5C (it read the DEWPOINT as the temperature —
  a 23F overnight low in Las Vegas in July), and Gemma converted 1008 hPa to
  both 29.76 and 29.86 inHg within one reasoning block (arithmetic slip; the
  final QNH landed right by luck of which value it kept).
- Fix idea: update the status bullet (persistence claim -> "diurnal wind
  cycle captured; sky/vis correctly persisted") and the Known-problems
  emit-quality entry to cite the real gaps; fold both findings into the
  step-12 worksheet's sub-task list — TX/TN value+timing and QNH trend are
  already there; add "sanity-check TX/TN against the diurnal range of
  observed temperatures" and "double-check any unit conversion once".
- Verify: doc-only; re-read the bullet against the log.
