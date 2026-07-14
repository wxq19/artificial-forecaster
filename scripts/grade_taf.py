"""Detailed TAF grader -- produces an exhaustive hour-by-hour audit log showing how
each of the three scorers (TAFVER, amendment busts, skill) arrives at its numbers, with
every point/error shown and the totals summed explicitly. Non-LLM; reads a METAR list +
a TAF, scores through the same primitives the pipeline uses, writes a markdown log.

Usage:
  uv run python scripts/grade_taf.py --metars FILE --taf FILE --issue-date YYYY-MM-DD \
      [--station KLSV] [--out logs/grade_<station>_<stamp>.md]

METAR file: one report per line, optionally prefixed with a YYYYMMDDHHMM epoch (ignored;
obs_time comes from the report body + the issue month/year). TAF file: the raw TAF (may
span lines; they are joined). A malformed TX/TN token lacking its value is dropped and
flagged rather than crashing the parse.
"""

import argparse
import re
import tempfile
from datetime import datetime
from pathlib import Path

from forecaster import metar, store
from forecaster.tafparse import parse as parse_taf
from forecaster.tafstate import (
    absolute_validity, build_truth, conservative_state, default_profile, predominant_state,
)
from forecaster.tafamend import score_amend
from forecaster.tafskill import score_skill
from forecaster.tafver import score_tafver

_EPOCH = re.compile(r"^\d{12}\s+")
_BAD_TEMP = re.compile(r"\bT[XN](?![M]?\d{2}/)\S*")   # TX/TN token missing its value


def load_metars(path: str):
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(metar.parse(_EPOCH.sub("", line)))
    return out


def load_taf(path: str) -> tuple[str, str, bool]:
    raw0 = re.sub(r"\s+", " ", Path(path).read_text().strip()).rstrip("=").strip()
    fixed = re.sub(r"\s+", " ", _BAD_TEMP.sub("", raw0)).strip()
    return raw0, fixed, fixed != raw0


# --------------------------------------------------------------------------- render


def h(dt):
    return f"{dt:%d/%H}Z"


def _truth_rows(obs, vf, vt):
    hours, _ = build_truth(obs, vf, vt)
    rows = []
    for x in hours:
        if x.status != "available":
            rows.append((x.hour, x.status, x.reason, None, None))
            continue
        rows.append((x.hour, "avail", None, conservative_state(x), predominant_state(x)))
    return hours, rows


def sec_inputs(md, station, raw0, fixed, repaired, taf, vf, vt, obs, hours):
    md += [f"# TAF grade -- {station}", "",
           f"Generated {datetime.now():%Y-%m-%d %H:%M} local. Non-LLM audit; every point is shown and summed.",
           "", "## 1. Inputs", "",
           f"- **Valid:** {vf:%Y-%m-%dT%H:%MZ} .. {vt:%Y-%m-%dT%H:%MZ}  ({int((vt-vf).total_seconds()//3600)} h)",
           f"- **Issue:** day {taf.issue_day:02d} {taf.issue_time:%H%M}Z",
           f"- **Obs supplied:** {len(obs)} in the scoring reader "
           f"(window + carry-in/out); {sum(1 for o in obs if vf <= o['obs_time'] < vt)} inside the window.",
           f"- **Hourly truth:** {sum(1 for x in hours if x.status=='available')} of {len(hours)} "
           f"hours have obs coverage.", ""]
    if repaired:
        md += ["> **Source repair:** the TAF's `TN` group was malformed (missing its "
               "temperature value and `/` separator: `TN1211Z`). It was dropped so the rest "
               "could parse; **TN cannot be scored** (flagged as `no_forecast_extreme` below).", ""]
    md += ["**TAF (as supplied):**", "", "```", raw0, "```", ""]
    md += ["**Decoded groups:**", "",
           f"- INITIAL {vf:%d/%H}Z..{vt:%d/%H}Z: wind {taf.prevailing.wind_dir_deg}/"
           f"{taf.prevailing.wind_speed}"
           + (f"G{taf.prevailing.wind_gust}" if taf.prevailing.wind_gust else "")
           + f", {taf.prevailing.vis_m}m, QNH {taf.prevailing.qnh_inhg}"]
    for g in taf.groups:
        md.append(f"- {g.change_type} {g.from_day:02d}/{g.from_hour:02d}..{g.to_day:02d}/{g.to_hour:02d}: "
                  f"wind {g.wind_dir_deg}/{g.wind_speed}" + (f"G{g.wind_gust}" if g.wind_gust else "")
                  + f", QNH {g.qnh_inhg}, changed={sorted(g.explicit_fields)}")
    tx = taf.max_temp
    md.append(f"- TX {tx.temp_c}C @ {tx.day:02d}{tx.hour:02d}Z" if tx else "- TX: (none)")
    md.append("")
    return md


def sec_truth(md, rows):
    md += ["## 2. Observed hourly truth (conservative / union view)", "",
           "This is the pessimistic hourly truth TAFVER scores against (lowest cig/vis/altimeter, "
           "max wind/gust, union weather). Amend/skill use the predominant view; where they differ "
           "it is noted inline.", "",
           "| hour | wind | gust | vis (m) | ceiling | altimeter | wx |",
           "|---|---|---|---|---|---|---|"]
    for hour, status, reason, cons, _pred in rows:
        if cons is None:
            md.append(f"| {h(hour)} | _{reason}_ | | | | | |")
            continue
        wind = (f"{cons.wind_dir}/{cons.wind_speed}" if cons.wind_dir is not None
                else f"--/{cons.wind_speed}" if cons.wind_speed is not None else "--")
        gust = f"G{cons.wind_gust}" if cons.wind_gust else "-"
        cig = "unlim" if cons.ceiling_status == "known_unlimited" else (
            f"{cons.ceiling_ft}ft" if cons.ceiling_ft is not None else "?")
        vis = "unlim" if cons.vis_status == "known_unlimited" else (
            str(cons.vis_m) if cons.vis_m is not None else "?")
        alt = f"{cons.qnh_inhg:.2f}" if cons.qnh_inhg is not None else "-"
        wx = " ".join(cons.weather) or "-"
        md.append(f"| {h(hour)} | {wind} | {gust} | {vis} | {cig} | {alt} | {wx} |")
    md.append("")
    return md


def sec_tafver(md, s):
    md += ["## 3. TAFVER (official percent-correct, ACCI15-120 Att 7)", "",
           "Atomic unit: one resolved opportunity (a forecast group) x one UTC hour, scored 0/1 "
           "(present weather is fractional CSI) against the conservative truth. A BECMG transition "
           "hour has TWO opportunities (the old baseline + the becoming state); both are scored. "
           "The combined denominator is opportunities x elements.", ""]
    elements = ["ceiling", "visibility", "wind_speed", "wind_dir", "wind_gust",
                "present_weather", "altimeter"]
    for el in elements:
        el_rows = sorted([r for r in s.rows if r.element == el],
                         key=lambda r: (r.interval_start, r.group_type))
        md += [f"### 3.{elements.index(el)+1} {el}", "",
               "| hour | grp | forecast | observed | pts | note |",
               "|---|---|---|---|---|---|"]
        e = a = 0.0
        for r in el_rows:
            fcst = r.fcst_cat or r.fcst_value or "-"
            obsv = r.obs_cat or r.obs_value or "-"
            if r.status == "scored":
                pts = f"{r.points_earned:g}/{r.points_available}"
                e += r.points_earned
                a += r.points_available
            else:
                pts = f"n/a ({r.reason})"
            grp = r.group_type[:4] + (str(r.group_index) if r.group_type != "INITIAL" else "")
            md.append(f"| {h(r.interval_start)} | {grp} | {fcst} | {obsv} | {pts} | "
                      f"{r.reason or ''} |")
        pct = f"{100*e/a:.1f}%" if a else "n/a"
        md += [f"| **TOTAL** | | | | **{e:g}/{a:g}** | **{pct}** |", ""]

    md += ["### 3.8 TAFVER combined (the anti-averaging sum)", "",
           "| element | earned | available | percent |", "|---|---|---|---|"]
    for es in s.element_summaries:
        if es.bucket == "ALL":
            pct = f"{es.percent:.1f}" if es.percent is not None else "n/a"
            md.append(f"| {es.element} | {es.earned:g} | {es.available} | {pct} |")
    md += [f"| **COMBINED** | **{s.combined_earned:g}** | **{s.combined_available}** | "
           f"**{'' if s.combined_percent is None else f'{s.combined_percent:.1f}%'}** |", "",
           f"**Combined = sum(earned) / sum(available) = {s.combined_earned:g} / "
           f"{s.combined_available} = "
           f"{'' if s.combined_percent is None else f'{s.combined_percent:.2f}%'}**"
           f"{'  (PROVISIONAL policy)' if s.provisional else ''}", ""]
    if s.group_type_summaries:
        md += ["Diagnostic buckets (combined sums all regardless):", "",
               "| bucket | earned | available | percent |", "|---|---|---|---|"]
        for g in s.group_type_summaries:
            pct = f"{g.percent:.1f}" if g.percent is not None else "n/a"
            md.append(f"| {g.bucket} | {g.earned:g} | {g.available} | {pct} |")
        md.append("")
    if s.category_stats:
        md += ["Category accuracy + bias (cig/vis):", "",
               "| element | cat | fcst-hrs | obs-hrs | accuracy | bias |", "|---|---|---|---|---|---|"]
        for c in s.category_stats:
            acc = f"{c.accuracy:.2f}" if c.accuracy is not None else "n/a"
            bias = f"{c.bias:.2f}" if c.bias is not None else "n/a"
            md.append(f"| {c.element} | {c.category} | {c.fcst_hours} | {c.obs_hours} | {acc} | {bias} |")
        md.append("")
    return md


def sec_amend(md, s):
    rules = ["category", "wind", "altimeter", "thunderstorm", "tempo", "change_timing"]
    md += ["## 4. Amendment-implied busts (DAFI 15-129 sec 3.4)", "",
           "Each hour, every doctrine rule is checked against the predominant truth. `P`=in-spec, "
           "`X`=BUST, `-`=unavailable. Consecutive busts of one rule collapse to a rule episode; "
           "episodes sharing an onset hour merge into one amendment trigger.", "",
           "| hour | " + " | ".join(rules) + " |",
           "|---|" + "|".join(["---"] * len(rules)) + "|"]
    by = {}
    for r in s.hourly_results:
        by.setdefault(r.hour, {})[r.rule] = r
    for hour in sorted(by):
        cells = []
        for rule in rules:
            rr = by[hour].get(rule)
            cells.append("P" if rr and rr.result == "pass" else
                         "X" if rr and rr.result == "fail" else "-")
        md.append(f"| {h(hour)} | " + " | ".join(cells) + " |")
    md.append("")
    # fail details
    fails = [r for r in s.hourly_results if r.result == "fail"]
    if fails:
        md += ["**Bust details:**", ""]
        for r in sorted(fails, key=lambda r: (r.hour, r.rule)):
            md.append(f"- {h(r.hour)} {r.rule}: {r.detail or ''}")
        md.append("")
    md += ["**Rule episodes:**", ""]
    md += ([f"- {e.rule}: {h(e.onset)}..{h(e.end)} ({e.hours}h)"
            + (" [after-amd-service]" if e.after_amd_service else "")
            + (f" -- {e.worst_detail}" if e.worst_detail else "") for e in s.rule_episodes]
           or ["- (none)"])
    md += ["", "**Amendment triggers (headline):**", ""]
    md += ([f"- {h(t.onset)}: {', '.join(t.rules)}" for t in s.triggers] or ["- (none)"])
    md += ["",
           f"**Potential amendment triggers = {s.trigger_count}**"
           f"  (after-amd-service excluded: {s.triggers_after_amd_service})",
           f"**In-spec fraction = in-spec hours / scored hours = {s.hours_in_spec} / "
           f"{s.hours_scored} = "
           f"{'' if s.in_spec_fraction is None else f'{s.in_spec_fraction:.3f}'}**", ""]
    return md


def sec_skill(md, s):
    md += ["## 5. Skill (magnitude of error + event skill)", "",
           "Signed error is forecast - observed (positive = over-forecast). Hourly elements use the "
           "predominant view and the prevailing forecast only.", ""]
    # axis 1 hourly elements
    for el in ["wind_speed", "wind_dir", "wind_gust", "ceiling", "visibility"]:
        rows = sorted([r for r in s.element_rows if r.element == el and r.grain == "hour"],
                      key=lambda r: r.hour)
        md += [f"### 5.x {el}", "", "| hour | fcst | obs | signed err | abs err | note |",
               "|---|---|---|---|---|---|"]
        for r in rows:
            if r.status == "scored":
                md.append(f"| {h(r.hour)} | {r.fcst_value:g} | {r.obs_value:g} | "
                          f"{r.signed_error:+g} | {r.abs_error:g} | |")
            else:
                md.append(f"| {h(r.hour)} | - | - | | | {r.reason} |")
        md.append("")
    # qnh + txtn
    md += ["### 5.y QNH (per group) + TX/TN (per TAF)", "",
           "| element | grp | fcst | obs | signed err | note |", "|---|---|---|---|---|---|"]
    for r in [x for x in s.element_rows if x.grain in ("group", "taf")]:
        fv = f"{r.fcst_value:g}" if r.fcst_value is not None else "-"
        ov = f"{r.obs_value:g}" if r.obs_value is not None else "-"
        se = f"{r.signed_error:+.2f}" if r.signed_error is not None else ""
        md.append(f"| {r.element} | {r.group_type} | {fv} | {ov} | {se} | {r.reason or ''} |")
    md.append("")
    # element stats
    md += ["### 5.z Element bias / MAE (pooled overall)", "",
           "| element | n | bias | MAE |", "|---|---|---|---|"]
    for st in s.element_stats:
        if st.bin == "overall":
            b = f"{st.bias:.2f}" if st.bias is not None else ""
            md.append(f"| {st.element} | {st.n} | {b} | {st.mae:.2f} |")
    md.append("")
    # events
    md += ["### 5.e Event contingency (union view)", "",
           "| event | hit | miss | false_alarm | correct_neg | POD | FAR | CSI | HSS |",
           "|---|---|---|---|---|---|---|---|---|"]
    n_evt = 0
    for c in s.contingency:
        if c.a + c.b + c.c == 0:
            continue
        n_evt += 1
        f = lambda v: "" if v is None else f"{v:.2f}"  # noqa: E731
        md.append(f"| {c.event} | {c.a} | {c.c} | {c.b} | {c.d} | {f(c.pod)} | {f(c.far)} | "
                  f"{f(c.csi)} | {f(c.hss)} |")
    if n_evt == 0:
        md.append("| _(no catalog event was forecast or observed this period)_ | | | | | | | | |")
    md.append("")
    # ordinal
    md += ["### 5.o Ordinal category distance (MACE)", "",
           "| hour | fcst cat | obs cat | |delta| |", "|---|---|---|---|"]
    tot = 0
    for hourZ, fc, oc in s.category_series:
        d = abs("ABCDE".index(fc) - "ABCDE".index(oc))
        tot += d
        md.append(f"| {hourZ} | {fc} | {oc} | {d} |")
    n = len(s.category_series)
    md += [f"| **sum** | | | **{tot}** |", "",
           f"**MACE = sum|delta| / n = {tot} / {n} = "
           f"{'' if s.mace is None else f'{s.mace:.3f}'}**  |  worst excursion {s.worst_excursion}", ""]
    return md


def sec_summary(md, tv, am, sk):
    md += ["## 6. Headline summary", "",
           "| scorer | headline |", "|---|---|",
           f"| TAFVER | {'' if tv.combined_percent is None else f'{tv.combined_percent:.1f}%'} "
           f"correct ({tv.combined_earned:g}/{tv.combined_available} pts){' [provisional]' if tv.provisional else ''} |",
           f"| Amend | {am.trigger_count} potential amendment triggers; "
           f"in-spec {'' if am.in_spec_fraction is None else f'{am.in_spec_fraction:.2f}'} |",
           f"| Skill | MACE {'' if sk.mace is None else f'{sk.mace:.2f}'}; "
           f"worst excursion {sk.worst_excursion['delta'] if sk.worst_excursion else '-'} |", ""]
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description="Detailed hour-by-hour TAF grading log.")
    ap.add_argument("--metars", required=True)
    ap.add_argument("--taf", required=True)
    ap.add_argument("--issue-date", required=True, help="YYYY-MM-DD supplying the TAF's year/month")
    ap.add_argument("--station", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    issue_ref = datetime.strptime(args.issue_date, "%Y-%m-%d")
    metars = load_metars(args.metars)
    raw0, fixed, repaired = load_taf(args.taf)
    taf = parse_taf(fixed)
    station = args.station or taf.station
    _, vf, vt = absolute_validity(taf, issue_ref)

    # load obs through the store seam so the dict shape (+ carry-in/out) matches the pipeline
    dbf = tempfile.mkstemp(suffix=".duckdb")[1]
    Path(dbf).unlink()
    con = store.connect(dbf, read_only=False)
    store.init_schema(con)
    # split by (year, month) for insert_obs's single-month contract
    from collections import defaultdict
    by_ym = defaultdict(list)
    for o in metars:
        by_ym[(issue_ref.year, issue_ref.month)].append(o)
    for (yr, mo), lst in by_ym.items():
        store.insert_obs(con, lst, year=yr, month=mo, source="manual")
    obs = store.scoring_window(con, station, vf, vt)
    con.close()
    Path(dbf).unlink(missing_ok=True)

    profile = default_profile(station)
    tv = score_tafver(taf, obs, vf, vt, profile=profile)
    am = score_amend(taf, obs, vf, vt, profile=profile)
    sk = score_skill(taf, obs, vf, vt, profile=profile)

    hours, rows = _truth_rows(obs, vf, vt)
    md: list[str] = []
    md = sec_inputs(md, station, raw0, fixed, repaired, taf, vf, vt, obs, hours)
    md = sec_truth(md, rows)
    md = sec_tafver(md, tv)
    md = sec_amend(md, am)
    md = sec_skill(md, sk)
    md = sec_summary(md, tv, am, sk)

    Path("logs").mkdir(exist_ok=True)
    out = args.out or f"logs/grade_{station}_{datetime.now():%Y%m%d-%H%M%S}.md"
    Path(out).write_text("\n".join(md), encoding="utf-8")
    print(f"{station} {vf:%Y-%m-%dT%H:%MZ}..{vt:%H:%MZ}  "
          f"TAFVER={'' if tv.combined_percent is None else f'{tv.combined_percent:.1f}%'}  "
          f"triggers={am.trigger_count}  in_spec={am.in_spec_fraction}  MACE={sk.mace}")
    print(f"Report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
