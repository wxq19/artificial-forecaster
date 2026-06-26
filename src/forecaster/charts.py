"""Chart rendering — the ONLY file that imports matplotlib.

Seam like store.py/llm.py: callers hand in plain obs rows (list[dict] from
store.window) and get back PNG bytes; matplotlib never escapes this module. Uses
the headless Agg backend (no display on WSL/HPC). Charts feed the VLM as images,
so legibility at small size matters more than polish.
"""

import io

import matplotlib

matplotlib.use("Agg")  # headless: no display on WSL/SuperCloud
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from forecaster import wxcodes

# present-weather palette: family -> base hue, intensity -> alpha. Vis-driven
# families (fog/dust/haze) have no -/+ intensity, so they render at 'moderate'.
_WX_COLOR = {
    "rain": "#d62728",
    "snow": "#1f77b4",
    "freezing": "#9467bd",
    "thunder": "#8b0000",
    "fog": "#7f7f7f",
    "dust": "#d2b48c",
    "haze": "#808000",
    "ice": "#17becf",
    "other": "#555555",
}
_WX_ALPHA = {"light": 0.45, "moderate": 0.72, "heavy": 1.0, "vicinity": 0.2}


def _col(rows, key):
    """Column as floats, None -> nan so matplotlib leaves a GAP (missing data,
    or 'unlimited' ceiling / 'no gust') instead of erroring or drawing to zero."""
    return [r[key] if r[key] is not None else float("nan") for r in rows]


def _curate_families(classified) -> list[str]:
    """Families to surface on the meteogram band: top-2 by frequency (obs count)
    UNION top-3 by peak severity, ordered ascending so the worst lands on top.
    Frequency catches what dominated; severity catches brief-but-dangerous wx."""
    freq: dict[str, int] = {}
    peak: dict[str, float] = {}
    for groups in classified:
        seen = set()
        for g in groups:
            peak[g.family] = max(peak.get(g.family, 0.0), g.severity)
            if g.family not in seen:        # count each family once per ob
                freq[g.family] = freq.get(g.family, 0) + 1
                seen.add(g.family)
    top_freq = sorted(freq, key=lambda f: freq[f], reverse=True)[:2]
    top_sev = sorted(peak, key=lambda f: peak[f], reverse=True)[:3]
    chosen = set(top_freq) | set(top_sev)
    return sorted(chosen, key=lambda f: peak[f])


def _draw_wx_band(ax, classified, xnums, families, med) -> None:
    """Fill the present-weather color band on `ax`: one row per family (worst on
    top), each phenomenon a colored span from its ob to the next. Families not in
    `families` are skipped, so the meteogram band can show a curated subset while
    wx_timeline passes them all."""
    row = {f: i for i, f in enumerate(families)}
    for i, groups in enumerate(classified):
        w = (xnums[i + 1] - xnums[i]) if i + 1 < len(xnums) else med
        for g in groups:
            if g.family not in row:
                continue
            ax.add_patch(Rectangle(
                (xnums[i], row[g.family] - 0.4), w, 0.8,
                facecolor=_WX_COLOR.get(g.family, "#555555"),
                alpha=_WX_ALPHA.get(g.intensity, 0.7), edgecolor="none",
            ))
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=8)
    ax.set_ylim(-0.6, len(families) - 0.4)


def meteogram(rows: list[dict], *, station: str, hours: int) -> bytes:
    """Stacked multi-panel meteogram of recent METARs -> PNG bytes. Panels share a
    UTC time axis: T/Td, wind, visibility, ceiling, pressure. `rows` are
    store.window() dicts (chronological). Returns PNG bytes; the caller decides
    whether to persist (for a log) or base64 it for the VLM."""
    times = [r["obs_time"] for r in rows]
    fig = plt.figure(figsize=(9, 13.5))
    gs = fig.add_gridspec(7, 1, height_ratios=[3, 3, 3, 3, 3, 2.4, 1.4])
    ax_t = fig.add_subplot(gs[0])
    ax_w = fig.add_subplot(gs[1], sharex=ax_t)
    ax_v = fig.add_subplot(gs[2], sharex=ax_t)
    ax_c = fig.add_subplot(gs[3], sharex=ax_t)
    ax_p = fig.add_subplot(gs[4], sharex=ax_t)
    ax_x = fig.add_subplot(gs[5], sharex=ax_t)
    lax = fig.add_subplot(gs[6])   # present-weather key: independent (not time) x
    axes = [ax_t, ax_w, ax_v, ax_c, ax_p, ax_x]

    ax_t.plot(times, _col(rows, "temp_c"), color="tab:red", label="T")
    ax_t.plot(times, _col(rows, "dewpoint_c"), color="tab:green", label="Td")
    ax_t.set_ylabel("C")
    ax_t.legend(loc="upper left")
    ax_t.grid(True, alpha=0.3)

    ax_w.plot(times, _col(rows, "wind_speed"), color="tab:blue", label="spd")
    ax_w.plot(
        times, _col(rows, "wind_gust"), color="tab:blue",
        linestyle=":", marker=".", label="gust",
    )
    ax_w.set_ylabel("kt")
    ax_w.legend(loc="upper left")
    ax_w.grid(True, alpha=0.3)

    ax_v.plot(times, _col(rows, "vis_sm"), color="tab:purple", marker=".")
    ax_v.set_ylabel("vis sm")
    ax_v.grid(True, alpha=0.3)

    ax_c.plot(times, _col(rows, "ceiling_ft"), color="tab:brown", marker=".")
    ax_c.set_ylabel("ceil ft")  # gaps = unlimited
    ax_c.grid(True, alpha=0.3)

    ax_p.plot(times, _col(rows, "altimeter_inhg"), color="tab:gray")
    ax_p.set_ylabel("inHg")
    ax_p.grid(True, alpha=0.3)

    # present-weather band: a curated colored timeline (top-2 frequent UNION
    # top-3 severe families). Hue = family, opacity = intensity. The standalone
    # wx_timeline() shows ALL families plus the full phenomenon x intensity key.
    classified = [wxcodes.classify_ob(r["weather"], r["vis_sm"]) for r in rows]
    band_families = _curate_families(classified)
    xnums = [mdates.date2num(t) for t in times]
    deltas = [xnums[i + 1] - xnums[i] for i in range(len(xnums) - 1)]
    med = sorted(deltas)[len(deltas) // 2] if deltas else 1 / 24
    _draw_wx_band(ax_x, classified, xnums, band_families, med)
    ax_x.set_ylabel("wx")
    _wx_legend(lax, band_families)   # small phenomenon x intensity key for the band

    # hour ticks under EVERY panel (shared-x hides inner ones by default); the
    # bottom strip anchors the day.
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%HZ"))
        ax.tick_params(axis="x", labelbottom=True, labelsize=8)
    ax_x.xaxis.set_major_formatter(mdates.DateFormatter("%d/%HZ"))

    fig.suptitle(
        f"{station} meteogram - last {hours}h "
        f"({times[0]:%Y-%m-%d %H:%MZ} to {times[-1]:%Y-%m-%d %H:%MZ})"
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)  # release the figure - matplotlib leaks if you don't
    return buf.getvalue()


def wx_timeline(rows: list[dict], *, station: str, hours: int) -> bytes:
    """Categorical present-weather timeline -> PNG bytes. One row per weather
    FAMILY (worst on top, by peak severity from wxcodes), each phenomenon filled
    as a colored span from the ob that reports it until the next ob clears it.
    Hue = family, alpha = intensity. Complements the meteogram's numeric panels:
    color regions are gestalt a VLM reads reliably, with no text to misparse."""
    times = [r["obs_time"] for r in rows]
    xnums = [mdates.date2num(t) for t in times]
    classified = [wxcodes.classify_ob(r["weather"], r["vis_sm"]) for r in rows]

    # families present, ordered by PEAK severity (least at bottom, worst on top)
    peak: dict[str, float] = {}
    for groups in classified:
        for g in groups:
            peak[g.family] = max(peak.get(g.family, 0.0), g.severity)
    families = sorted(peak, key=lambda f: peak[f])

    # width of a span = gap to the next ob; the last ob uses the median interval
    deltas = [xnums[i + 1] - xnums[i] for i in range(len(xnums) - 1)]
    med = sorted(deltas)[len(deltas) // 2] if deltas else 1 / 24

    n = len(families)
    th, lh = 0.7 * n + 1.2, 0.3 * n + 0.6   # timeline / legend axes heights (in)
    fig, (ax, lax) = plt.subplots(
        2, 1, figsize=(11, th + lh), gridspec_kw={"height_ratios": [th, lh]}
    )
    _draw_wx_band(ax, classified, xnums, families, med)
    ax.set_xlim(xnums[0], xnums[-1] + med)
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%HZ"))
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_title(f"{station} present-weather timeline - last {hours}h")

    _wx_legend(lax, families)   # phenomenon (rows) x intensity (cols) swatch grid
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _wx_legend(lax, families) -> None:
    """Draw the present-weather key as a grid: rows = families (same order as the
    timeline, worst on top), columns = intensity. Each cell is the family hue at
    that intensity's opacity, so both axes read at once."""
    cols = ["light", "moderate", "heavy", "vicinity"]
    rows = list(reversed(families))            # worst family on top
    lax.set_xlim(-3.0, len(cols) + 2.0)   # extra margin both sides -> centered
    lax.set_ylim(0, len(rows) + 0.7)
    for ri, fam in enumerate(rows):
        y = len(rows) - 1 - ri                 # row 0 at top
        lax.text(-0.15, y + 0.45, fam, ha="right", va="center", fontsize=9)
        for ci, inten in enumerate(cols):
            lax.add_patch(Rectangle(
                (ci, y), 0.9, 0.9, facecolor=_WX_COLOR[fam],
                alpha=_WX_ALPHA[inten], edgecolor="lightgray", linewidth=0.5,
            ))
    for ci, inten in enumerate(cols):
        lax.text(ci + 0.45, len(rows) + 0.1, inten, ha="center", va="bottom", fontsize=9)
    lax.set_xticks([])
    lax.set_yticks([])
    for spine in lax.spines.values():
        spine.set_visible(False)
