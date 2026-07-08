"""Chart rendering — the ONLY file that imports matplotlib.

Seam like store.py/llm.py: callers hand in plain obs rows (list[dict] from
store.window) and get back PNG bytes; matplotlib never escapes this module. Uses
the headless Agg backend (no display on WSL/HPC). Charts feed the VLM as images,
so legibility at small size matters more than polish.
"""

import io

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless: no display on WSL/SuperCloud
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# MetPy is PyPI (no conda) and drives matplotlib, so the skew-T renderer stays inside
# this seam rather than adding a second plotting file.
import metpy.calc as mpcalc
from metpy.plots import Hodograph, SkewT
from metpy.units import units

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


def skewt(profile) -> bytes:
    """Enriched forecast skew-T from a fcstsounding.FcstProfile -> PNG bytes.

    Plots temperature/dewpoint, a surface-parcel path with CAPE/CIN shading and the LCL,
    a height-colored hodograph, and the model's stability indices. `profile` is
    duck-typed: any object exposing pres/tmpc/dwpc/drct/sknt/hght (lists), indices (dict),
    and station/model/run/fhr/valid metadata. The right column is pinned to the skew-T
    box so the hodograph top and indices bottom align with the chart."""
    p = np.array(profile.pres) * units.hPa
    T = np.array(profile.tmpc) * units.degC
    Td = np.array(profile.dwpc) * units.degC
    u, v = mpcalc.wind_components(np.array(profile.sknt) * units.knots,
                                  np.array(profile.drct) * units.degrees)
    hght = np.array(profile.hght) * units.m

    fig = plt.figure(figsize=(11, 9))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1])
    skew = SkewT(fig, rotation=45, subplot=gs[0, 0])
    skew.plot(p, T, "r", linewidth=2, label="Temperature")
    skew.plot(p, Td, "g", linewidth=2, label="Dewpoint")
    skew.plot_barbs(p[::3], u[::3], v[::3])
    skew.ax.set_ylim(1000, 150)
    skew.ax.set_xlim(-40, 50)
    skew.ax.set_xlabel("Temperature (°C)")
    skew.ax.set_ylabel("Pressure (hPa)")
    skew.plot_dry_adiabats(alpha=0.25)
    skew.plot_moist_adiabats(alpha=0.25)
    skew.plot_mixing_lines(alpha=0.25)

    # surface-parcel path + CAPE/CIN shading + LCL (MetPy does the thermo)
    try:
        parcel = mpcalc.parcel_profile(p, T[0], Td[0]).to("degC")
        skew.plot(p, parcel, "k", linewidth=1.2, linestyle="--", label="Parcel")
        skew.shade_cape(p, T, parcel)
        skew.shade_cin(p, T, parcel, Td)
        lcl_p, lcl_t = mpcalc.lcl(p[0], T[0], Td[0])
        skew.plot(lcl_p, lcl_t, "ko", markerfacecolor="black")
        skew.ax.text(lcl_t.m + 1, lcl_p.m, "LCL", fontsize=8, va="center")
    except Exception:  # noqa: BLE001 -- thermo edge cases shouldn't kill the plot
        pass
    skew.ax.legend(loc="upper center", fontsize=8, ncol=4, frameon=False)

    # right column: hodograph on top, indices box beneath
    gs_r = gs[0, 1].subgridspec(2, 1, height_ratios=[1.2, 1], hspace=0.08)
    ax_h = fig.add_subplot(gs_r[0])
    ax_txt = fig.add_subplot(gs_r[1])
    ax_txt.axis("off")

    top = p >= 200 * units.hPa            # keep the hodograph to the troposphere
    spd = np.hypot(u[top].m, v[top].m)
    rng = max(30, int(np.ceil((spd.max() if spd.size else 30) / 10) * 10) + 10)
    hod = Hodograph(ax_h, component_range=rng)
    hod.add_grid(increment=10 if rng <= 40 else 20)
    hod.plot_colormapped(u[top], v[top], hght[top])
    ax_h.set_title("Hodograph (kt, by height)", fontsize=9)

    order = ("CAPE", "CINS", "LIFT", "SHOW", "KINX", "TOTL", "PWAT", "LCLP")
    lines = [f"{n:<5}{profile.indices[n]:>8.0f}" if n in ("CAPE", "CINS")
             else f"{n:<5}{profile.indices[n]:>8.1f}"
             for n in order if n in profile.indices]
    ax_txt.text(0.0, 0.0, f"{profile.model.upper()} indices\n" + "\n".join(lines),
                family="monospace", fontsize=9, va="bottom", ha="left",
                transform=ax_txt.transAxes,
                bbox=dict(boxstyle="round", facecolor="#f4f4f4", edgecolor="#bbb"))

    fig.suptitle(
        f"{profile.model.upper()} forecast skew-T  |  {profile.station}  "
        f"run {profile.run:%Y-%m-%d %HZ}  f{profile.fhr:03d}  valid {profile.valid}",
        fontsize=11)

    # SkewT under-fills its gridspec cell; pin hodograph top -> chart top and indices
    # bottom -> chart bottom by reading the skew-T box after a draw (uniform tight-crop
    # at save then preserves the alignment).
    fig.canvas.draw()
    sp = skew.ax.get_position()
    fw, fh = fig.get_size_inches()
    hp = ax_h.get_position()
    hh = hp.width * fw / fh                # square hodograph (equal aspect fills it)
    ax_h.set_position([hp.x0, sp.y1 - hh, hp.width, hh])
    tp = ax_txt.get_position()
    ax_txt.set_position([tp.x0, sp.y0, tp.width, (sp.y1 - hh - 0.03) - sp.y0])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)  # release the figure - matplotlib leaks if you don't
    return buf.getvalue()
