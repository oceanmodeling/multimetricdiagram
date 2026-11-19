"""Taylor diagram figure and axes: make_fig_axes()."""

from numbers import Real
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.ticker import MaxNLocator

from ..._text import text_polar
from ...axes import _check_axes_sizes
from ...labels import _OBSTACLE_GID

_OPTIONS_ATTR = '_mmd_taylor_options'  # on ax_polar, read by add_markerstick()
_CC_TICKS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
_TICK_LOCATOR = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])  # round MSD/OSD radii
_LABEL_GAP = 0.4  # fontsize fractions, from an axis or the arc to its tick labels
_TITLE_GAP = 0.6  # fontsize fractions, from tick labels to their title
_TITLES_GAP = 2.0  # fontsize fractions, between the two titles under a 2-quadrant axis
_CC_TITLE = "CC (marker)"  # beyond the outer arc


def make_fig_axes(*, return_artists: bool = False, **overrides) -> tuple[Any, ...]:
    """
    Create the Taylor diagram figure and its polar and legend axes.

    Each model is placed at radius MSD (its standard deviation) and angle arccos(CC); the star
    is the observations (radius OSD, CC = 1), so a marker's distance from the star is URMSD.
    Above the diagram is an empty legend axes for add_labeledmarker_legend() or matplotlib's
    ax.legend() / fig.legend(). Page size, margins and font size are those of the top-level
    make_fig_axes(); there is no shading or grayscale bar.

    Args:
        return_artists: False (default): return (fig, ax_polar, ax_legend). True: also return
            a dict of the background artists, grouped, to restyle or remove them (e.g.
            thinner lines for a smaller figure). Keys, in drawing order, each an artist or a
            list of artists (all text is text_polar() patches):
            - 'cc_radials': the dotted radials of constant CC (Line2Ds).
            - 'msd_circles': the dashed MSD/OSD circles about the origin (Line2Ds).
            - 'rmsd_circles': the solid RMSD/OSD circles about the star (Line2Ds).
            - 'rmsd_tick_labels': the horizontal axis's tick labels, right of the origin.
            - 'msd_tick_labels': the MSD/OSD axis's tick labels.
            - 'axis_titles': the horizontal axis's title, then the MSD/OSD axis's.
            - 'cc_tick_labels': the CC values along the outer arc.
            - 'cc_title': 'CC (marker)', beyond the outer arc.
            - 'star': the star (Line2D).
            Not included: the axes' own spines and patch (the outer arc is
            ax_polar.spines['polar']).
        **overrides: Optional settings replacing the defaults in the function body (config):
            figsize_inch, margins_t_b_l_r_inch, vgap_legend_rel_polar_inch and fontsize, as in
            the top-level make_fig_axes(); and
            - quadrants: None (default: 2 if any dict in metrics has CC < 0, else 1), or 1
              (CC 0 to 1) or 2 (CC -1 to 1) to choose. On 1 quadrant, add_markerstick() raises
              for CC < 0.
            - metrics: list of calculate_metrics() dicts (e.g. those to be plotted), for the
              quadrants=None and r_max=None choices; default None.
            - normalized: True (default; radii in units of OSD, star at 1) or False (data
              units; then units (str, e.g. 'm/s') and osd (the observations' OSD, from
              calculate_metrics()) are required, and r_max or metrics; the star is at osd).
            - r_max: outer radius; None (default): with metrics, the first round MSD tick at
              or above max(1.5 x the star radius, 1.05 x the largest marker radius), i.e.
              max(1.5, 1.05 x the largest MSD/OSD) normalized and max(1.5 x osd, 1.05 x the
              largest MSD) unnormalized, so the outer arc is labeled; without metrics 1.5
              (normalized only).
            - pad_t_b_l_r_inch: room around the diagram for its labels (default depends on
              quadrants).
            - grid_info: dict whose keys are merged into grid_info_defaults: line width,
              linestyle, color and zorder of each gridline family; outer arc width and color.
            - patternonly: False (default) or True, for metrics from
              calculate_metrics(patternonly=True) (MB = 0, so no sticks): the axis titles
              then name no RMSD and no "(marker)" (see Notes).

    Returns:
        (fig, ax_polar, ax_legend): the Figure, the diagram's polar axes, and the legend axes;
        with return_artists=True, (fig, ax_polar, ax_legend, artists), artists the dict above.

    Raises:
        TypeError: an unknown keyword (the message lists the valid names); patternonly is not
            a bool.
        ValueError: quadrants is not None, 1 or 2; metrics is not a list of dicts with 'CC'
            (and 'MSDonOSD' or 'MSD' to choose r_max); normalized is not a bool;
            normalized=False without units, osd, or both r_max and metrics, or normalized=True
            with units or osd; units is not a
            str; osd or r_max is not a finite number > 0, or r_max is not beyond the star; an
            unknown grid_info key (the message lists the valid keys); the polar or legend
            axes would get a width or height <= 0 (the message names each and its size in
            inches); or patternonly=True with a metrics dict whose MB is not 0 (not from
            calculate_metrics(patternonly=True)).

    Notes:
        Gridlines, gray: radial lines of constant CC (dotted); circles about the origin of
        constant MSD/OSD (dashed); and circles about the star of constant RMSD/OSD =
        URMSD/OSD (solid): a marker lies on the circle of its URMSD/OSD, and with ddof = 0
        its stick ends on the circle of its RMSD/OSD. With normalized=False, the circles are
        MSD and RMSD = URMSD, in data units. The outer arc is solid black.

        The horizontal axis right of the origin is labeled where the star-centered circles
        cross it, on both sides of the star, with their RMSD/OSD values; the star itself
        and the origin are not labeled there. The MSD/OSD axis (vertical, or with 2
        quadrants the horizontal axis left of the origin) is labeled where the
        origin-centered circles cross it, from 0 at the origin to r_max. CC is labeled along
        the outer arc. Titles: the horizontal axis "Solid contours: RMSD/OSD (stick tip),
        URMSD/OSD (marker)", the MSD/OSD axis "Dashed contours: MSD/OSD (marker)" (with
        normalized=False, "Solid contours: RMSD (stick tip), URMSD (marker) [units]" and
        "Dashed contours: MSD (marker) [units]"), and "CC (marker)" beyond the outer arc; with 2
        quadrants the MSD title moves left of its half's middle if needed to clear the other.
        With patternonly=True the titles are "Solid contours: URMSD/OSD" and "Dashed contours:
        MSD/OSD" ("Solid contours: URMSD [units]" and "Dashed contours: MSD [units]" with
        normalized=False); "CC (marker)" is unchanged.
        adjust_markerstick_labels() keeps labels off the star and this text.

        Gridlines use zorder < 2 so markersticks (default zorder 2) draw on top: 1.1 CC
        radials, 1.2 MSD/OSD circles, 1.3 RMSD/OSD circles, 1.5 star.
    """
    config = {
        "figsize_inch": (8.5, 11),
        "margins_t_b_l_r_inch": (1, 1, 0.5, 0.5),
        "vgap_legend_rel_polar_inch": 0,
        "fontsize": 10,
        "quadrants": None,
        "metrics": None,
        "normalized": True,
        "units": None,
        "osd": None,
        "r_max": None,
        "pad_t_b_l_r_inch": None,
        "grid_info": {},
        "patternonly": False,
    }
    grid_info_defaults = {
        "cc_radials_lw": 0.8,
        "cc_radials_ls": ":",
        "cc_radials_color": "0.5",
        "cc_radials_zo": 1.1,
        "msd_circles_lw": 0.6,
        "msd_circles_ls": "--",
        "msd_circles_color": "0.5",
        "msd_circles_zo": 1.2,
        "rmsd_circles_lw": 0.6,
        "rmsd_circles_ls": "-",
        "rmsd_circles_color": "0.5",
        "rmsd_circles_zo": 1.3,
        "outer_arc_lw": 1.5,
        "outer_arc_color": "k",
    }
    pad_defaults = {1: (0.45, 0.55, 0.7, 0.45), 2: (0.7, 0.55, 0.5, 0.5)}
    unknown = [k for k in overrides if k not in config]
    if unknown:
        raise TypeError(
            f"make_fig_axes() got unexpected keyword argument(s) "
            f"{', '.join(map(repr, unknown))}; valid names: {', '.join(config)}"
        )
    config.update(overrides)
    unknown = [k for k in config["grid_info"] if k not in grid_info_defaults]
    if unknown:
        raise ValueError(
            f"unknown grid_info key(s) {', '.join(map(repr, unknown))}; "
            f"valid keys: {', '.join(grid_info_defaults)}"
        )
    patternonly = config["patternonly"]
    if not isinstance(patternonly, (bool, np.bool_)):
        raise TypeError(f"patternonly must be a bool, got {patternonly!r}")
    quadrants = _quadrants(config["quadrants"], config["metrics"])
    if patternonly:
        _check_patternonly_metrics(config["metrics"])
    normalized, units = config["normalized"], config["units"]
    star_r, r_max = _check_options(
        quadrants, normalized, units, config["osd"], config["r_max"], config["metrics"]
    )
    figsize_inch = config["figsize_inch"]
    margins_t_b_l_r_inch = config["margins_t_b_l_r_inch"]
    fontsize = config["fontsize"]
    grid_info = {**grid_info_defaults, **config["grid_info"]}
    pad_t, pad_b, pad_l, pad_r = config["pad_t_b_l_r_inch"] or pad_defaults[quadrants]

    # layout (inches): diagram (box_w x box_h) at the bottom, legend axes on top
    wid_inch = figsize_inch[0] - margins_t_b_l_r_inch[2] - margins_t_b_l_r_inch[3]
    box_w = wid_inch - pad_l - pad_r
    box_h = box_w if quadrants == 1 else box_w / 2
    box_bottom = margins_t_b_l_r_inch[1] + pad_b
    # the polar axes box is square: matplotlib pads a half disk's bounding box to a square, so
    # that box reaches a quarter of its width below and above the half disk
    axes_bottom = box_bottom - (box_w - box_h) / 2
    legend_bottom = box_bottom + box_h + pad_t + config["vgap_legend_rel_polar_inch"]
    legend_h = figsize_inch[1] - legend_bottom - margins_t_b_l_r_inch[0]
    _check_axes_sizes(
        figsize_inch,
        margins_t_b_l_r_inch,
        {"polar axes": (box_w, box_h), "legend axes": (wid_inch, legend_h)},
        also_take_room="pad_t_b_l_r_inch and vgap_legend_rel_polar_inch also take room",
    )

    fig = plt.figure(figsize=figsize_inch)
    fig_w, fig_h = figsize_inch
    pos = (
        (margins_t_b_l_r_inch[2] + pad_l) / fig_w,
        axes_bottom / fig_h,
        box_w / fig_w,
        box_w / fig_h,
    )
    ax_polar = fig.add_axes(pos, projection='polar', zorder=0)
    ax_polar.set_thetamin(0.0)  # type: ignore
    ax_polar.set_thetamax(90.0 * quadrants)  # type: ignore
    ax_polar.set_ylim(0, r_max)
    ax_polar.set_autoscale_on(False)
    ax_polar.set_xticks([])
    ax_polar.set_yticks([])
    ax_polar.grid(False)
    ax_polar.set_facecolor('none')
    ax_polar.spines['polar'].set(
        linewidth=grid_info["outer_arc_lw"], color=grid_info["outer_arc_color"], linestyle="-"
    )
    setattr(
        ax_polar,
        _OPTIONS_ATTR,
        {
            "quadrants": quadrants,
            "normalized": bool(normalized),
            "osd": None if normalized else star_r,
            "star_r": star_r,
            "r_max": r_max,
            "patternonly": bool(patternonly),
        },
    )

    pos = (
        margins_t_b_l_r_inch[2] / fig_w,
        legend_bottom / fig_h,
        wid_inch / fig_w,
        legend_h / fig_h,
    )
    ax_legend = fig.add_axes(pos, zorder=0)
    ax_legend.axis('off')

    msd_ticks = _radius_ticks(r_max)
    rmsd_radii = _rmsd_radii(msd_ticks[1] - msd_ticks[0], quadrants, star_r, r_max)
    artists = _draw_gridlines(ax_polar, quadrants, star_r, r_max, msd_ticks, rmsd_radii, grid_info)
    artists |= _draw_axis_labels(
        ax_polar,
        quadrants,
        normalized,
        units,
        star_r,
        r_max,
        msd_ticks,
        rmsd_radii,
        fontsize,
        72 * box_w / (r_max * (1 if quadrants == 1 else 2)),
        patternonly,
    )
    artists |= _draw_cc_labels(ax_polar, quadrants, r_max, fontsize)
    # star at the reference point: the observations
    (artists["star"],) = ax_polar.plot(
        0,
        star_r,
        marker="*",
        markersize=14,
        color="black",
        markeredgecolor="black",
        clip_on=False,
        zorder=1.5,
        gid=_OBSTACLE_GID,
    )
    if return_artists:
        return fig, ax_polar, ax_legend, artists
    return fig, ax_polar, ax_legend


def _quadrants(quadrants, metrics):
    """quadrants as given (1 or 2), or for None, 2 if any metrics dict has CC < 0, else 1."""
    if metrics is not None and (
        not isinstance(metrics, (list, tuple))
        or not all(isinstance(m, dict) and 'CC' in m for m in metrics)
    ):
        raise ValueError("metrics must be a list of calculate_metrics() dicts (with 'CC')")
    if quadrants is None:
        return 2 if metrics and any(m['CC'] < 0 for m in metrics) else 1
    return quadrants


def _check_patternonly_metrics(metrics):
    """For patternonly=True: every metrics dict must have zero bias (MB/OSD = 0, or MB = 0)."""
    biased = [i for i, m in enumerate(metrics or ()) if m.get('MBonOSD', m.get('MB', 0)) != 0]
    if biased:
        raise ValueError(
            f"patternonly=True needs zero-bias metrics, but metrics[{biased[0]}] has MB != 0; "
            "compute them with calculate_metrics(..., patternonly=True)"
        )


def _check_options(quadrants, normalized, units, osd, r_max, metrics):
    """(star radius, r_max) after checking the make_fig_axes() options; see its Raises."""
    if isinstance(quadrants, bool) or quadrants not in (1, 2):
        raise ValueError(f"quadrants must be None (automatic), 1 or 2, got {quadrants!r}")
    if not isinstance(normalized, (bool, np.bool_)):
        raise ValueError(  # noqa: TRY004 (ValueError is documented and tested)
            f"normalized must be True or False, got {normalized!r}"
        )
    if normalized:
        given = [name for name, value in (("units", units), ("osd", osd)) if value is not None]
        if given:
            raise ValueError(
                f"{' and '.join(given)} apply only with normalized=False "
                "(normalized radii are in units of OSD)"
            )
        star_r = 1.0
    else:
        missing = [name for name, value in (("units", units), ("osd", osd)) if value is None]
        if r_max is None and not metrics:
            missing.append("r_max (or metrics=, to choose it)")
        if missing:
            raise ValueError(f"normalized=False needs {', '.join(missing)}")
        if not isinstance(units, str):
            raise ValueError(f"units must be a str, got {units!r}")
        _check_positive("osd", osd)
        star_r = float(osd)
    if r_max is None:
        r_max = _auto_r_max(metrics, normalized, star_r) if metrics else 1.5
    _check_positive("r_max", r_max)
    if not r_max > star_r:
        raise ValueError(f"r_max={r_max!r} must be beyond the star (radius {star_r!r})")
    return star_r, float(r_max)


def _check_positive(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite number > 0, got {value!r}")


def _auto_r_max(metrics, normalized, star_r):
    """r_max for metrics: the first round MSD tick (as _radius_ticks()) at or above
    max(1.5 x the star radius, 1.05 x the largest marker radius, MSD/OSD or MSD)."""
    key = 'MSDonOSD' if normalized else 'MSD'
    if not all(key in m for m in metrics):
        raise ValueError(
            f"metrics must be calculate_metrics() dicts (with {key!r}) to choose r_max"
        )
    target = max(1.5 * star_r, 1.05 * max(m[key] for m in metrics))
    ticks = _TICK_LOCATOR.tick_values(0, target)
    return float(f"{ticks[ticks >= target * (1 - 1e-9)].min():.12g}")  # rounded: a tick itself


def _radius_ticks(r_max):
    """Evenly spaced round radii from 0 to r_max, for the MSD/OSD circles and labels."""
    ticks = _TICK_LOCATOR.tick_values(0, r_max)
    return ticks[(ticks >= 0) & (ticks <= r_max * (1 + 1e-9))]


def _rmsd_radii(step, quadrants, star_r, r_max):
    """Radii of the star-centered circles: multiples of step out to the diagram's farthest
    point from the star ((0, r_max), or (-r_max, 0) with 2 quadrants)."""
    farthest = np.hypot(star_r, r_max) if quadrants == 1 else star_r + r_max
    return step * np.arange(1, int(farthest / step * (1 + 1e-9)) + 1)


def _fmt(values):
    """Tick labels for values, all with the decimals the most precise one needs."""
    decimals = max(len(f"{v:.10g}".partition(".")[2]) for v in values)
    return [f"{v:.{decimals}f}" for v in values]


def _ink_size_em(texts):
    """(widest, tallest) ink extent of texts, in fontsize fractions."""
    extents = [
        TextPath((0, 0), t, size=1, prop=FontProperties(weight='normal')).get_extents()
        for t in texts
    ]
    return max(e.width for e in extents), max(e.height for e in extents)


def _draw_gridlines(ax, quadrants, star_r, r_max, msd_ticks, rmsd_radii, grid_info):
    """CC radials, MSD/OSD circles about the origin and RMSD/OSD circles about the star,
    clipped to the diagram by its axes patch; returns them, grouped (see make_fig_axes())."""
    drawn = {"cc_radials": [], "msd_circles": [], "rmsd_circles": []}
    for cc in _CC_TICKS + (tuple(-c for c in _CC_TICKS) if quadrants == 2 else ()):
        drawn["cc_radials"] += ax.plot(
            [np.arccos(cc)] * 2,
            [0, r_max],
            color=grid_info["cc_radials_color"],
            linestyle=grid_info["cc_radials_ls"],
            linewidth=grid_info["cc_radials_lw"],
            zorder=grid_info["cc_radials_zo"],
        )
    theta = np.linspace(0, quadrants * np.pi / 2, 361)
    for rho in msd_ticks[(msd_ticks > 0) & (msd_ticks < r_max)]:  # r_max: the outer arc
        drawn["msd_circles"] += ax.plot(
            theta,
            np.full_like(theta, rho),
            color=grid_info["msd_circles_color"],
            linestyle=grid_info["msd_circles_ls"],
            linewidth=grid_info["msd_circles_lw"],
            zorder=grid_info["msd_circles_zo"],
        )
    t = np.linspace(0, np.pi, 1441)  # upper half of each circle; the rest is below the axis
    for radius in rmsd_radii:
        x, y = star_r + radius * np.cos(t), radius * np.sin(t)
        drawn["rmsd_circles"] += ax.plot(
            np.arctan2(y, x),
            np.hypot(x, y),
            color=grid_info["rmsd_circles_color"],
            linestyle=grid_info["rmsd_circles_ls"],
            linewidth=grid_info["rmsd_circles_lw"],
            zorder=grid_info["rmsd_circles_zo"],
        )
    return drawn


def _axis_titles(normalized, units, patternonly=False):
    """(horizontal axis title, MSD axis title); patternonly: no RMSD (no sticks), no "(marker)"."""
    if patternonly:
        solid, dashed = ("URMSD/OSD", "MSD/OSD") if normalized else ("URMSD", "MSD")
    elif normalized:
        solid, dashed = "RMSD/OSD (stick tip), URMSD/OSD (marker)", "MSD/OSD (marker)"
    else:
        solid, dashed = "RMSD (stick tip), URMSD (marker)", "MSD (marker)"
    suffix = "" if normalized else f" [{units}]"
    return f"Solid contours: {solid}{suffix}", f"Dashed contours: {dashed}{suffix}"


def _draw_axis_labels(
    ax,
    quadrants,
    normalized,
    units,
    star_r,
    r_max,
    msd_ticks,
    rmsd_radii,
    fontsize,
    pt_per_data,
    patternonly=False,
):
    """Tick labels and titles of the RMSD/OSD axis (right of the origin) and the MSD/OSD axis
    (vertical, or left of the origin); pt_per_data: points per radius unit. Returns them,
    grouped (see make_fig_axes())."""
    tol = 1e-9 * r_max
    crossings = sorted(
        [(star_r - d, d) for d in rmsd_radii if star_r - d > tol]
        + [(star_r + d, d) for d in rmsd_radii if star_r + d <= r_max + tol]
    )
    labels = _fmt(list(msd_ticks) + [d for _x, d in crossings])
    msd_labels, rmsd_labels = labels[: len(msd_ticks)], labels[len(msd_ticks) :]
    label_w, label_h = _ink_size_em(labels)
    style = {"fontsize": fontsize, "color": "k", "gid": _OBSTACLE_GID}
    below = -(_LABEL_GAP + label_h + _TITLE_GAP)  # title out_offset under the tick labels

    rmsd_ticks = [
        text_polar(ax, 0.0, x, label, va="top", out_offset=-_LABEL_GAP, **style)
        for (x, _d), label in zip(crossings, rmsd_labels)
    ]
    rmsd_title, msd_title = _axis_titles(normalized, units, patternonly)
    titles = [text_polar(ax, 0.0, r_max / 2, rmsd_title, va="top", out_offset=below, **style)]

    if quadrants == 1:
        msd = [
            text_polar(ax, np.pi / 2, rho, label, ha="right", cw_offset=-_LABEL_GAP, **style)
            for rho, label in zip(msd_ticks, msd_labels)
        ]
        # rotated 90 deg: the title's top faces away from the axis
        left = _LABEL_GAP + label_w + _TITLE_GAP
        titles.append(
            text_polar(
                ax,
                np.pi / 2,
                r_max / 2,
                msd_title,
                rotation=90,
                va="bottom",
                out_offset=left,
                **style,
            )
        )
    else:
        msd = [
            text_polar(ax, np.pi, rho, label, va="top", out_offset=-_LABEL_GAP, **style)
            for rho, label in zip(msd_ticks, msd_labels)
        ]
        # centered under its half of the axis, or farther left to end _TITLES_GAP before the
        # horizontal axis title, which can reach past the origin
        em = fontsize / pt_per_data  # data units per fontsize
        rmsd_left = r_max / 2 - _ink_size_em([rmsd_title])[0] * em / 2
        msd_center = max(
            r_max / 2, _ink_size_em([msd_title])[0] * em / 2 - rmsd_left + _TITLES_GAP * em
        )
        titles.append(
            text_polar(ax, np.pi, msd_center, msd_title, va="top", out_offset=below, **style)
        )
    return {"rmsd_tick_labels": rmsd_ticks, "msd_tick_labels": msd, "axis_titles": titles}


def _draw_cc_labels(ax, quadrants, r_max, fontsize):
    """CC values along the outer arc, reading outward (upright), except 1 (it would crowd the
    horizontal axis's last tick label), and a horizontal _CC_TITLE
    just outside them, between the 0.6 and 0.7 labels (1 quadrant) or 0.4 and 0.5 (2
    quadrants). Returns them, grouped (see make_fig_axes())."""
    values = (0.0, *_CC_TICKS)
    if quadrants == 2:
        values += tuple(-c for c in _CC_TICKS) + (-1.0,)
    labels = [f"{c:g}" for c in values]
    style = {"fontsize": fontsize, "color": "k", "gid": _OBSTACLE_GID}
    ticks = []
    for c, label in zip(values, labels):
        theta = np.arccos(c)
        deg = np.degrees(theta)
        if deg <= 90:
            ticks.append(
                text_polar(
                    ax, theta, r_max, label, rotation=deg, ha="left", cw_offset=_LABEL_GAP, **style
                )
            )
        else:  # turned half a turn to stay upright, so it ends at the arc
            ticks.append(
                text_polar(
                    ax,
                    theta,
                    r_max,
                    label,
                    rotation=deg - 180,
                    ha="right",
                    cw_offset=-_LABEL_GAP,
                    **style,
                )
            )
    low, high = (0.6, 0.7) if quadrants == 1 else (0.4, 0.5)
    theta = np.arccos((low + high) / 2)
    # center distance beyond the arc: tick labels, a gap, then half the title along the radial
    cc_w, cc_h = _ink_size_em([_CC_TITLE])
    reach = (
        2 * _LABEL_GAP
        + _ink_size_em([f"{low:g}", f"{high:g}"])[0]
        + (cc_w * np.cos(theta) + cc_h * np.sin(theta)) / 2
    )
    title = text_polar(
        ax,
        theta,
        r_max,
        _CC_TITLE,
        cw_offset=reach * np.cos(theta),
        out_offset=reach * np.sin(theta),
        **style,
    )
    return {"cc_tick_labels": ticks, "cc_title": title}
