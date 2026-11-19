"""Multimetric diagram figure and axes: make_fig_axes()."""

from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.markers import MarkerStyle
from matplotlib.ticker import FuncFormatter
from matplotlib.transforms import Affine2D

from ._text import text_polar
from .labels import _OBSTACLE_GID
from .markerstick import _marker_centroid


def make_fig_axes(*, return_artists: bool = False, **overrides) -> tuple[Any, ...]:
    """
    Create the multimetric diagram figure and its polar and legend axes.

    The polar axes is annotated, with markerstick keys at upper left and upper right; below it
    is an MSD/OSD grayscale bar key (mirrored gray shading); above it is an empty legend axes
    for add_labeledmarker_legend() or matplotlib's ax.legend() / fig.legend().

    Args:
        return_artists: False (default): return (fig, ax_polar, ax_legend). True: also return
            a dict of the background artists, grouped, to restyle or remove them (e.g.
            thinner lines for a smaller figure). Keys, in drawing order, each an artist or a
            list of artists:
            - 'circular_gridlines': the circles at the r ticks (Line2Ds).
            - 'dashed_radials': the dashed radials at 22.5 and 157.5 deg (Line2Ds).
            - 'vertical_line': the MSD = OSD line at 90 deg and its continuation below the
              origin (Line2Ds).
            - 'unit_circle': the white r = 1 ring (Line2D).
            - 'star': the origin star (Line2D).
            - 'sector_text': 'MSD < OSD', 'MSD > OSD' and the RMSD/SDQ, URMSD/SDQ label
              (Texts).
            - 'cc_caption': the CC caption's lines (text_polar() patches).
            - 'shading': the MSD/OSD gray shading (QuadMesh).
            - 'cc_contours': the +0.1 to +0.9, -0.1 to -0.9 and +0.98 CC contours
              (ContourSets).
            - 'markerstick_keys': both keys' markers, sticks and text, right key first.
            - 'grayscale_bar_ax': the grayscale bar key's Axes.
            - 'grayscale_bar': its gray bins (Rectangles), MSD/OSD title (Text) and center line
              (Line2D).
            Not included: the axes' own spines and patches, and matplotlib's tick labels (the
            r tick labels, and the grayscale bar's MSD/OSD values).
        **overrides: Optional settings replacing the defaults in the function body (config):
            figsize_inch, margins_t_b_l_r_inch, vgap_legend_rel_polar_inch, fontsize, and the
            dicts grid_info and vdcc_info, whose keys are merged into grid_info_defaults and
            vdcc_info_defaults. E.g. grid_info 'unitcircle_lw' sets the white r = 1 ring;
            vdcc_info 'cc_lw' the CC contours and 'cc098_lw' (default 1.5) the dotted CC = 0.98
            contour.

    Returns:
        (fig, ax_polar, ax_legend): the Figure, the diagram's polar axes, and the legend axes;
        with return_artists=True, (fig, ax_polar, ax_legend, artists), artists the dict above.

    Raises:
        TypeError: an unknown keyword (the message lists the valid names).
        ValueError: an unknown grid_info or vdcc_info key (the message lists the valid keys);
            or the polar, legend or grayscale bar axes would get a width or height <= 0 from
            figsize_inch, margins_t_b_l_r_inch and the vdcc_info sizes and pads (the message
            names each such axes and its size in inches).

    Notes:
        The gray shading and grayscale bar show MSD/OSD. The shading is computed as
        |VD| = 1 - VS, where VS = 2*MSD*OSD/(MSD^2 + OSD^2) (_vd(); vd_steps are its bin
        edges). The grayscale bar labels each bin edge with the equivalent MSD/OSD ratio k,
        from VS = 2k/(1 + k^2): k <= 1 on the left half, k >= 1 on the right. Setting names
        containing 'vd' refer to |VD|.

        Shading, contours, gridlines and reference lines use zorder < 2 so markersticks
        (default zorder 2) draw on top. Levels: 0 axes, MSD/OSD shading; 1 CC contours,
        grayscale bar key rectangles; 1.3 circular gridlines (at the r ticks); 1.5 origin star;
        1.6 white unit circle; 1.7 dashed radials and vertical line; 2 markerstick keys, CC
        caption; 3 sector labels. The circular gridlines are drawn at the r ticks present when
        the figure is made and do not follow later changes to the r limits or ticks.
    """

    config = {
        "figsize_inch": (8.5, 11),
        "margins_t_b_l_r_inch": (1, 1, 0.5, 0.5),
        "vgap_legend_rel_polar_inch": 0,
        "fontsize": 10,
        "grid_info": {},
        "vdcc_info": {},
    }
    grid_info_defaults = {
        "circlegrids_lw": 0.4,
        "circlegrids_zo": 1.3,
        "radials_vsccequal_lw": 0.6,
        "radials_vsccequal_zo": 1.7,
        "vertical_msdosdequal_lw": 1,
        "vertical_msdosdequal_zo": 1.7,
        "unitcircle_lw": 1.0,
        "unitcircle_zo": 1.6,
    }
    vdcc_info_defaults = {
        "vdcc_bar_height_inch": 0.25,
        "vdcc_bar_width_frac": 0.80,
        "vdcc_pad_t_b_l_r_inch": (0.3, 0.7, 0.25, 0.25),
        "vdcc_n_theta": 540,
        "vdcc_n_r": 400,
        "vd_steps": np.linspace(0, 1, 11) ** 2,  # |VD| bin edges, 0 to 1, quadratic spacing
        "vd_gray_min": 0.5,
        "vd_gray_max": 0.95,
        "cc_pos": np.arange(0.1, 1, 0.1),
        "cc_neg": np.arange(-0.9, 0, 0.1),
        "cc_lw": 0.48,
        "cc098_lw": 1.5,  # dotted CC = 0.98 contour
        "cc_zo": 1,
        "cc_negcorr_dashpattern": (0, (10, 5)),
    }
    unknown = [k for k in overrides if k not in config]
    if unknown:
        raise TypeError(
            f"make_fig_axes() got unexpected keyword argument(s) "
            f"{', '.join(map(repr, unknown))}; valid names: {', '.join(config)}"
        )
    config.update(overrides)
    for name, defaults in (("grid_info", grid_info_defaults), ("vdcc_info", vdcc_info_defaults)):
        unknown = [k for k in config[name] if k not in defaults]
        if unknown:
            raise ValueError(
                f"unknown {name} key(s) {', '.join(map(repr, unknown))}; "
                f"valid keys: {', '.join(defaults)}"
            )
    figsize_inch = config["figsize_inch"]
    margins_t_b_l_r_inch = config["margins_t_b_l_r_inch"]
    vgap_legend_rel_polar_inch = config["vgap_legend_rel_polar_inch"]
    fontsize = config["fontsize"]
    grid_info = {**grid_info_defaults, **config["grid_info"]}
    vdcc_info = {**vdcc_info_defaults, **config["vdcc_info"]}

    # layout (inches): grayscale bar key at the bottom, polar axes above it, legend axes on top
    wid_inch = figsize_inch[0] - margins_t_b_l_r_inch[2] - margins_t_b_l_r_inch[3]
    vdcc_pad_top_inch, vdcc_pad_bottom_inch, _, _ = vdcc_info["vdcc_pad_t_b_l_r_inch"]

    axes_height_polar_inch = wid_inch
    # gap between the square axes box and the -45..225 deg wedge ("pacman"), above and below it
    y_adjust_inch = 0.5 * (wid_inch / 2) * (1 - np.sqrt(2) / 2)
    vdcc_bar_height_inch = vdcc_info["vdcc_bar_height_inch"]
    vdcc_bar_width_inch = vdcc_info["vdcc_bar_width_frac"] * wid_inch
    vdcc_box_bottom_inch = margins_t_b_l_r_inch[1]
    polar_visible_bottom_inch = (
        vdcc_box_bottom_inch + vdcc_pad_bottom_inch + vdcc_bar_height_inch + vdcc_pad_top_inch
    )
    polar_axes_bottom_inch = polar_visible_bottom_inch - y_adjust_inch
    polar_visible_top_inch = polar_visible_bottom_inch + axes_height_polar_inch - 2 * y_adjust_inch
    legend_bottom_inch = polar_visible_top_inch + vgap_legend_rel_polar_inch
    axes_height_legend_inch = figsize_inch[1] - legend_bottom_inch - margins_t_b_l_r_inch[0]
    _check_axes_sizes(
        figsize_inch,
        margins_t_b_l_r_inch,
        {
            "polar axes": (wid_inch, axes_height_polar_inch),
            "legend axes": (wid_inch, axes_height_legend_inch),
            "grayscale bar axes": (vdcc_bar_width_inch, vdcc_bar_height_inch),
        },
    )

    fig = plt.figure(figsize=figsize_inch)
    pos = (
        margins_t_b_l_r_inch[2] / figsize_inch[0],
        polar_axes_bottom_inch / figsize_inch[1],
        wid_inch / figsize_inch[0],
        axes_height_polar_inch / figsize_inch[1],
    )
    ax_polar = fig.add_axes(pos, projection='polar', zorder=0, clip_on=False)
    ax_polar.set_thetamin(-45.0)  # type: ignore
    ax_polar.set_thetamax(225.0)  # type: ignore

    pos = (
        margins_t_b_l_r_inch[2] / figsize_inch[0],
        legend_bottom_inch / figsize_inch[1],
        wid_inch / figsize_inch[0],
        axes_height_legend_inch / figsize_inch[1],
    )
    ax_legend = fig.add_axes(pos, zorder=0)
    ax_legend.axis('off')

    # drawing sections, in this order (artist creation order fixes the draw order at equal zorder)
    artists = _draw_polar_grid_and_reference_lines(
        ax_polar, fontsize, grid_info, vdcc_info['vdcc_n_r']
    )
    artists.update(_draw_sector_text_and_cc_caption(ax_polar, fontsize))
    gmap, shading_and_contours = _draw_vd_shading_and_cc_contours(ax_polar, vdcc_info)
    artists.update(shading_and_contours)
    artists['markerstick_keys'] = _draw_markerstick_keys(ax_polar, fontsize)

    # MSD/OSD grayscale bar key below the polar axes
    vdcc_x0_inch = margins_t_b_l_r_inch[2] + 0.5 * (wid_inch - vdcc_bar_width_inch)
    vdcc_extra_lift_inch = 0.25
    vdcc_y0_inch = vdcc_box_bottom_inch + vdcc_pad_bottom_inch + vdcc_extra_lift_inch
    vdcc_rect = (
        vdcc_x0_inch / figsize_inch[0],
        vdcc_y0_inch / figsize_inch[1],
        vdcc_bar_width_inch / figsize_inch[0],
        vdcc_bar_height_inch / figsize_inch[1],
    )
    artists.update(
        _draw_msdosd_grayscale_bar(fig, vdcc_rect, gmap, fontsize, grid_info, vdcc_info)
    )

    if return_artists:
        return fig, ax_polar, ax_legend, artists
    return fig, ax_polar, ax_legend


def _check_axes_sizes(
    figsize_inch,
    margins_t_b_l_r_inch,
    sizes_inch,
    also_take_room="the vdcc_info sizes and pads and vgap_legend_rel_polar_inch also take height",
):
    """Raise ValueError naming each axes whose (width, height) in inches is not positive;
    also_take_room ends the message, naming the other settings that use up the page."""
    too_small = [
        f"{name} {w:.3g} x {h:.3g} in"
        for name, (w, h) in sizes_inch.items()
        if not (w > 0 and h > 0)
    ]
    if too_small:
        raise ValueError(
            f"figsize_inch={tuple(figsize_inch)} and margins_t_b_l_r_inch="
            f"{tuple(margins_t_b_l_r_inch)} leave too little room for: {'; '.join(too_small)} "
            f"(width x height must be > 0; {also_take_room})"
        )


_R_MAX = np.sqrt(2)  # outer radius of the diagram


def _vd(theta_deg, r):
    """|VD| = 1 - VS at diagram position (theta in degrees, r = URMSD/SDQ)."""
    # singular on r = 1 (CC = 0 there, so VS is not encoded); the white unit circle covers it
    alpha = np.abs((theta_deg - 90.0) / 135.0)
    base = np.abs(1 - r**2)
    return 1 - base**alpha


def _cc(theta_deg, r):
    """CC at diagram position (theta in degrees, r = URMSD/SDQ)."""
    base = np.abs(1 - r**2)
    alpha = np.where(theta_deg <= 90.0, (theta_deg + 45.0) / 135.0, (225.0 - theta_deg) / 135.0)
    val = base**alpha
    return np.where(r <= 1, val, -val)


def _fmt_tick_k(k):
    """Grayscale bar tick label for MSD/OSD ratio k: '∞', or 3 - (integer digits) decimals,
    min 0."""
    if np.isinf(k):
        return "∞"

    def decimals_for(x):
        return max(0, 3 - max(1, len(str(int(abs(x))))))

    # count integer digits after rounding, which can add one (9.996 -> 10.0, not 10.00)
    decimals = decimals_for(round(float(k), decimals_for(k)))
    return f"{k:.{decimals}f}"


def _draw_polar_grid_and_reference_lines(ax_polar, fontsize, grid_info, n_unit_circle_pts):
    """Axes limits, r tick labels, circular gridlines, reference lines, unit circle and origin
    star; returns the drawn lines, grouped (see make_fig_axes() return_artists)."""
    r_max = _R_MAX
    ax_polar.set_clip_on(False)
    ax_polar.set_ylim(0, r_max)
    ax_polar.set_xticklabels([])
    ax_polar.set_rlabel_position(0)  # type: ignore
    ax_polar.tick_params(axis="y", pad=-1, labelsize=fontsize)
    ax_polar.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: "" if np.isclose(value, 0.0) or value > 1.3 else f"{value:.1f}"
        )
    )

    ax_polar.grid(False)  # matplotlib's gridlines ignore their zorder; circles drawn below
    ax_polar.set_facecolor('none')
    ax_polar.patch.set_edgecolor('none')

    # circular gridlines at the r ticks, as separate lines so 'circlegrids_zo' applies
    theta_circle = np.linspace(-np.pi / 4, (5 / 4) * np.pi, n_unit_circle_pts)
    circles = []
    for r_grid in ax_polar.get_yticks():
        if 0 < r_grid <= r_max:
            circles += ax_polar.plot(
                theta_circle,
                np.full(n_unit_circle_pts, r_grid),
                color="black",
                linewidth=grid_info["circlegrids_lw"],
                zorder=grid_info["circlegrids_zo"],
            )

    # dashed radials at 22.5° and 157.5°: VS and CC contribute equally to
    # (URMSD/SDQ)² = 1 - VS*CC
    radials = []
    for angle in [22.5, 157.5]:
        radials += ax_polar.plot(
            np.deg2rad([angle, angle]),
            [0, r_max],
            color="black",
            linestyle="--",
            linewidth=grid_info["radials_vsccequal_lw"],
            zorder=grid_info["radials_vsccequal_zo"],
        )

    # vertical line at 90°: MSD == OSD
    vertical = ax_polar.plot(
        np.deg2rad([90, 90]),
        [0, r_max],
        color="black",
        linestyle="-",
        linewidth=grid_info["vertical_msdosdequal_lw"],
        zorder=grid_info["vertical_msdosdequal_zo"],
    )

    # its continuation below the origin, at 270°
    vertical += ax_polar.plot(
        np.deg2rad([270, 270]),
        [0, 0.45],
        color="black",
        linestyle="-",
        linewidth=grid_info["vertical_msdosdequal_lw"],
        zorder=grid_info["vertical_msdosdequal_zo"],
        clip_on=False,
    )

    # white unit circle (r = 1): the CC = 0 contour ("solid thick 0.0" in the caption)
    (unit_circle,) = ax_polar.plot(
        theta_circle,
        np.ones(n_unit_circle_pts),
        color="white",
        linewidth=grid_info["unitcircle_lw"],
        zorder=grid_info["unitcircle_zo"],
        clip_on=False,
    )

    # star at the origin: perfect agreement
    (star,) = ax_polar.plot(
        0,
        0,
        marker="*",
        markersize=14,
        color="black",
        markeredgecolor="black",
        clip_on=False,
        zorder=1.5,
        gid=_OBSTACLE_GID,
    )
    return {
        "circular_gridlines": circles,
        "dashed_radials": radials,
        "vertical_line": vertical,
        "unit_circle": unit_circle,
        "star": star,
    }


def _draw_sector_text_and_cc_caption(ax_polar, fontsize):
    """MSD < OSD / MSD > OSD and RMSD/SDQ labels in the open sector; the CC-contour caption.
    Returns them, grouped (see make_fig_axes() return_artists)."""
    delta = 5
    msdosd_label_r = 0.3575
    sector_text = [
        ax_polar.text(
            np.deg2rad(270 - delta),
            msdosd_label_r,
            "MSD\n<\nOSD",
            va="center",
            ha="right",
            ma="center",
            clip_on=False,
            gid=_OBSTACLE_GID,
            fontsize=fontsize,
        ),
        ax_polar.text(
            np.deg2rad(270 + delta),
            msdosd_label_r,
            "MSD\n>\nOSD",
            va="center",
            ha="left",
            ma="center",
            clip_on=False,
            gid=_OBSTACLE_GID,
            fontsize=fontsize,
        ),
        ax_polar.text(
            np.deg2rad(-57.4),
            1.05,
            "RMSD/SDQ,\nURMSD/SDQ",
            va="center",
            ha="center",
            clip_on=False,
            gid=_OBSTACLE_GID,
            fontsize=fontsize,
        ),
    ]

    in_to_cw = 72.0 / fontsize
    cc_caption_texts = [
        "CC: white contours",
        "Inner to outer:",
        "dotted +0.98",
        "solid +0.9 to +0.1 by -0.1",
        "solid thick 0.0",
        "dashed -0.1 to -0.9 by -0.1",
    ]
    cc_caption_dr = 0.09
    cc_caption_r0 = 0.75
    cc_caption_r = [cc_caption_r0 + i * cc_caption_dr for i in range(len(cc_caption_texts))]
    cc_caption_indent_in = [0.22, 0.47, 0.72, 0.72, 0.72, 0.72]
    # anchors on the 225° radial; lines 4-6 move right to line 3's x so lines 3-6 align
    theta_225 = np.deg2rad(225)
    x_line3 = cc_caption_r[2] * np.cos(theta_225)
    caption = []
    for i, (line, r, indent_in) in enumerate(
        zip(cc_caption_texts, cc_caption_r, cc_caption_indent_in)
    ):
        theta = theta_225
        if i >= 3:
            x, y = x_line3, r * np.sin(theta_225)
            theta, r = np.arctan2(y, x), np.hypot(x, y)
        caption.append(
            text_polar(
                ax_polar,
                theta,
                r,
                line,
                fontsize=fontsize,
                ha="left",
                va="center",
                cw_offset=indent_in * in_to_cw,
                color="k",
                gid=_OBSTACLE_GID,
            )
        )
    return {"sector_text": sector_text, "cc_caption": caption}


def _draw_vd_shading_and_cc_contours(ax_polar, vdcc_info):
    """MSD/OSD gray shading and white CC contours; returns (the gray colormap, for the
    grayscale bar; the shading and contours, grouped as in make_fig_axes() return_artists)."""
    # shading shows MSD/OSD, computed as |VD| = 1 - VS (see make_fig_axes() Notes)
    gray_steps = np.linspace(
        vdcc_info['vd_gray_max'], vdcc_info['vd_gray_min'], len(vdcc_info['vd_steps']) - 1
    )
    gmap = ListedColormap([str(g) for g in gray_steps])
    norm = BoundaryNorm(vdcc_info['vd_steps'], len(cast(Any, gmap).colors))
    theta = np.linspace(-45, 225, vdcc_info['vdcc_n_theta'])
    r = np.linspace(0, _R_MAX, vdcc_info['vdcc_n_r'])
    theta_grid, r_grid = np.meshgrid(theta, r)
    vd_vals = _vd(theta_grid, r_grid)
    shading = ax_polar.pcolormesh(
        np.deg2rad(theta_grid), r_grid, vd_vals, cmap=gmap, norm=norm, shading="auto", zorder=0
    )

    cc_vals = _cc(theta_grid, r_grid)
    cs_pos = ax_polar.contour(
        np.deg2rad(theta_grid),
        r_grid,
        cc_vals,
        levels=vdcc_info['cc_pos'],
        colors="white",
        linewidths=vdcc_info["cc_lw"],
        linestyles="-",
        zorder=vdcc_info["cc_zo"],
    )
    cs_neg = ax_polar.contour(
        np.deg2rad(theta_grid),
        r_grid,
        cc_vals,
        levels=vdcc_info['cc_neg'],
        colors="white",
        linewidths=vdcc_info["cc_lw"],
        linestyles="--",
        zorder=vdcc_info["cc_zo"],
    )
    cs_neg.set_linestyle(vdcc_info['cc_negcorr_dashpattern'])
    cs_098 = ax_polar.contour(
        np.deg2rad(theta_grid),
        r_grid,
        cc_vals,
        levels=[0.98],
        colors="white",
        linewidths=vdcc_info["cc098_lw"],
        linestyles=":",
        zorder=vdcc_info["cc_zo"],
    )
    return gmap, {"shading": shading, "cc_contours": [cs_pos, cs_neg, cs_098]}


def _draw_markerstick_keys(ax_polar, fontsize):
    """The two markerstick keys (MB/SDQ > 0 at upper right, < 0 at upper left); returns their
    artists (markers, stick and text of the right key, then the left)."""
    r = 1.51
    th_mrkrs = [60, 120]
    lngs = [0.58, -0.58]
    mbs_lbls = ['>0', '<0']
    drawn = []
    for th_mrkr, lng, mbs_lbl in zip(th_mrkrs, lngs, mbs_lbls):
        tri_rotation = Affine2D().rotate_deg(th_mrkr - 90)
        tri_base_path = MarkerStyle('^').get_path().transformed(MarkerStyle('^').get_transform())
        tri_centroid = _marker_centroid(tri_base_path)
        tri_centered_rotation = Affine2D().translate(*(-tri_centroid)).rotate_deg(th_mrkr - 90)
        for mrkr_shape, mrkr_size in [
            (MarkerStyle('o', transform=tri_rotation), 10),
            (MarkerStyle('s', transform=tri_rotation), 8.064),
            (MarkerStyle('^', transform=tri_centered_rotation), 8),
        ]:
            drawn += ax_polar.plot(
                np.deg2rad(th_mrkr),
                r,
                marker=mrkr_shape,
                markersize=mrkr_size,
                markerfacecolor='none',
                markeredgecolor='k',
                gid=_OBSTACLE_GID,
                clip_on=False,
            )
        # clockwise unit tangent at the marker (stick direction for MB/SDQ > 0)
        tx, ty = np.sin(np.deg2rad(th_mrkr)), -np.cos(np.deg2rad(th_mrkr))
        x1, y1 = (
            r * np.cos(np.deg2rad(th_mrkr)) + lng * tx,
            r * np.sin(np.deg2rad(th_mrkr)) + lng * ty,
        )
        drawn += ax_polar.plot(
            [np.deg2rad(th_mrkr), np.arctan2(y1, x1)],
            [r, np.hypot(x1, y1)],
            'k',
            linewidth=1.5,
            clip_on=False,
            gid=_OBSTACLE_GID,
        )
        # labels at stick midpoint (MB/SDQ, >0/<0)
        rotation = -(90 - th_mrkr)
        midx, midy = (
            0.5 * (r * np.cos(np.deg2rad(th_mrkr)) + x1),
            0.5 * (r * np.sin(np.deg2rad(th_mrkr)) + y1),
        )
        thm, rm = np.arctan2(midy, midx), np.hypot(midx, midy)
        drawn.append(
            text_polar(
                ax_polar,
                thm,
                rm,
                'MB/SDQ',
                rotation=rotation,
                fontsize=fontsize,
                va='bottom',
                out_offset=0.3,
                color='k',
                gid=_OBSTACLE_GID,
            )
        )
        drawn.append(
            text_polar(
                ax_polar,
                thm,
                rm,
                mbs_lbl,
                rotation=rotation,
                fontsize=fontsize,
                va='top',
                out_offset=-0.3,
                color='k',
                gid=_OBSTACLE_GID,
            )
        )
        # label at stick-end (RMSD/SDQ)
        theta_end = np.arctan2(y1, x1)
        r_end = np.hypot(x1, y1)
        drawn.append(
            text_polar(
                ax_polar,
                theta_end,
                r_end,
                'RMSD/SDQ',
                rotation=rotation,
                fontsize=fontsize,
                va='top',
                out_offset=-0.5,
                color='k',
                gid=_OBSTACLE_GID,
            )
        )
        # labels at the marker; offset (fontsize units): half the 10-pt circle plus 0.3
        symbol_offset = 10 / 2 / fontsize + 0.3
        if lng > 0:
            drawn.append(
                text_polar(
                    ax_polar,
                    np.deg2rad(th_mrkr),
                    r,
                    'CC',
                    rotation=rotation,
                    fontsize=fontsize,
                    ha='right',
                    cw_offset=-symbol_offset,
                    color='k',
                    gid=_OBSTACLE_GID,
                )
            )
        else:
            drawn.append(
                text_polar(
                    ax_polar,
                    np.deg2rad(th_mrkr),
                    r,
                    'CC',
                    rotation=rotation,
                    fontsize=fontsize,
                    ha='left',
                    cw_offset=symbol_offset,
                    color='k',
                    gid=_OBSTACLE_GID,
                )
            )
        drawn.append(
            text_polar(
                ax_polar,
                np.deg2rad(th_mrkr),
                r,
                'MSD/OSD',
                rotation=rotation,
                fontsize=fontsize,
                va='bottom',
                out_offset=symbol_offset,
                color='k',
                gid=_OBSTACLE_GID,
            )
        )
        drawn.append(
            text_polar(
                ax_polar,
                np.deg2rad(th_mrkr),
                r,
                'URMSD/SDQ',
                rotation=rotation,
                fontsize=fontsize,
                va='top',
                out_offset=-symbol_offset,
                color='k',
                gid=_OBSTACLE_GID,
            )
        )
    return drawn


def _draw_msdosd_grayscale_bar(fig, rect, gmap, fontsize, grid_info, vdcc_info):
    """MSD/OSD grayscale bar key: mirrored gray bins with MSD/OSD tick labels, in its own axes;
    returns the axes and its artists (see make_fig_axes() return_artists)."""
    ax_vdcc = fig.add_axes(rect, zorder=0)
    ax_vdcc.set_yticks([])
    ax_vdcc.set_ylim(0.0, 1.0)
    for spine in ax_vdcc.spines.values():
        spine.set_edgecolor("black")

    # grayscale bar shading: equal-width bins; vd_steps sets only the bin count and tick values
    vd_steps = vdcc_info['vd_steps']
    n_bins = len(vd_steps) - 1
    x_right = 0.5 + 0.5 * np.linspace(0.0, 1.0, n_bins + 1)
    x_left = 0.5 - 0.5 * np.linspace(0.0, 1.0, n_bins + 1)[::-1]
    drawn = []
    for j in range(n_bins):
        drawn.append(
            ax_vdcc.axvspan(
                x_right[j], x_right[j + 1], color=gmap(j / (n_bins - 1)), zorder=1, ec="none"
            )
        )
        drawn.append(
            ax_vdcc.axvspan(
                x_left[j],
                x_left[j + 1],
                color=gmap((n_bins - 1 - j) / (n_bins - 1)),
                zorder=1,
                ec="none",
            )
        )
    ax_vdcc.set_xlim(0.0, 1.0)

    # tick labels: MSD/OSD ratio k at each |VD| step, solving VS = 2k/(1 + k²) (k >= 1 right)
    vs_steps = 1.0 - vd_steps
    with np.errstate(divide='ignore', invalid='ignore'):
        k_right = np.where(vs_steps == 0.0, np.inf, (1.0 + np.sqrt(1.0 - vs_steps**2)) / vs_steps)
    k_left = (1.0 / k_right)[::-1]
    tick_x = np.concatenate((x_left, x_right[1:]))
    tick_k = np.concatenate((k_left, k_right[1:]))

    ax_vdcc.set_xticks(tick_x)
    ax_vdcc.set_xticklabels([_fmt_tick_k(k) for k in tick_k], rotation=45, fontsize=fontsize)
    drawn.append(ax_vdcc.set_title("MSD/OSD", fontsize=fontsize))

    # center line (MSD/OSD = 1, VD = 0)
    drawn.append(
        ax_vdcc.axvline(
            0.5, color="black", linewidth=grid_info["vertical_msdosdequal_lw"], zorder=2
        )
    )
    return {"grayscale_bar_ax": ax_vdcc, "grayscale_bar": drawn}
