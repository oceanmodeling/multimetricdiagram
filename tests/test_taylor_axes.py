import numpy as np
import pytest
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from multimetricdiagram._text import _PolarTextPatch
from multimetricdiagram.alt import taylor
from multimetricdiagram.alt.taylor.axes import _OPTIONS_ATTR

UNNORMALIZED = {"normalized": False, "units": "m/s", "osd": 2.0, "r_max": 5.0}


def _patches(ax, keep):
    """(text, theta, r) of the text_polar() patches for which keep(patch) is true; text is
    matched against the candidate strings 0..9 with 0 or 2 decimals, else None."""
    candidates = [f"{v:.2f}" for v in np.arange(0, 10, 0.25)] + [str(v) for v in range(10)]
    refs = {
        s: TextPath((0, 0), s, size=1, prop=FontProperties(weight="normal")) for s in candidates
    }
    found = []
    for p in ax.patches:
        if isinstance(p, _PolarTextPatch) and keep(p):
            v = p.get_path().vertices
            text = next((s for s, ref in refs.items() if np.array_equal(ref.vertices, v)), None)
            found.append((text, round(float(p._theta), 9), round(float(p._r), 9)))
    return sorted(found, key=lambda t: (t[1], t[2]))


def _below_axis(p):
    return p._va == "top" and p._out_offset < 0 and p._rotation == 0


def _left_of_axis(p):
    return p._ha == "right" and p._cw_offset < 0 and p._rotation == 0


@pytest.mark.parametrize("quadrants", [1, 2])
@pytest.mark.parametrize("options", [{}, UNNORMALIZED], ids=["normalized", "unnormalized"])
def test_axes_limits_layout_and_stored_options(quadrants, options):
    fig, ax, ax_legend = taylor.make_fig_axes(quadrants=quadrants, **options)
    assert np.rad2deg(ax.get_xlim()) == pytest.approx((0, 90 * quadrants))
    star_r = options.get("osd", 1.0)
    assert ax.get_ylim() == (0, options.get("r_max", 1.5))
    assert getattr(ax, _OPTIONS_ATTR) == {
        "quadrants": quadrants,
        "normalized": not options,
        "osd": options.get("osd"),
        "star_r": star_r,
        "r_max": options.get("r_max", 1.5),
        "patternonly": False,
    }
    assert tuple(fig.get_size_inches()) == (8.5, 11)
    fig.canvas.draw()
    polar, legend = ax.get_window_extent(), ax_legend.get_window_extent()
    star = ax.transData.transform((0, star_r))
    assert legend.y0 > ax.transData.transform((np.pi / 2, options.get("r_max", 1.5)))[1]
    assert polar.x0 < star[0] < polar.x1  # the star is in the diagram
    # the diagram fills the polar axes' width
    right_end = ax.transData.transform((0, options.get("r_max", 1.5)))[0]
    assert right_end == pytest.approx(polar.x1, abs=1)


def test_unknown_keyword_raises_listing_valid_names():
    with pytest.raises(TypeError, match=r"'bogus'.*valid names: figsize_inch.*quadrants"):
        taylor.make_fig_axes(bogus=1)


def test_unknown_grid_info_key_raises_listing_valid_keys():
    with pytest.raises(ValueError, match=r"'bogus_lw'.*valid keys: cc_radials_lw"):
        taylor.make_fig_axes(grid_info={"bogus_lw": 1})


BAD_OPTIONS = {
    "quadrants 3": ({"quadrants": 3}, r"quadrants must be None \(automatic\), 1 or 2"),
    "quadrants True": ({"quadrants": True}, r"quadrants must be None \(automatic\), 1 or 2"),
    "metrics not list": ({"metrics": {"CC": 0.5}}, r"metrics must be a list"),
    "metrics no CC": ({"metrics": [{"MSD": 1.0}]}, r"metrics must be a list of .* \(with 'CC'\)"),
    "normalized str": ({"normalized": "yes"}, r"normalized must be True or False"),
    "no units": ({**UNNORMALIZED, "units": None}, r"normalized=False needs units"),
    "no osd": ({**UNNORMALIZED, "osd": None}, r"normalized=False needs osd"),
    "no r_max": ({**UNNORMALIZED, "r_max": None}, r"normalized=False needs r_max"),
    "none": ({"normalized": False}, r"normalized=False needs units, osd, r_max"),
    "units normalized": ({"units": "m"}, r"units apply only with normalized=False"),
    "osd normalized": ({"osd": 2.0}, r"osd apply only with normalized=False"),
    "units not str": ({**UNNORMALIZED, "units": 3}, r"units must be a str"),
    "osd 0": ({**UNNORMALIZED, "osd": 0.0}, r"osd must be a finite number > 0"),
    "osd nan": ({**UNNORMALIZED, "osd": np.nan}, r"osd must be a finite number > 0"),
    "r_max str": ({"r_max": "2"}, r"r_max must be a finite number > 0"),
    "r_max inside star": ({"r_max": 0.9}, r"r_max=0.9 must be beyond the star"),
    "r_max at osd": ({**UNNORMALIZED, "r_max": 2.0}, r"must be beyond the star \(radius 2.0\)"),
    "no room": ({"figsize_inch": (2, 3)}, r"leave too little room for: polar axes"),
}


@pytest.mark.parametrize("options, match", list(BAD_OPTIONS.values()), ids=list(BAD_OPTIONS))
def test_invalid_options_raise(options, match):
    with pytest.raises(ValueError, match=match):
        taylor.make_fig_axes(**options)


# (quadrants option, CC values of the metrics passed or None) -> quadrants drawn
QUADRANT_CASES = {
    "default, no metrics": (None, None, 1),
    "default, all CC >= 0": (None, [0.9, 0.0, 0.4], 1),
    "default, a CC < 0": (None, [0.9, -0.2], 2),
    "override 1 with a CC < 0": (1, [0.9, -0.2], 1),
    "override 2 with all CC >= 0": (2, [0.9, 0.5], 2),
    "override 2, no metrics": (2, None, 2),
}


@pytest.mark.parametrize(
    "quadrants, ccs, expected", list(QUADRANT_CASES.values()), ids=list(QUADRANT_CASES)
)
def test_quadrants_automatic_from_metrics_or_overridden(quadrants, ccs, expected):
    metrics = None if ccs is None else [{"CC": cc, "MSDonOSD": 1.0} for cc in ccs]
    _fig, ax, _l = taylor.make_fig_axes(quadrants=quadrants, metrics=metrics)
    assert np.rad2deg(ax.get_xlim()[1]) == pytest.approx(90 * expected)
    assert getattr(ax, _OPTIONS_ATTR)["quadrants"] == expected


def _families(ax):
    return {z: [ln for ln in ax.lines if ln.get_zorder() == z] for z in (1.1, 1.2, 1.3)}


@pytest.mark.parametrize("quadrants", [1, 2])
@pytest.mark.parametrize("options", [{}, UNNORMALIZED], ids=["normalized", "unnormalized"])
def test_three_gridline_families_with_different_linestyles(quadrants, options):
    _fig, ax, _l = taylor.make_fig_axes(quadrants=quadrants, **options)
    star_r = options.get("osd", 1.0)
    families = _families(ax)
    styles = {z: {ln.get_linestyle() for ln in lines} for z, lines in families.items()}
    assert all(len(s) == 1 for s in styles.values())
    assert len(set.union(*styles.values())) == 3
    for ln in families[1.1]:  # constant CC: radial lines from the origin
        theta, r = ln.get_data()
        assert theta[0] == theta[1] and r[0] == 0
    cc = sorted(np.round(np.cos([ln.get_xdata()[0] for ln in families[1.1]]), 9))
    ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    assert cc == sorted(ticks + ([-t for t in ticks] if quadrants == 2 else []))
    for ln in families[1.2]:  # constant MSD: circles about the origin
        assert np.ptp(ln.get_ydata()) == 0
    for ln in families[1.3]:  # constant RMSD: circles about the star
        theta, r = ln.get_data()
        d = np.hypot(r * np.cos(theta) - star_r, r * np.sin(theta))
        assert np.ptp(d) < 1e-12


# (options, expected right-axis labels (text, x), expected MSD-axis labels (text, r))
TICK_CASES = {
    "normalized": (
        {},
        [("0.75", 0.25), ("0.50", 0.5), ("0.25", 0.75), ("0.25", 1.25), ("0.50", 1.5)],
        [(f"{v:.2f}", v) for v in np.arange(0, 1.51, 0.25)],
    ),
    "unnormalized": (
        UNNORMALIZED,
        [("1", 1.0), ("1", 3.0), ("2", 4.0), ("3", 5.0)],  # the 2 circle crosses at the origin
        [(str(v), float(v)) for v in range(6)],
    ),
}


@pytest.mark.parametrize("quadrants", [1, 2])
@pytest.mark.parametrize("options, right, msd", list(TICK_CASES.values()), ids=list(TICK_CASES))
def test_tick_labels_at_circle_crossings(quadrants, options, right, msd):
    _fig, ax, _l = taylor.make_fig_axes(quadrants=quadrants, **options)
    star_r = options.get("osd", 1.0)
    on_right = _patches(ax, lambda p: _below_axis(p) and p._theta == 0)
    assert [(t, r) for t, _theta, r in on_right if t is not None] == right
    for text, _theta, x in on_right:
        if text is not None:  # on the star circle of radius text, crossing on either side
            assert abs(x - star_r) == pytest.approx(float(text))
    if quadrants == 1:
        on_msd = _patches(ax, lambda p: _left_of_axis(p) and p._theta == np.pi / 2)
    else:
        on_msd = _patches(ax, lambda p: _below_axis(p) and p._theta == np.pi)
    assert [(t, r) for t, _theta, r in on_msd if t is not None] == msd
    radii = {round(float(ln.get_ydata()[0]), 9) for ln in _families(ax)[1.2]}
    for text, _theta, r in on_msd:
        if text is not None:  # a circle, the origin or the arc
            assert r in radii | {0.0, options.get("r_max", 1.5)}


def _text_patches(ax, text):
    ref = TextPath((0, 0), text, size=1, prop=FontProperties(weight="normal")).vertices
    return [
        p
        for p in ax.patches
        if isinstance(p, _PolarTextPatch) and np.array_equal(p.get_path().vertices, ref)
    ]


def _has_text(ax, text):
    return bool(_text_patches(ax, text))


# normalized -> (horizontal axis title, MSD axis title); UNNORMALIZED has units "m/s"
AXIS_TITLES = {
    True: (
        "Solid contours: RMSD/OSD (stick tip), URMSD/OSD (marker)",
        "Dashed contours: MSD/OSD (marker)",
    ),
    False: (
        "Solid contours: RMSD (stick tip), URMSD (marker) [m/s]",
        "Dashed contours: MSD (marker) [m/s]",
    ),
}


@pytest.mark.parametrize("quadrants", [1, 2])
@pytest.mark.parametrize("normalized", [True, False])
def test_axis_texts_exact(quadrants, normalized):
    options = {} if normalized else UNNORMALIZED
    _fig, ax, _l = taylor.make_fig_axes(quadrants=quadrants, **options)
    for text in (*AXIS_TITLES[normalized], "CC (marker)"):
        assert len(_text_patches(ax, text)) == 1, text
    for old in (
        "MSD/OSD (dashed)",
        "RMSD/OSD, URMSD/OSD (solid)",
        "CC",
        "Correlation coefficient",
    ):
        assert not _has_text(ax, old)


@pytest.mark.parametrize("options", [{}, UNNORMALIZED], ids=["normalized", "unnormalized"])
def test_star_at_reference_point(options):
    _fig, ax, _l = taylor.make_fig_axes(**options)
    stars = [ln for ln in ax.lines if ln.get_marker() == "*"]
    assert len(stars) == 1
    assert tuple(stars[0].get_xydata()[0]) == (0, options.get("osd", 1.0))


# --- automatic r_max -----------------------------------------------------------------------

from multimetricdiagram.alt.taylor.axes import _radius_ticks


def _msd_metrics(msds, osd=1.0):
    return [{"CC": 0.9, "MSD": m, "MSDonOSD": m / osd} for m in msds]


@pytest.mark.parametrize(
    "msds, expected",
    [
        ([0.5, 1.2], 1.5),  # target 1.5 (the star's 1.5 x 1)
        ([0.5, 1.58], 2.0),  # target 1.659: ticks 0, 0.5, ..., 2
        ([3.2], 4.0),  # target 3.36: ticks 0, 1, ..., 4 (not 3.5)
        ([4.3], 5.0),  # target 4.515
        ([1.5 / 1.05], 1.5),  # target exactly 1.5
        ([2.0 / 1.05], 2.0),  # target exactly the tick 2.0: not rounded up
    ],
    ids=["1.2", "1.58", "3.2", "4.3", "at-1.5", "at-2.0"],
)
def test_automatic_r_max_normalized(msds, expected):
    _fig, ax, _l = taylor.make_fig_axes(metrics=_msd_metrics(msds))
    r_max = ax.get_ylim()[1]
    assert r_max == expected
    assert np.isclose(_radius_ticks(r_max), r_max).any()  # the outer arc is labeled


@pytest.mark.parametrize("msds, osd", [([1.0, 4.2], 2.0), ([0.1, 0.3], 0.25), ([30.0], 7.0)])
def test_automatic_r_max_unnormalized_is_a_labeled_tick(msds, osd):
    _fig, ax, _l = taylor.make_fig_axes(
        normalized=False, units="u", osd=osd, metrics=_msd_metrics(msds, osd)
    )
    r_max = ax.get_ylim()[1]
    assert r_max >= 1.5 * osd and r_max >= 1.05 * max(msds)
    assert np.isclose(_radius_ticks(r_max), r_max).any()  # the outer arc is labeled
    assert r_max <= max(1.5 * osd, 1.05 * max(msds)) * 2.5  # the first such tick, not far above


@pytest.mark.parametrize("options", [{}, UNNORMALIZED], ids=["normalized", "unnormalized"])
def test_explicit_r_max_wins_over_metrics(options):
    r_max = options.get("r_max", 1.5) * 2
    _fig, ax, _l = taylor.make_fig_axes(**{**options, "r_max": r_max}, metrics=_msd_metrics([0.5]))
    assert ax.get_ylim()[1] == r_max


def test_unnormalized_without_r_max_or_metrics_raises():
    with pytest.raises(ValueError, match=r"needs r_max \(or metrics=, to choose it\)"):
        taylor.make_fig_axes(normalized=False, units="u", osd=2.0)
    with pytest.raises(ValueError, match=r"with 'MSD'\) to choose r_max"):
        taylor.make_fig_axes(normalized=False, units="u", osd=2.0, metrics=[{"CC": 0.5}])
