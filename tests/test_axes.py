import io

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.projections.polar import PolarAxes

import multimetricdiagram as mmd


def test_make_fig_axes_returns_figure_polar_and_legend_axes():
    fig, ax_polar, ax_legend = mmd.make_fig_axes()
    assert isinstance(fig, Figure)
    assert isinstance(ax_polar, PolarAxes)
    assert not isinstance(ax_legend, PolarAxes)
    assert ax_polar.get_figure() is fig and ax_legend.get_figure() is fig
    assert not ax_legend.axison  # legend axes starts empty, axis off
    assert ax_polar.get_thetamin() == pytest.approx(-45.0)
    assert ax_polar.get_thetamax() == pytest.approx(225.0)
    assert np.rad2deg(ax_polar.get_xlim()) == pytest.approx((-45.0, 225.0))
    assert ax_polar.get_ylim() == pytest.approx((0.0, np.sqrt(2)))


@pytest.fixture(scope="module")
def default_and_overridden():
    """A default diagram and one with layout, grid_info and vdcc_info overrides."""
    default = mmd.make_fig_axes()
    overridden = mmd.make_fig_axes(
        figsize_inch=(7.0, 9.0),
        margins_t_b_l_r_inch=(0.5, 0.75, 0.25, 1.0),
        grid_info={"circlegrids_lw": 1.1},
        vdcc_info={"cc_lw": 2.0},
    )
    return default, overridden


def test_figsize_and_margins_overrides(default_and_overridden):
    _default, (fig, ax_polar, ax_legend) = default_and_overridden
    assert tuple(fig.get_size_inches()) == pytest.approx((7.0, 9.0))
    top, _bottom, left, right = 0.5, 0.75, 0.25, 1.0
    width_in = 7.0 - left - right
    for ax in (ax_polar, ax_legend):
        pos = ax.get_position(original=True)
        assert pos.x0 * 7.0 == pytest.approx(left)
        assert pos.width * 7.0 == pytest.approx(width_in)
    assert ax_polar.get_position(original=True).height * 9.0 == pytest.approx(width_in)  # square
    assert ax_legend.get_position(original=True).y1 * 9.0 == pytest.approx(9.0 - top)


def _circle_gridlines(ax):
    """The circular gridlines: black Line2Ds at one constant r > 0."""
    return [
        line
        for line in ax.lines
        if line.get_color() == "black"
        and len(line.get_ydata()) > 1
        and np.ptp(line.get_ydata()) == 0
        and line.get_ydata()[0] > 0
    ]


def _gridline_widths(ax):
    return [line.get_linewidth() for line in _circle_gridlines(ax)]


def _line_widths(ax):
    """Widths of the lines other than the circular gridlines."""
    circles = _circle_gridlines(ax)
    return [line.get_linewidth() for line in ax.lines if line not in circles]


def _contour_widths(ax):
    """Distinct line widths of each contour set, in drawing order."""
    return [
        tuple(sorted(set(np.round(np.atleast_1d(c.get_linewidth()), 9))))
        for c in ax.collections
        if hasattr(c, "levels")
    ]


def test_grid_and_vdcc_overrides_change_only_their_target(default_and_overridden):
    (_f0, ax0, _l0), (_f1, ax1, _l1) = default_and_overridden
    assert set(_gridline_widths(ax0)) == {0.4}
    assert set(_gridline_widths(ax1)) == {1.1}  # grid_info circlegrids_lw
    assert _line_widths(ax1) == _line_widths(ax0)  # radials, verticals, unit circle unchanged
    # vdcc_info cc_lw: the +/-0.1..0.9 contour sets change; the 0.98 contour does not
    widths0, widths1 = _contour_widths(ax0), _contour_widths(ax1)
    assert len(widths0) == len(widths1) == 3
    assert widths0[0] == widths0[1] == (0.48,)
    assert widths1[0] == widths1[1] == (2.0,)
    assert widths1[2] == widths0[2] == (1.5,)


def _render(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    buf.seek(0)
    return plt.imread(buf)


@pytest.fixture(scope="module")
def default_render():
    return _render(mmd.make_fig_axes()[0])


TOP_LEVEL_NAMES = [
    "figsize_inch",
    "margins_t_b_l_r_inch",
    "vgap_legend_rel_polar_inch",
    "fontsize",
    "grid_info",
    "vdcc_info",
]
GRID_INFO_DEFAULTS = {
    "circlegrids_lw": 0.4,
    "circlegrids_zo": 1.3,
    "radials_vsccequal_lw": 0.6,
    "radials_vsccequal_zo": 1.7,
    "vertical_msdosdequal_lw": 1,
    "vertical_msdosdequal_zo": 1.7,
    "unitcircle_lw": 1.0,
    "unitcircle_zo": 1.6,
}
VDCC_INFO_DEFAULTS = {
    "vdcc_bar_height_inch": 0.25,
    "vdcc_bar_width_frac": 0.80,
    "vdcc_pad_t_b_l_r_inch": (0.3, 0.7, 0.25, 0.25),
    "vdcc_n_theta": 540,
    "vdcc_n_r": 400,
    "vd_steps": np.linspace(0, 1, 11) ** 2,
    "vd_gray_min": 0.5,
    "vd_gray_max": 0.95,
    "cc_pos": np.arange(0.1, 1, 0.1),
    "cc_neg": np.arange(-0.9, 0, 0.1),
    "cc_lw": 0.4 * 1.2,
    "cc098_lw": 1.5,
    "cc_zo": 1,
    "cc_negcorr_dashpattern": (0, (10, 5)),
}


def _listed_names(message, marker):
    return message.split(marker, 1)[1].split(", ")


def test_unknown_keyword_raises_type_error():
    n_figs = len(plt.get_fignums())
    with pytest.raises(TypeError, match="'bogus_setting'") as excinfo:
        mmd.make_fig_axes(bogus_setting=1)
    assert _listed_names(str(excinfo.value), "valid names: ") == TOP_LEVEL_NAMES
    assert len(plt.get_fignums()) == n_figs  # raised before creating a figure


@pytest.mark.parametrize(
    "name, valid", [("grid_info", GRID_INFO_DEFAULTS), ("vdcc_info", VDCC_INFO_DEFAULTS)]
)
def test_unknown_subdict_key_raises_value_error(name, valid):
    n_figs = len(plt.get_fignums())
    with pytest.raises(ValueError, match=rf"{name}.*'bogus_key'") as excinfo:
        mmd.make_fig_axes(**{name: {"bogus_key": 1}})
    assert _listed_names(str(excinfo.value), "valid keys: ") == list(valid)
    assert len(plt.get_fignums()) == n_figs


@pytest.fixture(scope="module")
def ring_and_cc098_overrides():
    return (
        mmd.make_fig_axes(grid_info={"unitcircle_lw": 3.0})[1],
        mmd.make_fig_axes(vdcc_info={"cc098_lw": 0.7})[1],
    )


AXES_NAMES = ("polar axes", "legend axes", "grayscale bar axes")


@pytest.mark.parametrize(
    "kwargs, too_small",
    [
        ({"figsize_inch": (4, 5)}, ["legend axes"]),
        ({"figsize_inch": (4, 5.81)}, ["legend axes"]),  # legend height -0.0007 in
        ({"vdcc_info": {"vdcc_bar_height_inch": 0}}, ["grayscale bar axes"]),
        ({"vdcc_info": {"vdcc_bar_width_frac": -0.1}}, ["grayscale bar axes"]),
        ({"figsize_inch": (1.0, 11)}, list(AXES_NAMES)),  # zero width between the margins
        ({"margins_t_b_l_r_inch": (1, 1, 5, 4)}, list(AXES_NAMES)),  # negative width
    ],
    ids=["4x5", "4x5.81", "bar-height-0", "bar-width-frac<0", "width-0", "margins-too-wide"],
)
def test_too_small_axes_raise_value_error(kwargs, too_small):
    n_figs = len(plt.get_fignums())
    with pytest.raises(ValueError, match=r"figsize_inch=.* margins_t_b_l_r_inch=") as excinfo:
        mmd.make_fig_axes(**kwargs)
    assert [name for name in AXES_NAMES if name in str(excinfo.value)] == too_small
    assert len(plt.get_fignums()) == n_figs  # raised before creating a figure
