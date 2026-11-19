import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

import multimetricdiagram as mmd


@pytest.fixture
def polar_ax():
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="polar")
    ax.set_ylim(0, np.sqrt(2))
    return ax


def _draw(ax):
    ax.get_figure().canvas.draw()


def test_set_color_after_draw_persists_through_next_draw(polar_ax):
    h = mmd.text_polar(polar_ax, np.deg2rad(60), 0.8, "Text", color="b")
    _draw(polar_ax)
    h.set_color("g")
    _draw(polar_ax)
    assert h.get_facecolor() == to_rgba("g")
    assert h.get_edgecolor() == to_rgba("g")


def test_centered_text_is_centered_on_anchor(polar_ax):
    theta, r = np.deg2rad(60), 0.8
    h = mmd.text_polar(polar_ax, theta, r, "Centered", fontsize=12)
    _draw(polar_ax)
    bb = h.get_window_extent()
    ax_x, ax_y = polar_ax.transData.transform((theta, r))
    assert (bb.x0 + bb.x1) / 2 == pytest.approx(ax_x, abs=0.05 * bb.width)
    assert (bb.y0 + bb.y1) / 2 == pytest.approx(ax_y, abs=0.05 * bb.height)


def test_rlimit_change_moves_text_at_constant_size(polar_ax):
    theta, r = np.deg2rad(60), 0.8
    h = mmd.text_polar(polar_ax, theta, r, "Text", fontsize=12)
    _draw(polar_ax)
    before = h.get_window_extent()
    polar_ax.set_ylim(0, 2 * np.sqrt(2))
    _draw(polar_ax)
    after = h.get_window_extent()
    assert not np.allclose(after.extents, before.extents)  # transform recomputed at draw
    assert after.width == pytest.approx(before.width, rel=1e-6)  # fontsize is in points
    assert after.height == pytest.approx(before.height, rel=1e-6)
    ax_x, ax_y = polar_ax.transData.transform((theta, r))  # follows the new anchor position
    assert (after.x0 + after.x1) / 2 == pytest.approx(ax_x, abs=0.05 * after.width)
    assert (after.y0 + after.y1) / 2 == pytest.approx(ax_y, abs=0.05 * after.height)


@pytest.mark.parametrize("ha", ["left", "center", "right"])
@pytest.mark.parametrize("va", ["bottom", "center", "top"])
def test_alignment_places_text_on_correct_side(polar_ax, ha, va):
    theta, r = np.deg2rad(60), 0.8
    h = mmd.text_polar(polar_ax, theta, r, "Aligned", ha=ha, va=va)
    _draw(polar_ax)
    bb = h.get_window_extent()
    ax_x, ax_y = polar_ax.transData.transform((theta, r))
    x_ref = {"left": bb.x0, "center": (bb.x0 + bb.x1) / 2, "right": bb.x1}[ha]
    y_ref = {"bottom": bb.y0, "center": (bb.y0 + bb.y1) / 2, "top": bb.y1}[va]
    # the ink box's edge or center is at the anchor
    assert x_ref == pytest.approx(ax_x, abs=1e-6)
    assert y_ref == pytest.approx(ax_y, abs=1e-6)
    # and the text lies on the expected side of the anchor
    assert {
        "left": bb.x1 > ax_x + 0.5 * bb.width,
        "center": bb.x0 < ax_x < bb.x1,
        "right": bb.x0 < ax_x - 0.5 * bb.width,
    }[ha]
    assert {
        "bottom": bb.y1 > ax_y + 0.5 * bb.height,
        "center": bb.y0 < ax_y < bb.y1,
        "top": bb.y0 < ax_y - 0.5 * bb.height,
    }[va]


@pytest.mark.parametrize("kwargs", [{"ha": "middle"}, {"va": "baseline"}])
def test_invalid_alignment_raises(polar_ax, kwargs):
    with pytest.raises(ValueError):
        mmd.text_polar(polar_ax, 0.5, 0.5, "x", **kwargs)


NON_STRING_TEXTS = [12, 12.5, np.int64(7), np.float64(12.5), None]
NON_STRING_IDS = ["int", "float", "np-int64", "np-float64", "None"]


def _draws_text(patch, expected):
    """The patch's path is that of `expected` (empty for '')."""
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    path = patch.get_path()
    if expected == "":
        return len(path.vertices) == 0
    reference = TextPath((0, 0), expected, size=1, prop=FontProperties(weight="normal"))
    return np.array_equal(path.vertices, reference.vertices) and np.array_equal(
        path.codes, reference.codes
    )


@pytest.mark.parametrize("value", NON_STRING_TEXTS, ids=NON_STRING_IDS)
def test_non_string_text_converted_with_str(polar_ax, value):
    h = mmd.text_polar(polar_ax, np.deg2rad(60), 0.8, value)
    _draw(polar_ax)
    assert _draws_text(h, "" if value is None else str(value))


def test_empty_text_draws_nothing(polar_ax):
    h = mmd.text_polar(polar_ax, np.deg2rad(60), 0.8, "", ha="left", va="top")
    _draw(polar_ax)
    assert len(h.get_path().vertices) == 0
    assert np.all(np.isfinite(h.get_transform().get_matrix()))


def _ink_extents_pt(patch, anchor, rotation=0.0):
    """(x0, x1, y0, y1) in points of a drawn text_polar() patch in the text's own frame, about
    the anchor (theta, r)."""
    ax = patch.axes
    to_frame = (
        Affine2D()
        .translate(*(-ax.transData.transform(anchor)))
        .rotate_deg(-rotation)
        .scale(72 / ax.figure.dpi)
    )
    ext = patch.get_path().transformed(patch.get_transform() + to_frame).get_extents()
    return np.array([ext.x0, ext.x1, ext.y0, ext.y1])


@pytest.fixture
def diagram_ax():
    _fig, ax, _legend_ax = mmd.make_fig_axes()
    return ax


ANCHOR = (np.deg2rad(60), 0.8)


def test_font_size_is_nominal(diagram_ax):
    h = mmd.text_polar(diagram_ax, *ANCHOR, "Hg", fontsize=20)
    diagram_ax.figure.canvas.draw()
    x0, x1, y0, y1 = _ink_extents_pt(h, ANCHOR)
    ink = TextPath((0, 0), "Hg", size=20, prop=FontProperties(weight="normal")).get_extents()
    assert (x1 - x0, y1 - y0) == pytest.approx((ink.width, ink.height), rel=0.01)


@pytest.mark.parametrize("rotation", [30])
@pytest.mark.parametrize("out, cw", [(1, 0), (0, -1)])
def test_offset_of_one_fontsize_moves_text_fontsize_points(diagram_ax, rotation, out, cw):
    kwargs = {"fontsize": 20, "rotation": rotation}
    still = mmd.text_polar(diagram_ax, *ANCHOR, "Hg", **kwargs)
    moved = mmd.text_polar(diagram_ax, *ANCHOR, "Hg", out_offset=out, cw_offset=cw, **kwargs)
    diagram_ax.figure.canvas.draw()
    shift = _ink_extents_pt(moved, ANCHOR, rotation) - _ink_extents_pt(still, ANCHOR, rotation)
    assert shift == pytest.approx([20 * cw, 20 * cw, 20 * out, 20 * out], abs=0.1)
