import pytest

import multimetricdiagram as mmd
from multimetricdiagram.alt import taylor

MMD_KEYS = [
    "circular_gridlines",
    "dashed_radials",
    "vertical_line",
    "unit_circle",
    "star",
    "sector_text",
    "cc_caption",
    "shading",
    "cc_contours",
    "markerstick_keys",
    "grayscale_bar_ax",
    "grayscale_bar",
]
TAYLOR_KEYS = [
    "cc_radials",
    "msd_circles",
    "rmsd_circles",
    "rmsd_tick_labels",
    "msd_tick_labels",
    "axis_titles",
    "cc_tick_labels",
    "cc_title",
    "star",
]

DIAGRAMS = {
    "mmd": (mmd.make_fig_axes, {}),
    "taylor-1q": (taylor.make_fig_axes, {"quadrants": 1}),
    "taylor-2q": (taylor.make_fig_axes, {"quadrants": 2}),
}


def _flat(artists):
    """Every artist in the dict, list values unpacked, except the grayscale bar's Axes."""
    out = []
    for key, value in artists.items():
        if key != "grayscale_bar_ax":
            out += value if isinstance(value, list) else [value]
    return out


def _drawn(ax):
    return set(ax.lines) | set(ax.patches) | set(ax.collections) | set(ax.texts)


@pytest.fixture(scope="module", params=list(DIAGRAMS))
def diagram(request):
    make, kwargs = DIAGRAMS[request.param]
    return request.param, make(return_artists=True, **kwargs)


@pytest.mark.parametrize("name", list(DIAGRAMS))
def test_default_and_false_return_three_items(name):
    make, kwargs = DIAGRAMS[name]
    for out in (make(**kwargs), make(return_artists=False, **kwargs)):
        assert len(out) == 3
        assert all(not isinstance(item, dict) for item in out)


def test_true_returns_four_items_with_the_listed_keys(diagram):
    name, out = diagram
    assert len(out) == 4
    fig, ax_polar, ax_legend, artists = out
    assert isinstance(artists, dict)
    assert list(artists) == (MMD_KEYS if name == "mmd" else TAYLOR_KEYS)
    assert ax_polar in fig.axes and ax_legend in fig.axes


def test_each_background_artist_in_exactly_one_group(diagram):
    _name, (fig, ax_polar, ax_legend, artists) = diagram
    flat = _flat(artists)
    assert len(flat) == len(set(flat))  # no artist in two groups
    expected = _drawn(ax_polar)
    if "grayscale_bar_ax" in artists:
        bar = artists["grayscale_bar_ax"]
        expected |= _drawn(bar) | {bar.title}
        assert set(fig.axes) == {ax_polar, ax_legend, bar}
    assert set(flat) == expected
    assert not _drawn(ax_legend)
