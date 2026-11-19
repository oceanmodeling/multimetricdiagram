import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

import multimetricdiagram as mmd
from multimetricdiagram.legend import _LABEL_CLEARANCE


def _legend(entries):
    _fig, ax = plt.subplots(figsize=(4, 4))
    return mmd.add_labeledmarker_legend(ax, entries, loc="center")


def _label_rgba(entry):
    _legend_artist, _handles, labels_artists = _legend([entry])
    return to_rgba(labels_artists[0][0].get_color())


# (markerstick_style, expected default marker-label color); same rule as add_markerstick() labels
LABEL_COLOR_CASES = [
    (None, "r"),  # add_markerstick() default color
    ({"color": "b"}, "b"),
    ({"edgecolors": "g"}, "g"),
    ({"marker": {"edgecolors": "g"}}, "g"),
    ({"marker": {"edgecolors": "c"}}, "c"),
    ({"edgecolors": "none", "facecolors": "m"}, "m"),
    ({"marker": {"edgecolors": "none", "facecolors": "m"}}, "m"),
    ({"edgecolors": "none", "facecolors": "none"}, "k"),
    # a non-None marker alpha replaces a color's own alpha when drawn, so alpha-0 colors
    # are invisible only with alpha None
    ({"edgecolors": (0, 1, 0, 0), "facecolors": "m"}, (0, 1, 0, 0)),
    ({"alpha": None, "edgecolors": (0, 1, 0, 0), "facecolors": "m"}, "m"),
    ({"alpha": None, "edgecolors": "none", "facecolors": (1, 0, 1, 0)}, "k"),
]
LABEL_COLOR_IDS = [
    "default",
    "color",
    "edgecolors",
    "marker-edgecolors-g",
    "marker-edgecolors-c",
    "edge-none-face-m",
    "marker-edge-none-face-m",
    "edge-none-face-none",
    "edge-alpha0-marker-alpha0.6",
    "edge-alpha0-marker-alphaNone-face-m",
    "edge-none-face-alpha0-marker-alphaNone",
]


def _visible_rgb(color, alpha):
    """RGB of a color as drawn with the given artist alpha, or None if invisible."""
    rgba = to_rgba(color, alpha)
    return None if rgba[3] == 0 else rgba[:3]


@pytest.mark.parametrize("markerstick_style, expected", LABEL_COLOR_CASES, ids=LABEL_COLOR_IDS)
def test_marker_label_defaults_to_marker_edgecolor(markerstick_style, expected):
    entry = {
        "label": "Entry",
        "markerstick_style": markerstick_style,
        "label_info": [{"text": "x"}, {"text": "y", "position": "inside"}],
    }
    _legend_artist, handles, labels_artists = _legend([entry])
    for label in labels_artists[0]:
        assert to_rgba(label.get_color()) == to_rgba(expected)
    # read back from the legend marker: label matches its edge if visible, else its face if
    # visible, else black
    handle = handles[0]
    actual = (
        _visible_rgb(handle.get_markeredgecolor(), handle.get_alpha())
        or _visible_rgb(handle.get_markerfacecolor(), handle.get_alpha())
        or to_rgba("k")[:3]
    )
    assert to_rgba(labels_artists[0][0].get_color())[:3] == pytest.approx(actual)


@pytest.mark.parametrize("markerstick_style, _expected", LABEL_COLOR_CASES, ids=LABEL_COLOR_IDS)
def test_explicit_label_color_wins(markerstick_style, _expected):
    entry = {
        "label": "Entry",
        "markerstick_style": markerstick_style,
        "label_info": {"text": "x", "color": "orange"},
    }
    assert _label_rgba(entry) == to_rgba("orange")


def _row_centers(handles):
    handles[0].get_figure().canvas.draw()  # the legend is laid out at draw time
    return [(h.get_window_extent().y0 + h.get_window_extent().y1) / 2 for h in handles]


def test_rows_keep_entry_order():
    names = ["first", "second", "third", "fourth"]
    legend_artist, handles, _labels = _legend([{"label": n} for n in names])
    assert [t.get_text() for t in legend_artist.get_texts()] == names
    centers = _row_centers(handles)
    assert centers == sorted(centers, reverse=True)  # top to bottom


# --- layout by the legend handler: geometry read back after drawing ---


def _entries(position, n_rows=3):
    styles = [{"color": "b"}, {"marker": "s", "color": "g"}, {"marker": "^", "color": "m"}]
    texts = ["MMM", "YYYY", "Wide label"]
    return [
        {
            "label": f"Row {i}",
            "markerstick_style": styles[i % 3],
            "label_info": {"text": texts[i % 3], "position": position},
        }
        for i in range(n_rows)
    ]


def _center(bbox):
    return np.array([(bbox.x0 + bbox.x1) / 2, (bbox.y0 + bbox.y1) / 2])


def _geometry_pt(fig, legend_artist, handles, labels_artists):
    """After a draw: legend frame size, and label centers relative to their markers, in points."""
    fig.canvas.draw()
    to_pt = 72 / fig.dpi
    frame = legend_artist.get_frame().get_window_extent()
    offsets = [
        (_center(label.get_window_extent()) - _center(h.get_window_extent())) * to_pt
        for h, row in zip(handles, labels_artists)
        for label in row
    ]
    return np.array([frame.width, frame.height]) * to_pt, np.array(offsets)


def _overlaps(a, b):
    return a.x0 < b.x1 and b.x0 < a.x1 and a.y0 < b.y1 and b.y0 < a.y1


ALL_POSITIONS = pytest.mark.parametrize("position", ["outside", "inside", "beside"])


def _frame_size_pt(fig, legend_artist):
    fig.canvas.draw()
    frame = legend_artist.get_frame().get_window_extent()
    return np.array([frame.width, frame.height]) * 72 / fig.dpi


@ALL_POSITIONS
def test_layout_same_in_points_at_dpi_100_and_300(position):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    result = mmd.add_labeledmarker_legend(ax, _entries(position), loc="center")
    # a plain legend with the same rows: matplotlib rounds text heights to whole pixels, so
    # legend sizes in points change a little with dpi; ours may change no more than that
    plain_fig, plain_ax = plt.subplots(figsize=(4, 4), dpi=100)
    plain = plain_ax.legend(
        [Line2D([], [], marker="o", linestyle="none") for _ in result[1]],
        [t.get_text() for t in result[0].get_texts()],
        loc="center",
    )
    size_100, offsets_100 = _geometry_pt(fig, *result)
    plain_100 = _frame_size_pt(plain_fig, plain)
    fig.set_dpi(300)
    plain_fig.set_dpi(300)
    size_300, offsets_300 = _geometry_pt(fig, *result)
    plain_300 = _frame_size_pt(plain_fig, plain)
    assert np.all(np.abs(size_300 - size_100) <= np.abs(plain_300 - plain_100) + 0.5)
    assert offsets_300 == pytest.approx(offsets_100, abs=0.5)


@ALL_POSITIONS
def test_frame_encloses_rows_and_rows_do_not_overlap(position):
    fig, ax = plt.subplots(figsize=(5, 5))
    legend_artist, handles, labels_artists = mmd.add_labeledmarker_legend(
        ax, _entries(position), loc="center"
    )
    fig.canvas.draw()
    _assert_enclosed_and_separate(legend_artist, handles, labels_artists)


def _assert_enclosed_and_separate(legend_artist, handles, labels_artists):
    """Frame encloses every marker, text and label; no label overlaps another row's items."""
    frame = legend_artist.get_frame().get_window_extent()
    rows = [
        [h.get_window_extent(), t.get_window_extent(), *(a.get_window_extent() for a in row)]
        for h, t, row in zip(handles, legend_artist.get_texts(), labels_artists)
    ]
    for row in rows:
        for bbox in row:
            assert frame.x0 <= bbox.x0 and bbox.x1 <= frame.x1
            assert frame.y0 <= bbox.y0 and bbox.y1 <= frame.y1
    for i, row in enumerate(rows):
        for label in row[2:]:
            for j, other in enumerate(rows):
                if j != i:
                    assert not any(_overlaps(label, bbox) for bbox in other), (i, j)


@pytest.mark.parametrize(
    "position, check",
    [
        ("outside", lambda label, c: label.y0 > c[1] and label.x0 < c[0] < label.x1),
        ("inside", lambda label, c: label.y1 < c[1] and label.x0 < c[0] < label.x1),
        ("beside", lambda label, c: label.x1 < c[0] and label.y0 < c[1] < label.y1),
    ],
)
def test_label_position_relative_to_marker_center(position, check):
    fig, ax = plt.subplots(figsize=(4, 4))
    _l, handles, labels_artists = mmd.add_labeledmarker_legend(
        ax, _entries(position, n_rows=2), loc="center"
    )
    fig.canvas.draw()
    for h, row in zip(handles, labels_artists):
        assert check(row[0].get_window_extent(), _center(h.get_window_extent()))


@pytest.mark.parametrize(
    "position, given, legend_kwargs",
    [
        ("outside", {}, {}),
        ("inside", {}, {}),
        ("beside", {}, {}),
        ("outside", {"out_offset": 0.4}, {}),
        ("inside", {"out_offset": -0.4}, {}),
        ("beside", {"cw_offset": -0.4}, {}),
        ("outside", {}, {"markerscale": 2}),
    ],
    ids=["outside", "inside", "beside", "outside-0.4", "inside-0.4", "beside-0.4", "markerscale"],
)
@pytest.mark.parametrize("text", ["Long label"])
@pytest.mark.parametrize("style", [{}, {"marker": "^", "markersize": 8, "linewidths": 3.0}])
def test_label_gap_from_marker_edge(position, given, legend_kwargs, style, text):
    fig, ax = plt.subplots(figsize=(4, 4))
    entry = {
        "label": "A",
        "markerstick_style": style,
        "label_info": {"text": text, "position": position, **given},
    }
    _l, handles, labels_artists = mmd.add_labeledmarker_legend(
        ax, [entry], loc="center", **legend_kwargs
    )
    fig.canvas.draw()
    to_pt = 72 / fig.dpi
    h = handles[0]  # its extent: the marker's size (o, ^: path within +-1/2 size), no edge
    marker = h.get_window_extent().padded(h.get_markeredgewidth() / 2 / to_pt)
    label = labels_artists[0][0].get_window_extent()
    gap = {
        "outside": label.y0 - marker.y1,
        "inside": marker.y0 - label.y1,
        "beside": marker.x0 - label.x1,
    }[position] * to_pt
    expected = abs(next(iter(given.values()), 0.1)) * 10  # offset x fontsize
    assert gap == pytest.approx(expected, abs=0.1)


def test_dpi_argument_rejected():
    _fig, ax = plt.subplots()
    with pytest.raises(TypeError, match="no longer accepts dpi.*does not depend on dpi"):
        mmd.add_labeledmarker_legend(ax, _entries("outside"), dpi=300)


def test_earlier_legend_kept_with_add_artist_survives():
    fig, ax = plt.subplots()
    (line,) = ax.plot([0, 1], [0, 1], label="data")
    first = ax.legend(handles=[line], loc="upper right")
    ax.add_artist(first)
    legend_artist, _handles, _labels = mmd.add_labeledmarker_legend(
        ax, _entries("outside"), loc="lower left"
    )
    fig.canvas.draw()
    assert first in ax.get_children() and legend_artist is ax.get_legend()
    assert first.get_window_extent().width > 0 and first.get_texts()[0].get_text() == "data"


def test_invalid_label_position_raises():
    _fig, ax = plt.subplots()
    entry = {"label": "A", "label_info": {"text": "x", "position": "above"}}
    with pytest.raises(ValueError, match="'outside', 'inside', or 'beside'"):
        mmd.add_labeledmarker_legend(ax, [entry])


def test_entry_without_label_raises():
    _fig, ax = plt.subplots()
    entries = [{"label": "A"}, {"label_info": {"text": "x"}}]
    with pytest.raises(ValueError, match=r"^entries\[1\] dict needs a 'label' key$"):
        mmd.add_labeledmarker_legend(ax, entries)
    assert ax.get_legend() is None


@pytest.mark.parametrize("entry", ["A", ("A",), None], ids=["str", "tuple", "None"])
def test_entry_not_dict_raises(entry):
    _fig, ax = plt.subplots()
    with pytest.raises(TypeError, match=r"^entries\[1\] must be a dict, got "):
        mmd.add_labeledmarker_legend(ax, [{"label": "A"}, entry])
    assert ax.get_legend() is None


def test_one_dict_same_as_list_of_one():
    entry = {
        "label": "Row",
        "markerstick_style": {"marker": "s", "color": "g"},
        "label_info": [{"text": "MMM"}, {"text": "YYYY", "position": "beside"}],
    }
    drawn = []
    for entries in (entry, [entry], (entry,)):
        fig, ax = plt.subplots(figsize=(4, 4))
        legend_artist, handles, labels_artists = mmd.add_labeledmarker_legend(
            ax, entries, loc="center"
        )
        assert type(handles) is list and len(handles) == 1
        assert type(labels_artists) is list and len(labels_artists) == 1
        assert [a.get_text() for a in labels_artists[0]] == ["MMM", "YYYY"]
        assert [t.get_text() for t in legend_artist.get_texts()] == ["Row"]
        fig.canvas.draw()
        drawn.append(_drawn_rows_pt(fig, legend_artist, handles, labels_artists))
    assert drawn[0] == drawn[1] == drawn[2]


@pytest.mark.parametrize(
    "entries", ["A", None, iter([{"label": "A"}])], ids=["str", "None", "iterator"]
)
def test_entries_not_dict_list_or_tuple_raises(entries):
    _fig, ax = plt.subplots()
    with pytest.raises(TypeError, match=r"^entries must be a dict or a list \(or tuple\) of "):
        mmd.add_labeledmarker_legend(ax, entries)
    with pytest.raises(ValueError, match=r"^entries dict needs a 'label' key$"):
        mmd.add_labeledmarker_legend(ax, {"label_info": {"text": "x"}})
    assert ax.get_legend() is None


LI = r"entries\[0\]\['label_info'\]"


@pytest.mark.parametrize(
    "label_info, error, match",
    [
        ({"position": "inside"}, ValueError, rf"^{LI} dict needs a 'text' key$"),
        ([{"text": "a"}, {}], ValueError, rf"^{LI}\[1\] dict needs a 'text' key$"),
        ([{"text": "a"}, "b"], TypeError, rf"^{LI}\[1\] must be a dict, got str$"),
        ("ab", TypeError, rf"^{LI} must be None, a dict or a list of dicts, got str$"),
        (
            ({"text": "a"},),
            TypeError,
            rf"^{LI} must be None, a dict or a list of dicts, got tuple$",
        ),
    ],
    ids=["dict-no-text", "list-item-no-text", "list-item-str", "str", "tuple"],
)
def test_bad_label_info_raises(label_info, error, match):
    _fig, ax = plt.subplots()
    with pytest.raises(error, match=match):
        mmd.add_labeledmarker_legend(ax, [{"label": "A", "label_info": label_info}])
    assert ax.get_legend() is None


@pytest.mark.parametrize("position", ["outside", "inside", "beside"])
def test_two_columns_frame_encloses_and_rows_do_not_overlap(position):
    fig, ax = plt.subplots(figsize=(7, 5))
    legend_artist, handles, labels_artists = mmd.add_labeledmarker_legend(
        ax, _entries(position, n_rows=5), loc="center", ncol=2
    )
    fig.canvas.draw()
    _assert_enclosed_and_separate(legend_artist, handles, labels_artists)


def test_title_stays_clear_of_first_row_labels():
    fig, ax = plt.subplots(figsize=(5, 5))
    legend_artist, handles, labels_artists = mmd.add_labeledmarker_legend(
        ax, _entries("outside"), loc="center", title="Legend title"
    )
    fig.canvas.draw()
    assert legend_artist.get_title().get_text() == "Legend title"
    title = legend_artist.get_title().get_window_extent()
    for label in (a for row in labels_artists for a in row):
        assert not _overlaps(label.get_window_extent(), title)
    _assert_enclosed_and_separate(legend_artist, handles, labels_artists)


def test_numeric_and_none_label_texts():
    fig, ax = plt.subplots(figsize=(5, 5))
    entries = [
        {"label": 3, "label_info": {"text": 12}},
        {"label": np.float64(12.5), "label_info": {"text": np.int64(7), "position": "inside"}},
        {"label": None, "label_info": {"text": None, "position": "beside"}},
    ]
    legend_artist, handles, labels_artists = mmd.add_labeledmarker_legend(
        ax, entries, loc="center"
    )
    fig.canvas.draw()
    assert [t.get_text() for t in legend_artist.get_texts()] == ["3", "12.5", ""]
    assert [row[0].get_text() for row in labels_artists] == ["12", "7", ""]
    _assert_enclosed_and_separate(legend_artist, handles, labels_artists)


# --- one column: row pitch from the rows' order --------------------------------------------

OUTSIDE_ROW = {"label": "Near-Surface Temperature", "label_info": {"text": "StnID"}}
BESIDE_INSIDE_ROW = {
    "label": "Near-Bottom Temperature",
    "markerstick_style": {"marker": "s", "color": "b"},
    "label_info": [
        {"text": "StnID", "position": "beside"},
        {"text": "Depth [m]", "position": "inside"},
    ],
}


def _drawn_rows_pt(fig, legend_artist, handles, labels_artists):
    """Per row, boxes (x0, x1, y0, y1) in points: marker (edge included), text, labels."""
    to_pt = 72 / fig.dpi
    rows = []
    for h, t, labels in zip(handles, legend_artist.get_texts(), labels_artists):
        pad = h.get_markeredgewidth() / 2
        boxes = [h.get_window_extent(), t.get_window_extent()]
        boxes += [a.get_window_extent() for a in labels]
        row = [(b.x0 * to_pt, b.x1 * to_pt, b.y0 * to_pt, b.y1 * to_pt) for b in boxes]
        row[0] = (row[0][0] - pad, row[0][1] + pad, row[0][2] - pad, row[0][3] + pad)
        rows.append(row)
    return rows


def test_one_column_pitch_follows_row_order():
    gaps = []  # between the two rows' marker centers, points
    for entries in ([OUTSIDE_ROW, BESIDE_INSIDE_ROW], [BESIDE_INSIDE_ROW, OUTSIDE_ROW]):
        fig, ax = plt.subplots(figsize=(5, 5))
        result = mmd.add_labeledmarker_legend(ax, entries, loc="center")
        fig.canvas.draw()
        _assert_enclosed_and_separate(*result)
        clear = _LABEL_CLEARANCE * result[0].get_texts()[0].get_fontsize()
        rows = _drawn_rows_pt(fig, *result)
        for i, j in ((0, 1), (1, 0)):
            for p in [rows[i][0], *rows[i][2:]]:  # a marker or label of row i ...
                for q in rows[j]:  # ... and any item of row j
                    h_gap = max(q[0] - p[1], p[0] - q[1])
                    v_gap = max(q[2] - p[3], p[2] - q[3])
                    # less one pixel: drawn text heights are rounded to whole pixels
                    assert max(h_gap, v_gap) >= clear - 72 / fig.dpi, (entries[0]["label"], i, j)
        gaps.append(np.diff(_row_centers(result[1]))[0] * -72 / fig.dpi)
    # the 'outside' label above the top row needs no room between the rows; in reverse order
    # the gap holds both the 'inside' and the 'outside' label
    assert gaps[0] < gaps[1] - 10
