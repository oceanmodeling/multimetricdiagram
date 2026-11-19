import numpy as np
import pytest

import multimetricdiagram as mmd
from multimetricdiagram.alt import taylor


def _mmd_at(theta_deg, r, mb_on_sdq=0.03):
    """Self-consistent top-level metrics dict (ddof=0) putting the marker at (theta_deg, r),
    r < 1."""
    mirrored = theta_deg > 90  # MSD < OSD
    th = 180 - theta_deg if mirrored else theta_deg
    cc = (1 - r * r) ** ((th + 45) / 135)
    vs = (1 - r * r) / cc
    k = (1 + np.sqrt(1 - vs * vs)) / vs
    return {
        "RMSDonSDQ": np.hypot(r, mb_on_sdq),
        "URMSDonSDQ": r,
        "MBonSDQ": mb_on_sdq,
        "CC": cc,
        "VS": vs,
        "MSD": 1 / k if mirrored else k,
        "OSD": 1.0,
        "n": 50,
        "ddof": 0,
    }


def _taylor_metrics(msd, cc, mb):
    urmsd = np.sqrt(msd**2 + 1 - 2 * msd * cc)
    rmsd = np.hypot(urmsd, mb)
    return {
        "MSD": msd,
        "OSD": 1.0,
        "CC": cc,
        "RMSD": rmsd,
        "URMSD": urmsd,
        "MB": mb,
        "n": 50,
        "ddof": 0,
        "MSDonOSD": msd,
        "RMSDonOSD": rmsd,
        "URMSDonOSD": urmsd,
        "MBonOSD": mb,
    }


def _leaders(ax):
    return [ln for ln in ax.lines if ln.get_gid() == "multimetricdiagram-leader"]


def _extents(results):
    return [label.get_window_extent().extents.copy() for _m, _s, ls in results for label in ls]


def _adjust(ax, specs, add=mmd.add_markerstick):
    """Add (metrics, label_info, style) markersticks, adjust; (results, moved per label)."""
    results = [add(ax, m, label_info=li, markerstick_style=st) for m, li, st in specs]
    ax.figure.draw_without_rendering()
    before = _extents(results)
    mmd.adjust_markerstick_labels(ax, results)
    ax.figure.draw_without_rendering()
    moved = [not np.allclose(a, b) for a, b in zip(_extents(results), before)]
    return results, moved


def test_isolated_label_stays_without_leader():
    _fig, ax, _l = mmd.make_fig_axes()
    results, moved = _adjust(ax, [(_mmd_at(60, 0.6), {"text": "Label"}, None)])
    assert moved == [False] and not _leaders(ax)
    assert results[0][2][0]._offset_inches == (0.0, 0.0)


@pytest.mark.parametrize("quadrants", [1, 2])
def test_isolated_label_stays_without_leader_taylor(quadrants):
    _fig, ax, _l = taylor.make_fig_axes(quadrants=quadrants)
    specs = [(_taylor_metrics(0.8, 0.9, 0.05), {"text": "Model"}, None)]
    _results, moved = _adjust(ax, specs, taylor.add_markerstick)
    assert moved == [False] and not _leaders(ax)


# two markers on one radial line, big labels facing each other in the gap between them
FACING = [
    (_mmd_at(70, 0.85), {"text": "XX", "position": "inside", "fontsize": 30}, None),
    (_mmd_at(70, 0.55), {"text": "XX", "position": "outside", "fontsize": 30}, None),
]


@pytest.mark.parametrize("order", [(0, 1), (1, 0)], ids=["inner-label-first", "outer-first"])
def test_of_two_labels_overlapping_only_each_other_the_first_stays(order):
    _fig, ax, _l = mmd.make_fig_axes()
    _results, moved = _adjust(ax, [FACING[i] for i in order])
    assert moved == [False, True]
    assert len(_leaders(ax)) == 1


# label_info of the first markerstick, then the other markersticks (no labels)
OVERLAPS = {
    "another marker": [(_mmd_at(70, 0.6), {"text": "Label"}, None), (_mmd_at(70, 0.66), [], None)],
    "another stick": [
        (_mmd_at(70, 0.6), {"text": "Label"}, None),
        (_mmd_at(90, 0.62, 0.4), [], None),
    ],
    "the star": [(_mmd_at(60, 0.03), {"text": "A label"}, {"markersize": 4})],
    "a tick label": [
        (_mmd_at(-30, 0.4), {"text": "A long label", "position": "outside", "fontsize": 12}, None)
    ],
    "its own marker": [(_mmd_at(60, 0.6), {"text": "Label", "position": None}, None)],
}


@pytest.mark.parametrize("specs", list(OVERLAPS.values()), ids=list(OVERLAPS))
def test_label_overlapping_something_moves(specs):
    _fig, ax, _l = mmd.make_fig_axes()
    _results, moved = _adjust(ax, specs)
    assert moved[0]


def test_repeat_call_gives_identical_result():
    _fig, ax, _l = mmd.make_fig_axes()
    specs = FACING + OVERLAPS["another stick"] + OVERLAPS["the star"]
    results, _moved = _adjust(ax, specs)
    first = (_extents(results), [ln.get_xydata().copy() for ln in _leaders(ax)])
    mmd.adjust_markerstick_labels(ax, results)
    ax.figure.draw_without_rendering()
    second = (_extents(results), [ln.get_xydata().copy() for ln in _leaders(ax)])
    assert len(first[1]) == len(second[1]) > 0
    for a, b in zip(first[0] + first[1], second[0] + second[1]):
        np.testing.assert_array_equal(a, b)


# --- return value: the moved labels and their leader lines --------------------------------


def _add(ax, specs, add=mmd.add_markerstick):
    return [add(ax, m, label_info=li, markerstick_style=st) for m, li, st in specs]


def test_returns_empty_list_when_nothing_moves():
    _fig, ax, _l = mmd.make_fig_axes()
    specs = [(_mmd_at(60, 0.6), {"text": "A"}, None), (_mmd_at(150, 0.5), {"text": "B"}, None)]
    assert mmd.adjust_markerstick_labels(ax, _add(ax, specs)) == []
    assert not _leaders(ax)


@pytest.mark.parametrize(
    "specs, n_moved",
    [(FACING, 1), (FACING + OVERLAPS["another stick"] + OVERLAPS["the star"], 4)],
    ids=["one-of-two-moves", "all-four-move"],
)
def test_returns_the_moved_labels_with_their_leader_lines(specs, n_moved):
    _fig, ax, _l = mmd.make_fig_axes()
    results = _add(ax, specs)
    ax.figure.draw_without_rendering()
    before = _extents(results)
    pairs = mmd.adjust_markerstick_labels(ax, results)
    ax.figure.draw_without_rendering()
    labels = [label for _m, _s, ls in results for label in ls]
    moved = [lab for lab, a, b in zip(labels, before, _extents(results)) if not np.allclose(a, b)]
    assert [label for label, _line in pairs] == moved  # the moved ones, in label order
    assert len(moved) == n_moved
    assert [line for _label, line in pairs if line is not None] == _leaders(ax)
    marker_of = {label: m for m, _s, ls in results for label in ls}
    for label, line in pairs:
        if line is not None:  # it ends at its own marker
            end = ax.transData.transform(line.get_xydata()[-1])
            marker = ax.transData.transform(marker_of[label].get_offsets()[0])
            np.testing.assert_allclose(end, marker, atol=1e-6)
