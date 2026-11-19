import matplotlib.pyplot as plt
import numpy as np
import pytest

import multimetricdiagram as mmd
from multimetricdiagram.labels import (
    _DIRECTIONS,
    _RINGS,
    _convex_overlap,
    _leader,
    _ring_shifts,
    _Scene,
)

OBS = np.array([1.0, 2.5, 3.0, 4.5, 6.0, 5.5, 7.0, 8.5])
MOD = np.array([1.5, 2.0, 3.8, 4.0, 6.5, 6.2, 6.9, 9.4])


def _diagram(with_labels=True, n=4):
    """Polar axes with n markersticks at the same spot (only MB differs), so labels overlap."""
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(projection="polar")
    ax.set_thetamin(-45)
    ax.set_thetamax(225)
    ax.set_ylim(0, np.sqrt(2))
    results = []
    for i in range(n):
        metrics = mmd.calculate_metrics(MOD + 0.01 * i, OBS)
        label_info = {"text": f"Label {i}"} if with_labels else None
        results.append(mmd.add_markerstick(ax, metrics, label_info=label_info))
    return fig, ax, results


def _labels(results):
    return [label for _marker, _stick, labels in results for label in labels]


@pytest.fixture
def adjusted_and_reference():
    """Two identical diagrams; labels adjusted on the first only, the second drawn as reference."""
    fig_a, ax_a, results_a = _diagram()
    fig_b, _ax_b, results_b = _diagram()
    mmd.adjust_markerstick_labels(ax_a, results_a)
    fig_b.canvas.draw()
    return fig_a, _labels(results_a), fig_b, _labels(results_b)


@pytest.mark.parametrize("n", [0, 3])
def test_adjust_without_labels_returns(n):
    fig, ax, results = _diagram(with_labels=False, n=n)
    n_lines = len(ax.lines)
    assert mmd.adjust_markerstick_labels(ax, results) == []
    assert len(ax.lines) == n_lines  # no leader lines
    fig.canvas.draw()


def test_adjust_never_removes_other_lines():
    _fig, ax, results = _diagram(n=2)
    (plain,) = ax.plot([0.5, 1.0], [0.2, 0.4])
    (tagged,) = ax.plot([0.5, 1.0], [0.6, 0.8], gid="user-line")
    sticks = [stick for _marker, stick, _labels in results]
    for _ in range(2):
        mmd.adjust_markerstick_labels(ax, results)
        assert plain in ax.lines and tagged in ax.lines
        assert all(stick in ax.lines for stick in sticks)


RECT = np.array([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


@pytest.mark.parametrize(
    "p0, p1, hits",
    [
        ((-1.0, 0.5), (2.0, 0.5), True),  # crosses
        ((-1.0, -1.0), (2.0, 2.0), True),  # crosses diagonally
        ((1.0, 0.5), (2.0, 0.5), True),  # touches the right edge
        ((-1.0, 1.0), (0.0, 1.0), True),  # touches a corner
        ((0.2, 0.2), (0.8, 0.7), True),  # inside
        ((0.5, 0.5), (0.5, 0.5), True),  # a point inside
        ((2.0, 2.0), (3.0, 3.0), False),  # outside
        ((-1.0, 2.0), (2.0, 1.5), False),  # passes above
        ((-1.0, 2.0), (2.0, 2.0), False),  # parallel to x, above
        ((-1.0, 0.5), (-0.5, 0.5), False),  # parallel to x, within the y band, left of it
        ((1.5, -1.0), (1.5, 2.0), False),  # parallel to y, right of it
        ((0.0, -1.0), (0.0, 2.0), True),  # along the left edge
    ],
)
def test_convex_overlap_segment_and_rectangle(p0, p1, hits):
    assert _convex_overlap(np.array([p0, p1]), RECT) is hits
    assert _convex_overlap(RECT, np.array([p1, p0])) is hits  # order does not matter


def _empty_scene():
    return _Scene([], np.zeros((0, 4)), 0.0, lambda quad: False)


def test_leader_starts_where_its_line_crosses_no_label():
    label = np.array([(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)])
    anchor = (0.0, 0.5)
    crosses, length, start = _leader(label, anchor, 0, _empty_scene(), [])
    assert not crosses and tuple(start) == (2.0, 0.5) and length == 2.0  # nearest point
    blocker = np.array([(1.0, 0.4), (1.5, 0.4), (1.5, 0.6), (1.0, 0.6)])  # in the way
    crosses, length, start = _leader(label, anchor, 0, _empty_scene(), [blocker])
    assert not crosses and tuple(start) in {(2.0, 0.0), (2.0, 1.0)}  # a near corner instead


# --- candidate spots around a marker -------------------------------------------------------


def _ring_case():
    from matplotlib.transforms import Bbox

    bbox = Bbox.from_bounds(400.0, 300.0, 60.0, 14.0)  # a label's box, w x h
    anchor = np.array([200.0, 150.0])
    return bbox, anchor, (8.0, 6.0), (130.0, 100.0)  # gap_min, gap_max (x, y)


def test_ring_shifts_lie_between_the_two_rings():
    bbox, anchor, gap_min, gap_max = _ring_case()
    shifts = _ring_shifts(bbox, anchor, gap_min, gap_max)
    assert len(shifts) == _RINGS * _DIRECTIONS
    centers = np.array([bbox.x0, bbox.y0]) + shifts + (bbox.width / 2, bbox.height / 2)
    # gap from the anchor to the box, horizontally and vertically (negative: overlapping)
    gap_x = np.abs(centers[:, 0] - anchor[0]) - bbox.width / 2
    gap_y = np.abs(centers[:, 1] - anchor[1]) - bbox.height / 2
    # each box clears the inner ring and stays within the outer ring
    fx = (gap_x - gap_min[0]) / (gap_max[0] - gap_min[0])
    fy = (gap_y - gap_min[1]) / (gap_max[1] - gap_min[1])
    ring = np.maximum(fx, fy)
    assert np.all(ring > -1e-9) and np.all(ring < 1 + 1e-9)
    assert np.isclose(ring.min(), 0) and np.isclose(ring.max(), 1)


def test_ring_shifts_nearest_first_and_all_around():
    bbox, anchor, gap_min, gap_max = _ring_case()
    shifts = _ring_shifts(bbox, anchor, gap_min, gap_max)
    rel = np.array([bbox.x0, bbox.y0]) + shifts + (bbox.width / 2, bbox.height / 2) - anchor
    dist = np.hypot(*rel.T)
    assert np.all(np.diff(dist) >= -1e-9)
    angles = np.sort(np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) % 360)
    gaps = np.diff(np.concatenate([angles, [angles[0] + 360]]))
    assert gaps.max() < 360 / _DIRECTIONS + 1e-6  # every direction, none skipped
    for quadrant in range(4):  # above, below, left and right of the marker too
        assert np.any((angles >= 90 * quadrant) & (angles < 90 * (quadrant + 1)))


def test_importing_the_package_does_not_import_textalloc():
    import subprocess
    import sys

    code = "import sys, multimetricdiagram.alt.taylor; print('textalloc' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
