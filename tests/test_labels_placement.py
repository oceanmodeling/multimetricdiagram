import numpy as np
import pytest
from matplotlib.transforms import Affine2D

import multimetricdiagram as mmd
from multimetricdiagram.alt import taylor


def _mmd_at(theta_deg, r, mb_on_sdq=0.03):
    """Top-level metrics dict (ddof=0) putting the marker at (theta_deg, r), r < 1."""
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


def _quad(label):
    x0, y0, x1, y1 = label.get_path().get_extents().extents
    return label.get_transform().transform([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _sat_overlap(p, q):
    """Whether convex polygons p and q (a segment is 2 vertices) overlap."""
    for poly in (p, q):
        for ex, ey in np.roll(poly, -1, axis=0) - poly:
            if ex == 0 and ey == 0:
                continue
            a, b = p @ (-ey, ex), q @ (-ey, ex)
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


def _drawn_marker(ax, marker):
    point = ax.transData.transform(marker.get_offsets()[0])
    return (
        marker.get_paths()[0]
        .transformed(Affine2D(marker.get_transforms()[0]).translate(*point))
        .vertices
    )


def _adjust(ax, add, specs):
    results = [add(ax, m, label_info=li, markerstick_style=st) for m, li, st in specs]
    ax.figure.draw_without_rendering()
    before = [_quad(label).copy() for _m, _s, ls in results for label in ls]
    mmd.adjust_markerstick_labels(ax, results)
    ax.figure.draw_without_rendering()
    labels = [label for _m, _s, ls in results for label in ls]
    moved = [not np.allclose(_quad(lab), b) for lab, b in zip(labels, before)]
    return results, labels, moved


def _overlaps_marker_or_stick(ax, results, label):
    quad = _quad(label)
    for marker, stick, _l in results:
        if _sat_overlap(quad, _drawn_marker(ax, marker)):
            return True
        xy = ax.transData.transform(np.column_stack(stick.get_data()))
        if _sat_overlap(quad, xy):
            return True
    return False


# 12 labeled markersticks packed close together: the rings around a marker run out of spots
CLUSTER_TAYLOR = [
    (
        _taylor_metrics(0.78 + 0.01 * (i % 4), 0.9 + 0.01 * (i // 4), 0.05),
        {"text": f"L{i:02d}"},
        None,
    )
    for i in range(8)
]
CLUSTER_MMD = [
    (_mmd_at(60 + 1.5 * (i % 4), 0.5 + 0.02 * (i // 4)), {"text": f"L{i:02d}"}, None)
    for i in range(8)
]


@pytest.mark.parametrize("diagram", ["taylor", "mmd"])
def test_no_moved_label_overlaps_a_marker_or_stick_in_a_tight_cluster(diagram):
    if diagram == "taylor":
        _fig, ax, _l = taylor.make_fig_axes()
        results, labels, moved = _adjust(ax, taylor.add_markerstick, CLUSTER_TAYLOR)
    else:
        _fig, ax, _l = mmd.make_fig_axes()
        results, labels, moved = _adjust(ax, mmd.add_markerstick, CLUSTER_MMD)
    assert sum(moved) >= 4
    assert not [lab._text for lab in labels if _overlaps_marker_or_stick(ax, results, lab)]
    quads = [_quad(lab) for lab in labels]
    assert not any(_sat_overlap(quads[i], quads[j]) for i in range(8) for j in range(i + 1, 8))


def _inside_sector(ax, label):
    theta, r = ax.transData.inverted().transform(_quad(label)).T
    thetamin, thetamax = ax.get_xlim()
    rel = np.mod(theta - thetamin, 2 * np.pi)
    return bool(np.all(r <= ax.get_ylim()[1]) and np.all(rel <= thetamax - thetamin + 1e-9))


@pytest.mark.parametrize("quadrants", [1, 2])
def test_taylor_labels_end_inside_the_sector(quadrants):
    _fig, ax, _l = taylor.make_fig_axes(quadrants=quadrants)
    specs = [
        (_taylor_metrics(1.45, 0.6, 0.05), {"text": "Near the arc"}, None),  # outside: past r_max
        (_taylor_metrics(0.6, 0.999, 0.05), {"text": "Near the axis"}, None),  # below the axis
    ]
    _results, labels, moved = _adjust(ax, taylor.add_markerstick, specs)
    assert moved == [True, True]
    assert all(_inside_sector(ax, lab) for lab in labels)


def test_mmd_labels_end_inside_the_sector():
    _fig, ax, _l = mmd.make_fig_axes()
    obs = np.arange(20.0)
    on_arc = mmd.calculate_metrics(-obs + 0.01 * np.sin(obs), obs)  # CC ~ -1: r ~ sqrt(2)
    specs = [
        (on_arc, {"text": "Near the arc"}, None),
        (_mmd_at(222, 0.5), {"text": "Into the gap", "position": "beside"}, None),
    ]
    _results, labels, moved = _adjust(ax, mmd.add_markerstick, specs)
    assert moved == [True, True]
    assert all(_inside_sector(ax, lab) for lab in labels)


def test_label_with_no_clear_spot_warns_naming_it():
    _fig, ax, _l = taylor.make_fig_axes()
    specs = [(_taylor_metrics(0.8, 0.9, 0.05), {"text": "Far too big", "fontsize": 120}, None)]
    with pytest.warns(UserWarning, match=r"no clear spot for label\(s\) Far too big"):
        _adjust(ax, taylor.add_markerstick, specs)


# --- leader choice -------------------------------------------------------------------------

from multimetricdiagram import labels as labels_module


def test_shorter_of_two_clean_leaders_wins():
    # a wide label: the spot whose center is farther has the shorter leader (its edge is closer)
    scene = labels_module._Scene([], np.zeros((0, 4)), 0.0, lambda quad: False)
    quad = np.array([(-5.0, -0.5), (5.0, -0.5), (5.0, 0.5), (-5.0, 0.5)])
    anchor = np.array([0.0, 0.0])
    above, right = np.array([0.0, 3.0]), np.array([5.6, 0.0])  # centers 3.0 and 5.6 away
    best = labels_module._best_spot(quad, 0, anchor, scene, [], [above, right])
    assert np.array_equal(best, right)  # leader 0.6 vs 2.5


def test_equal_leaders_take_the_nearest_spot():
    scene = labels_module._Scene([], np.zeros((0, 4)), 0.0, lambda quad: False)
    quad = np.array([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    best = labels_module._best_spot(quad, 0, np.zeros(2), scene, [], [(0, 3.0), (3.0, 0)])
    assert np.array_equal(best, (0, 3.0))


# the labeled marker sits on its own label (position None), between two other markersticks; the
# nearest clear spot's leader line would cross one of their sticks (cases found by search)
STICK_IN_THE_WAY = {
    "taylor": (
        taylor.make_fig_axes,
        taylor.add_markerstick,
        [
            (_taylor_metrics(0.8, 0.9, 0.03), {"text": "Label A", "position": None}, None),
            (_taylor_metrics(0.76, 0.92, 0.3), [], None),
            (_taylor_metrics(0.84, 0.88, -0.3), [], None),
        ],
    ),
    "mmd": (
        mmd.make_fig_axes,
        mmd.add_markerstick,
        [
            (_mmd_at(60, 0.5), {"text": "Label A", "position": None}, None),
            (_mmd_at(62, 0.56, 0.15), [], None),
            (_mmd_at(58, 0.44, -0.15), [], None),
        ],
    ),
}


def _leader_crosses_others(ax, results, leader, own):
    seg = ax.transData.transform(leader.get_xydata())
    for i, (marker, stick, _l) in enumerate(results):
        if i != own and (
            _sat_overlap(seg, _drawn_marker(ax, marker))
            or _sat_overlap(seg, ax.transData.transform(np.column_stack(stick.get_data())))
        ):
            return True
    return False


@pytest.mark.parametrize("diagram", list(STICK_IN_THE_WAY))
def test_clean_leader_preferred_over_nearer_spot(diagram, monkeypatch):
    make, add, specs = STICK_IN_THE_WAY[diagram]
    nearest_crossed = []
    real = labels_module._best_spot

    def spy(quad, owner, anchor, scene, placed, shifts, leaders=()):
        for d in shifts:  # the nearest clear spot, as the search before round 27 took it
            if not labels_module._hits(quad + d, None, scene, placed):
                nearest_crossed.append(
                    labels_module._leader(quad + d, anchor, owner, scene, placed)[0]
                )
                break
        return real(quad, owner, anchor, scene, placed, shifts, leaders)

    monkeypatch.setattr(labels_module, "_best_spot", spy)
    _fig, ax, _l = make()
    results, _labels, moved = _adjust(ax, add, specs)
    leaders = [ln for ln in ax.lines if ln.get_gid() == "multimetricdiagram-leader"]
    assert moved == [True] and len(leaders) == 1
    assert nearest_crossed[0]  # the nearest clear spot's leader would cross a stick or marker
    assert not _leader_crosses_others(ax, results, leaders[0], 0)


# --- margin from a label's own markerstick -------------------------------------------------


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip((p - a) @ ab / max(ab @ ab, 1e-12), 0, 1)
    return float(np.hypot(*(a + t * ab - p)))


def _gap(quad, pts, closed):
    """Smallest distance (px) between a rotated box and a polyline; 0 if they meet."""
    segs = [pts[i : i + 2] for i in range(len(pts) - 1)] + ([pts[[-1, 0]]] if closed else [])
    if any(_sat_overlap(quad, s) for s in segs):
        return 0.0
    edges = [quad[[i, (i + 1) % 4]] for i in range(4)]
    return min(
        min(_seg_dist(p, *s) for p in quad for s in segs),
        min(_seg_dist(p, *e) for p in pts for e in edges),
    )


def _own_gaps(ax, marker, stick, label):
    """(gap to the marker's drawn outline, gap to the stick), each less half its line width."""
    px = ax.figure.dpi / 72
    point = ax.transData.transform(marker.get_offsets()[0])
    path = marker.get_paths()[0].transformed(
        Affine2D(marker.get_transforms()[0]).translate(*point)
    )
    outline = np.concatenate(path.to_polygons(closed_only=False))
    stick_xy = ax.transData.transform(np.column_stack(stick.get_data()))
    quad = _quad(label)
    return (
        _gap(quad, outline, True) - marker.get_linewidths()[0] * px / 2,
        _gap(quad, stick_xy, False) - stick.get_linewidth() * px / 2,
    )


def _capture_margin(monkeypatch):
    margins, real = [], labels_module._scene

    def spy(*args, **kwargs):
        scene = real(*args, **kwargs)
        margins.append(scene.margin)
        return scene

    monkeypatch.setattr(labels_module, "_scene", spy)
    return margins


# labels that must move: on their own marker (position None), alone or among other markersticks
OWN_MARGIN_CASES = {
    "taylor alone": (
        taylor.make_fig_axes,
        taylor.add_markerstick,
        [(_taylor_metrics(0.8, 0.9, 0.2), {"text": "Label A", "position": None}, None)],
    ),
    "mmd alone": (
        mmd.make_fig_axes,
        mmd.add_markerstick,
        [(_mmd_at(60, 0.5, 0.15), {"text": "Label A", "position": None}, None)],
    ),
    "taylor stick in the way": STICK_IN_THE_WAY["taylor"],
    "mmd stick in the way": STICK_IN_THE_WAY["mmd"],
}


@pytest.mark.parametrize("case", list(OWN_MARGIN_CASES))
def test_moved_label_keeps_the_margin_from_its_own_marker_and_stick(case, monkeypatch):
    make, add, specs = OWN_MARGIN_CASES[case]
    margins = _capture_margin(monkeypatch)
    _fig, ax, _l = make()
    results, labels, moved = _adjust(ax, add, specs)
    assert moved == [True] and margins[0] > 0
    marker, stick, _ls = results[0]
    gap_marker, gap_stick = _own_gaps(ax, marker, stick, labels[0])
    assert gap_marker >= margins[0] - 0.01 and gap_stick >= margins[0] - 0.01


# --- leader lines crossing earlier leader lines --------------------------------------------

UNIT_SQUARE = np.array([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
EMPTY_SCENE = labels_module._Scene([], np.zeros((0, 4)), 0.0, lambda quad: False)
NEAR, FAR = np.array([3.0, 0.0]), np.array([0.0, 4.0])  # leaders from (2.5, 0) and (0, 3.5)
ACROSS_NEAR = np.array([(1.5, -1.0), (1.5, 1.0)])  # an earlier leader, across the nearer one


def test_leader_crossing_an_earlier_leader_counts_as_a_crossing():
    def best(leaders):
        shifts = [NEAR, FAR]
        return labels_module._best_spot(
            UNIT_SQUARE, 0, np.zeros(2), EMPTY_SCENE, [], shifts, leaders
        )

    assert np.array_equal(best([]), NEAR)  # the nearest clear spot, its leader clean
    assert np.array_equal(best([(1, ACROSS_NEAR)]), FAR)  # crossing markerstick 1's leader


def test_leaders_of_the_same_markerstick_do_not_count():
    # a markerstick's own leaders meet at its marker
    shifts = [NEAR, FAR]
    leaders = [(0, ACROSS_NEAR)]
    best = labels_module._best_spot(UNIT_SQUARE, 0, np.zeros(2), EMPTY_SCENE, [], shifts, leaders)
    assert np.array_equal(best, NEAR)
    crosses, _length, _start = labels_module._leader(
        UNIT_SQUARE + NEAR, np.zeros(2), 0, EMPTY_SCENE, [], [(1, ACROSS_NEAR)]
    )
    assert crosses  # every start point's line crosses x = 1.5


# --- a leader line must not run along its own stick ----------------------------------------


def _along_then_off_spots(stick):
    """A stand-in for _candidate_shifts(): two ring spots, nearest first: beyond the stick's end
    on its line, then off to the side, farther out; no grid."""

    def spots(bbox, anchor, axes_bbox, _gap_min, _gap_max):
        end = stick.axes.transData.transform(np.column_stack(stick.get_data()))[1]
        u = (end - anchor) / np.hypot(*(end - anchor))
        half = np.array([bbox.width, bbox.height]) / 2
        along = end + u * (0.03 * axes_bbox.width + half[0])  # the stick is horizontal
        off = anchor + np.array([-u[1], u[0]]) * 1.5 * np.hypot(*(along - anchor))
        return np.array([along, off]) - half - (bbox.x0, bbox.y0), np.zeros((0, 2))

    return spots


def _leader_angle_and_gap(ax, result, leader, margin):
    """(angle (deg) between a leader line and its stick, smallest distance from the stick of the
    leader's part beyond the marker's circumscribed extent plus margin, less half the stick's
    width)."""
    marker, stick, _labels = result
    px = ax.figure.dpi / 72
    start, anchor = ax.transData.transform(leader.get_xydata())
    stick_xy = ax.transData.transform(np.column_stack(stick.get_data()))
    d, s = start - anchor, stick_xy[1] - stick_xy[0]
    angle = np.degrees(np.arccos(d @ s / np.hypot(*d) / np.hypot(*s)))
    outline = _drawn_marker(ax, marker)
    reach = np.hypot(*(outline - anchor).T).max() + marker.get_linewidths()[0] * px / 2 + margin
    pts = [anchor + f * d for f in np.linspace(0, 1, 501) if f * np.hypot(*d) >= reach]
    gap = min(_seg_dist(p, *stick_xy) for p in pts) - stick.get_linewidth() * px / 2
    return angle, gap


def test_leader_keeps_clear_of_its_own_stick():
    def place(rule):
        with pytest.MonkeyPatch.context() as mp:
            margins = _capture_margin(mp)
            _fig, ax, _l = mmd.make_fig_axes()
            result = mmd.add_markerstick(
                ax, _mmd_at(90, 0.4, 0.2), label_info={"text": "Label A", "position": None}
            )  # on its marker, so it moves; the stick runs right
            mp.setattr(labels_module, "_candidate_shifts", _along_then_off_spots(result.stick))
            if not rule:
                mp.setattr(labels_module, "_along_own_stick", lambda *args: False)
            ((_label, leader),) = mmd.adjust_markerstick_labels(ax, [result])
            return (*_leader_angle_and_gap(ax, result, leader, margins[0]), margins[0])

    angle, gap, _margin = place(rule=False)  # the nearest clear spot: its leader is on the stick
    assert angle < 1 and gap < 0
    angle, gap, margin = place(rule=True)
    assert angle > 60 and gap >= margin - 0.01
