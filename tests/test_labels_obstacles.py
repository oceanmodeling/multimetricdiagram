import numpy as np
import pytest
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

import multimetricdiagram as mmd
from multimetricdiagram._text import _PolarTextPatch
from multimetricdiagram.alt import taylor


def _text_boxes(ax, text):
    ref = TextPath((0, 0), text, size=1, prop=FontProperties(weight="normal")).vertices
    return [
        p.get_window_extent()
        for p in ax.patches
        if isinstance(p, _PolarTextPatch) and np.array_equal(p.get_path().vertices, ref)
    ]


def _star_box(ax):
    (star,) = [ln for ln in ax.lines if ln.get_marker() == "*"]
    return star.get_window_extent()


def _mmd_at(theta_deg, r, mb_on_sdq=0.03):
    """Self-consistent top-level metrics dict (ddof=0) putting the marker at (theta_deg, r),
    r < 1 (inverts add_markerstick()'s placement formulas)."""
    mirrored = theta_deg > 90  # MSD < OSD
    th = 180 - theta_deg if mirrored else theta_deg
    cc = (1 - r * r) ** ((th + 45) / 135)
    vs = (1 - r * r) / cc
    k = (1 + np.sqrt(1 - vs * vs)) / vs  # MSD/OSD >= 1 with this VS
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


def _taylor_metrics(msd, cc, mb, osd=1.0):
    urmsd = np.sqrt(msd**2 + osd**2 - 2 * msd * osd * cc)
    rmsd = np.hypot(urmsd, mb)
    return {
        "MSD": msd,
        "OSD": osd,
        "CC": cc,
        "RMSD": rmsd,
        "URMSD": urmsd,
        "MB": mb,
        "n": 50,
        "ddof": 0,
        "MSDonOSD": msd / osd,
        "RMSDonOSD": rmsd / osd,
        "URMSDonOSD": urmsd / osd,
        "MBonOSD": mb / osd,
    }


def _adjusted_labels(ax, add, metrics_list, text, style=None):
    results = [
        add(ax, m, markerstick_style=style, label_info={"text": text}) for m in metrics_list
    ]
    mmd.adjust_markerstick_labels(ax, results)
    ax.figure.canvas.draw()
    return [label for _m, _s, labels in results for label in labels]


def _overlaps(label, bbox):
    """Whether a label's drawn (rotated) ink box meets bbox, by the separating axis test."""
    x0, y0, x1, y1 = label.get_path().get_extents().extents
    quad = label.get_transform().transform([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    box = np.array(
        [(bbox.x0, bbox.y0), (bbox.x1, bbox.y0), (bbox.x1, bbox.y1), (bbox.x0, bbox.y1)]
    )
    for poly in (quad, box):
        for ex, ey in np.roll(poly, -1, axis=0) - poly:
            a, b = quad @ (-ey, ex), box @ (-ey, ex)
            if a.max() < b.min() or b.max() < a.min():
                return False
    return True


def test_mmd_label_of_marker_at_star_avoids_star():
    _fig, ax, _l = mmd.make_fig_axes()
    m = mmd.calculate_metrics(np.arange(10.0) + 0.1, np.arange(10.0))  # URMSD = 0
    assert m["URMSDonSDQ"] == pytest.approx(0, abs=1e-12)
    labels = _adjusted_labels(ax, mmd.add_markerstick, [m], "A label", {"markersize": 4})
    assert not _overlaps(labels[0], _star_box(ax))


def test_mmd_labels_avoid_r_tick_labels_and_sector_text():
    _fig, ax, _l = mmd.make_fig_axes()
    metrics = [_mmd_at(theta, r) for theta, r in ((2, 0.55), (6, 0.75), (-10, 0.35))]
    labels = _adjusted_labels(ax, mmd.add_markerstick, metrics, "Label")
    ticks = [t.get_window_extent() for t in ax.yaxis.get_ticklabels() if t.get_text()]
    sector = [t.get_window_extent() for t in ax.texts]
    assert ticks and len(sector) == 3
    assert not any(_overlaps(label, o) for label in labels for o in ticks + sector)


@pytest.mark.parametrize("quadrants", [1, 2])
def test_taylor_labels_avoid_axis_titles_and_tick_labels(quadrants):
    _fig, ax, _l = taylor.make_fig_axes(quadrants=quadrants)
    # markers just above the right axis, whose labels the search could put below it
    metrics = [_taylor_metrics(msd, 0.999, 0.05) for msd in (0.45, 0.62, 1.3)]
    labels = _adjusted_labels(ax, taylor.add_markerstick, metrics, "A long label")
    fixed = _text_boxes(ax, "Solid contours: RMSD/OSD (stick tip), URMSD/OSD (marker)")
    fixed += _text_boxes(ax, "Dashed contours: MSD/OSD (marker)")
    fixed += [b for v in ("0.25", "0.50", "0.75") for b in _text_boxes(ax, v)]
    assert len(fixed) >= 5
    assert not any(_overlaps(label, o) for label in labels for o in fixed)


def test_user_tagged_artist_is_avoided():
    _fig, ax, _l = taylor.make_fig_axes()
    m = _taylor_metrics(0.8, 0.95, 0.05)
    theta, r = np.arccos(0.95), 0.8
    # a box-like text right where the label would go, tagged as an obstacle
    note = ax.text(theta, r + 0.08, "NOTE NOTE", ha="center", gid="multimetricdiagram-obstacle")
    labels = _adjusted_labels(ax, taylor.add_markerstick, [m], "Model")
    assert not _overlaps(labels[0], note.get_window_extent())
