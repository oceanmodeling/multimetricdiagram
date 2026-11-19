import re

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D

import multimetricdiagram as mmd
from multimetricdiagram.markerstick import MarkerStick

OBS = np.array([1.0, 2.5, 3.0, 4.5, 6.0, 5.5, 7.0, 8.5])
MOD = np.array([1.5, 2.0, 3.8, 4.0, 6.5, 6.2, 6.9, 9.4])  # MB > 0
MOD_NEG = 2 * OBS - MOD  # MB < 0


@pytest.fixture
def polar_ax():
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(projection="polar")
    ax.set_thetamin(-45)
    ax.set_thetamax(225)
    ax.set_ylim(0, np.sqrt(2))
    return ax


@pytest.fixture
def metrics():
    return mmd.calculate_metrics(MOD, OBS)


def _zorders(result):
    marker, stick, labels = result
    return marker.get_zorder(), stick.get_zorder(), [lab.get_zorder() for lab in labels]


TWO_LABELS = [{"text": "A"}, {"text": "B", "position": "inside"}]


def test_returns_a_markerstick_of_the_drawn_artists(polar_ax, metrics):
    result = mmd.add_markerstick(polar_ax, metrics, label_info=TWO_LABELS)
    assert isinstance(result, MarkerStick)
    assert result.marker in polar_ax.collections and result.stick in polar_ax.lines
    assert isinstance(result.labels, list) and len(result.labels) == 2
    assert all(isinstance(lab, PathPatch) and lab in polar_ax.patches for lab in result.labels)
    marker, stick, labels = result
    assert (marker, stick, labels) == (result.marker, result.stick, result.labels)


def test_zorder_flat_applies_to_marker_stick_and_labels(polar_ax, metrics):
    result = mmd.add_markerstick(
        polar_ax, metrics, markerstick_style={"zorder": 5}, label_info=TWO_LABELS
    )
    assert _zorders(result) == (5, 5, [5, 5])


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("vector", [False, True])
def test_accepts_calculate_metrics_output(polar_ax, ddof, vector):
    if vector:
        m = mmd.calculate_metrics(
            np.column_stack([MOD, 0.5 * MOD[::-1]]),
            np.column_stack([OBS, 0.4 * OBS[::-1]]),
            ddof=ddof,
        )
    else:
        m = mmd.calculate_metrics(MOD, OBS, ddof=ddof)
    marker, stick, labels = mmd.add_markerstick(polar_ax, m, label_info={"text": "A"})
    assert marker in polar_ax.collections
    assert stick in polar_ax.lines
    assert len(labels) == 1 and isinstance(labels[0], PathPatch)


INCONSISTENT_MSG = ".*".join(
    re.escape(s)
    for s in (
        "Inconsistent metrics",
        "(RMSD/SDQ)² = (URMSD/SDQ)²/f + (MB/SDQ)²",
        "(URMSD/SDQ)² = 1 - VS*CC",
        "VS = 2*MSD*OSD/(MSD² + OSD²)",
    )
)


@pytest.mark.parametrize("key", ["RMSDonSDQ", "CC", "MSD"])  # one key per consistency check
def test_rejects_inconsistent_metrics(polar_ax, metrics, key):
    metrics[key] *= 0.9
    with pytest.raises(ValueError, match=INCONSISTENT_MSG):
        mmd.add_markerstick(polar_ax, metrics)


def test_rejects_missing_metrics(polar_ax, metrics):
    del metrics["CC"]
    with pytest.raises(ValueError, match="Missing metrics"):
        mmd.add_markerstick(polar_ax, metrics)


def _display_polar(ax, xy_display):
    """(angle in radians, distance in pixels) of a display point about the polar origin."""
    ox, oy = ax.transData.transform((0.0, 0.0))
    dx, dy = xy_display[0] - ox, xy_display[1] - oy
    return np.arctan2(dy, dx), np.hypot(dx, dy)


def _label_and_marker_positions(ax, marker, label):
    ax.get_figure().canvas.draw()
    bb = label.get_window_extent()
    label_pos = _display_polar(ax, ((bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2))
    marker_pos = _display_polar(ax, ax.transData.transform(marker.get_offsets()[0]))
    return label_pos, marker_pos


@pytest.mark.parametrize("position, outward", [("outside", True), ("inside", False)])
def test_label_position_outside_inside(polar_ax, metrics, position, outward):
    marker, _stick, labels = mmd.add_markerstick(
        polar_ax, metrics, label_info={"text": "Lbl", "position": position}
    )
    (_, label_r), (_, marker_r) = _label_and_marker_positions(polar_ax, marker, labels[0])
    assert bool(label_r > marker_r) is outward


@pytest.mark.parametrize("mod, ccw", [(MOD, True), (MOD_NEG, False)], ids=["MB>=0", "MB<0"])
def test_label_position_beside(polar_ax, mod, ccw):
    m = mmd.calculate_metrics(mod, OBS)
    assert bool(m["MBonSDQ"] >= 0) is ccw
    marker, _stick, labels = mmd.add_markerstick(
        polar_ax, m, label_info={"text": "Lbl", "position": "beside"}
    )
    (label_theta, _), (marker_theta, _) = _label_and_marker_positions(polar_ax, marker, labels[0])
    # MB/SDQ >= 0: label counterclockwise of marker; MB/SDQ < 0: clockwise
    assert bool(label_theta > marker_theta) is ccw


def test_label_position_invalid_raises(polar_ax, metrics):
    with pytest.raises(ValueError, match="'position' must be"):
        mmd.add_markerstick(polar_ax, metrics, label_info={"text": "A", "position": "above"})


@pytest.mark.parametrize("label_info", ["A", 3, ({"text": "A"},)], ids=["str", "int", "tuple"])
def test_label_info_wrong_type_raises(polar_ax, metrics, label_info):
    with pytest.raises(TypeError, match="label_info must be"):
        mmd.add_markerstick(polar_ax, metrics, label_info=label_info)


def _drawn_marker(ax, shape, unrotated=False):
    """(marker point, drawn marker path), both in display coordinates; unrotated: the path
    turned back about the point by the marker's rotation (theta - 90 deg)."""
    marker, _stick, _labels = mmd.add_markerstick(
        ax, mmd.calculate_metrics(MOD, OBS), markerstick_style={"marker": shape}
    )
    ax.get_figure().canvas.draw()  # sets the marker's size transform at the figure dpi
    theta = marker.get_offsets()[0][0]
    point = marker.get_offset_transform().transform(marker.get_offsets())[0]
    size_and_offset = Affine2D(marker.get_transforms()[0]).translate(*point)
    if unrotated:
        size_and_offset.rotate_around(*point, np.pi / 2 - theta)
    return point, marker.get_paths()[0].transformed(size_and_offset)


# matplotlib warns that line markers ignore the (default) edge color; not tested here
UNFILLED_WARNING = pytest.mark.filterwarnings("ignore:You passed an? edgecolor:UserWarning")


@UNFILLED_WARNING
@pytest.mark.parametrize("shape", ["o", "s", "D", "+", "x", "|", "_", "1", "2", "3", "4"])
def test_marker_box_centered_on_its_point(polar_ax, shape):
    # closed symmetric markers (area centroid = box center) and unclosed markers (box center)
    point, path = _drawn_marker(polar_ax, shape, unrotated=True)
    ext = path.get_extents()
    center = ((ext.x0 + ext.x1) / 2, (ext.y0 + ext.y1) / 2)
    assert np.hypot(*(np.subtract(center, point))) <= 0.05


# (markerstick_style, expected default label color)
LABEL_COLOR_CASES = [
    (None, "r"),
    ({"color": "b"}, "b"),
    ({"edgecolors": "g"}, "g"),
    ({"marker": {"edgecolors": "g"}}, "g"),
    ({"stick": {"color": "b"}}, "r"),  # follows the marker, not the stick
    ({"edgecolors": "none", "facecolors": "m"}, "m"),
    ({"marker": {"edgecolors": "none", "facecolors": "m"}}, "m"),
    ({"edgecolors": "none", "facecolors": "none"}, "k"),
    # a non-None marker alpha replaces a color's own alpha when drawn, so alpha-0 colors
    # are invisible only with alpha None
    ({"edgecolors": (0, 1, 0, 0), "facecolors": "m"}, (0, 1, 0, 0)),
    ({"alpha": None, "edgecolors": (0, 1, 0, 0), "facecolors": "m"}, "m"),
    ({"alpha": None, "edgecolors": "none", "facecolors": (1, 0, 1, 0)}, "k"),
    (
        {"edgecolors": "face", "facecolors": "m"},
        "m",
    ),  # scatter draws a 'face' edge in the face color
]
LABEL_COLOR_IDS = [
    "default",
    "color",
    "edgecolors",
    "marker-edgecolors",
    "stick-color",
    "edge-none-face-m",
    "marker-edge-none-face-m",
    "edge-none-face-none",
    "edge-alpha0-marker-alpha0.6",
    "edge-alpha0-marker-alphaNone-face-m",
    "edge-none-face-alpha0-marker-alphaNone",
    "edge-face-face-m",
]


def _visible_rgb(rgba_array):
    """RGB of a collection's single color, or None if it is 'none' or fully transparent."""
    if len(rgba_array) == 0 or rgba_array[0][3] == 0:
        return None
    return tuple(rgba_array[0][:3])


@pytest.mark.parametrize("style, expected", LABEL_COLOR_CASES, ids=LABEL_COLOR_IDS)
def test_label_default_color_follows_marker(polar_ax, metrics, style, expected):
    marker, _stick, labels = mmd.add_markerstick(
        polar_ax,
        metrics,
        markerstick_style=style,
        label_info=[{"text": "A"}, {"text": "B", "position": "inside"}],
    )
    for label in labels:
        assert label.get_facecolor() == to_rgba(expected, label.get_alpha())
    # read back from the drawn marker: label matches its edge if visible, else its face if
    # visible, else black
    edge_rgb = _visible_rgb(marker.get_edgecolor())
    face_rgb = _visible_rgb(marker.get_facecolor())
    actual = edge_rgb or face_rgb or to_rgba("k")[:3]
    assert labels[0].get_facecolor()[:3] == pytest.approx(actual)


@pytest.mark.parametrize("style, _expected", LABEL_COLOR_CASES, ids=LABEL_COLOR_IDS)
def test_label_explicit_color_wins(polar_ax, metrics, style, _expected):
    _marker, _stick, labels = mmd.add_markerstick(
        polar_ax, metrics, markerstick_style=style, label_info={"text": "A", "color": "orange"}
    )
    assert labels[0].get_facecolor() == to_rgba("orange", labels[0].get_alpha())


def _handmade_metrics(msd, osd, cc, mb_on_sdq=0.2, n=50):
    """Self-consistent metrics dict (ddof=1) with the given MSD, OSD, CC and MB/SDQ."""
    vs = 2 * msd * osd / (msd**2 + osd**2)
    urmsd_on_sdq = np.sqrt(1 - vs * cc)
    return {
        "RMSDonSDQ": np.sqrt(urmsd_on_sdq**2 * (n - 1) / n + mb_on_sdq**2),
        "URMSDonSDQ": urmsd_on_sdq,
        "MBonSDQ": mb_on_sdq,
        "CC": cc,
        "VS": vs,
        "MSD": msd,
        "OSD": osd,
        "n": n,
        "ddof": 1,
    }


# (MSD, OSD, CC) with MSD > OSD -> (expected r = URMSD/SDQ, expected theta in degrees);
# VS = 0.8 for MSD, OSD = 2, 1
PLACEMENT_CASES = [
    ((2.0, 1.0, 0.9), (np.sqrt(0.28), -45 + 135 * np.log(0.9) / np.log(0.72))),  # r < 1
    ((2.0, 1.0, 0.0), (1.0, -45 + 135 * 0.8)),  # r == 1: theta from VS
    ((2.0, 1.0, -0.5), (np.sqrt(1.4), -45 + 135 * np.log(0.5) / np.log(0.4))),  # 1 < r < sqrt(2)
    ((1.0, 1.0, 1.0), (0.0, 90.0)),  # r == 0
    ((1.0, 1.0, -1.0), (np.sqrt(2), 90.0)),  # r == sqrt(2)
]
PLACEMENT_IDS = ["r<1", "r==1", "r>1", "r==0", "r==sqrt2"]


def _marker_polar(marker):
    theta, r = marker.get_offsets()[0]
    return np.rad2deg(theta), r


@pytest.mark.parametrize("msd_osd_cc, expected", PLACEMENT_CASES, ids=PLACEMENT_IDS)
def test_marker_placement_follows_formulas(polar_ax, msd_osd_cc, expected):
    marker, _stick, _labels = mmd.add_markerstick(polar_ax, _handmade_metrics(*msd_osd_cc))
    theta_deg, r = _marker_polar(marker)
    assert r == pytest.approx(expected[0], abs=1e-12)
    assert theta_deg == pytest.approx(expected[1], abs=1e-9)


@pytest.mark.parametrize("cc", [0.9, 0.0, -0.5], ids=["r<1", "r==1", "r>1"])
def test_msd_less_than_osd_mirrors_into_left_half(polar_ax, cc):
    right, _s, _l = mmd.add_markerstick(polar_ax, _handmade_metrics(2.0, 1.0, cc))
    left, _s, _l = mmd.add_markerstick(polar_ax, _handmade_metrics(1.0, 2.0, cc))
    theta_right, r_right = _marker_polar(right)
    theta_left, r_left = _marker_polar(left)
    assert theta_right < 90 < theta_left
    assert theta_left == pytest.approx(225 - (theta_right + 45), abs=1e-9)
    assert r_left == pytest.approx(r_right, abs=1e-12)


def _metrics_at(r, cc, msd, osd, mb_on_sdq=0.2, n=50):
    """_handmade_metrics() with URMSD/SDQ set separately, e.g. as rounded near r = 1."""
    m = _handmade_metrics(msd, osd, cc, mb_on_sdq, n)
    m["URMSDonSDQ"] = r
    m["RMSDonSDQ"] = np.sqrt(r**2 * (n - 1) / n + mb_on_sdq**2)
    return m


def _mirrored_if(theta_deg, msd, osd):
    return theta_deg if msd > osd else 225 - (theta_deg + 45)


MSD_OSD = pytest.mark.parametrize("msd, osd", [(2.0, 1.0), (1.0, 2.0)], ids=["MSD>OSD", "MSD<OSD"])
R_BELOW = 1 - 1e-16  # 0.9999999999999999, the float just below 1
R_ABOVE = float(np.nextafter(1.0, 2.0))  # 1 + 2.2e-16 (1 + 1e-16 rounds to exactly 1.0)
# (CC, URMSD/SDQ): each sign combination rounding can give near the CC = 0 ring; VS = 0.8
NEAR_RING = {
    "CC>0 r<1": (1e-17, R_BELOW),
    "CC<0 r>1": (-1e-17, R_ABOVE),
    "CC=0 r<1": (0.0, R_BELOW),
    "CC=0 r>1": (0.0, R_ABOVE),
    "CC<0 r<1": (-1e-17, R_BELOW),  # CC's sign inconsistent with r
    "CC>0 r>1": (1e-17, R_ABOVE),
    "1-r^2=1e-13": (1e-13 / 0.8, np.sqrt(1 - 1e-13)),  # inside the 1e-12 tolerance
}


@MSD_OSD
@pytest.mark.parametrize("cc, r", list(NEAR_RING.values()), ids=list(NEAR_RING))
def test_marker_on_cc0_ring_placed_by_vs(polar_ax, cc, r, msd, osd):
    m = _metrics_at(r, cc, msd, osd)
    theta_deg, r_drawn = _marker_polar(mmd.add_markerstick(polar_ax, m)[0])
    assert np.isfinite(theta_deg) and r_drawn == r
    assert theta_deg == pytest.approx(_mirrored_if(-45 + 135 * 0.8, msd, osd), abs=1e-9)


def _skip_unless_sign_combination(m, case):
    """Skip, not fail, if rounding here (numpy version, platform) gives another combination."""
    cc_sign = "CC>0" if m["CC"] > 0 else "CC<0" if m["CC"] < 0 else "CC=0"
    r_side = "r<1" if m["URMSDonSDQ"] < 1 else "r>1" if m["URMSDonSDQ"] > 1 else "r=1"
    if f"{cc_sign} {r_side}" != case:
        pytest.skip(
            f"precondition not met: this input rounds to {cc_sign} {r_side}, not {case} "
            f"(CC={m['CC']:.2e}, URMSD/SDQ - 1={m['URMSDonSDQ'] - 1:.2e})"
        )


def test_skip_unless_sign_combination_skips_rather_than_fails():
    m = {"CC": 0.0, "URMSDonSDQ": 1.0}
    with pytest.raises(pytest.skip.Exception, match="precondition not met: .* CC=0 r=1, not CC>0"):
        _skip_unless_sign_combination(m, "CC>0 r<1")
    _skip_unless_sign_combination(m, "CC=0 r=1")  # matching: no skip


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # log of 0 or of a negative number
@pytest.mark.parametrize(
    "cc, r",
    [(-1e-9, np.sqrt(1 - 2e-10)), (0.0, np.sqrt(1 - 2e-10)), (1e-9, np.sqrt(1 + 2e-10))],
    ids=["CC<0 r<1", "CC=0 r<1", "CC>0 r>1"],
)
def test_non_finite_theta_raises(polar_ax, cc, r):
    # outside the ring tolerance, yet within the identity check's np.isclose tolerance
    with pytest.raises(ValueError, match="marker angle is not finite"):
        mmd.add_markerstick(polar_ax, _metrics_at(r, cc, 2.0, 1.0))
    assert not polar_ax.lines and not polar_ax.collections  # nothing drawn


@pytest.mark.parametrize("mb_on_sdq", [0.3, -0.3], ids=["MB>0", "MB<0"])
def test_stick_end_point(polar_ax, mb_on_sdq):
    marker, stick, _labels = mmd.add_markerstick(
        polar_ax, _handmade_metrics(2.0, 1.0, 0.9, mb_on_sdq)
    )
    thetas, rs = stick.get_data()
    theta, r = marker.get_offsets()[0]
    assert (thetas[0], rs[0]) == pytest.approx((theta, r))  # starts at the marker
    # tangent at the marker, length |MB/SDQ|, clockwise for MB/SDQ > 0
    expected_end = (
        r * np.cos(theta) + mb_on_sdq * np.sin(theta),
        r * np.sin(theta) - mb_on_sdq * np.cos(theta),
    )
    end = (rs[1] * np.cos(thetas[1]), rs[1] * np.sin(thetas[1]))
    assert end == pytest.approx(expected_end, abs=1e-12)
    assert rs[1] == pytest.approx(np.hypot(r, mb_on_sdq))
    assert bool(thetas[1] < theta) is (mb_on_sdq > 0)


@pytest.fixture
def _defaults():
    """The shared-defaults module (imported here so only tests using it depend on it)."""
    from multimetricdiagram import _defaults

    return _defaults


@pytest.mark.parametrize("bad", ["sample(1)", 2, None])
def test_rejects_invalid_ddof(polar_ax, metrics, bad):
    metrics["ddof"] = bad
    with pytest.raises(ValueError, match=r"metrics\['ddof'\] must be 0 or 1"):
        mmd.add_markerstick(polar_ax, metrics)


@pytest.mark.parametrize(
    "value",
    [12, 12.5, np.int64(7), np.float64(12.5), None, ""],
    ids=["int", "float", "np-int64", "np-float64", "None", "empty"],
)
def test_label_text_converted_with_str(polar_ax, metrics, value):
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    _marker, _stick, labels = mmd.add_markerstick(polar_ax, metrics, label_info={"text": value})
    polar_ax.get_figure().canvas.draw()
    path = labels[0].get_path()
    expected = "" if value is None else str(value)
    if expected:
        reference = TextPath((0, 0), expected, size=1, prop=FontProperties(weight="normal"))
        assert np.array_equal(path.vertices, reference.vertices)
    else:
        assert len(path.vertices) == 0


def _frame_extents_pt(ax, marker, path_px):
    """(x0, x1, y0, y1) in points of a display path in the marker's frame: origin at the marker
    point, +y outward, +x clockwise."""
    theta, r = marker.get_offsets()[0]
    to_frame = (
        Affine2D()
        .translate(*(-ax.transData.transform((theta, r))))
        .rotate(np.pi / 2 - theta)
        .scale(72 / ax.figure.dpi)
    )
    ext = path_px.transformed(to_frame).get_extents()
    return np.array([ext.x0, ext.x1, ext.y0, ext.y1])


def _label_gaps_pt(ax, marker, labels):
    """Per label, gaps (points) from the marker's drawn edge to the label's drawn (ink) box:
    (outward, inward, clockwise, counterclockwise)."""
    ax.figure.canvas.draw()
    point = ax.transData.transform(marker.get_offsets()[0])
    size_and_offset = Affine2D(marker.get_transforms()[0]).translate(*point)
    edge = marker.get_linewidths()[0] / 2
    m = _frame_extents_pt(ax, marker, marker.get_paths()[0].transformed(size_and_offset))
    m += np.array([-edge, edge, -edge, edge])
    gaps = []
    for label in labels:
        lab = _frame_extents_pt(ax, marker, label.get_path().transformed(label.get_transform()))
        gaps.append((lab[2] - m[3], m[2] - lab[3], lab[0] - m[1], m[0] - lab[1]))
    return np.array(gaps)


PRESET_LABELS = [
    {"text": text, "position": p}
    for text, p in (
        ("Lbl", "outside"),
        ("Lbl", "inside"),
        ("Lbl", "beside"),
        ("Long label", "beside"),
    )
]


def _given_offset_labels(mb_on_sdq_negative):
    beside = 0.4 if mb_on_sdq_negative else -0.4  # opposite the stick, as the preset
    return [
        {"text": "Lbl", "position": "outside", "out_offset": 0.4},
        {"text": "Lbl", "position": "inside", "out_offset": -0.4},
        {"text": "Lbl", "position": "beside", "cw_offset": beside},
        {"text": "Long label", "position": "beside", "cw_offset": beside},
    ]


@pytest.mark.parametrize("given", [False, True], ids=["preset", "given"])
@pytest.mark.parametrize("mod", [MOD, MOD_NEG], ids=["MB>=0", "MB<0"])
@pytest.mark.parametrize("size_style", [{"markersize": 8, "linewidths": 3.0}], ids=["8pt"])
@pytest.mark.parametrize("shape", ["o", "^"])
def test_label_gap_from_marker_edge(polar_ax, shape, size_style, mod, given):
    m = mmd.calculate_metrics(mod, OBS)
    negative = m["MBonSDQ"] < 0
    labels_info = _given_offset_labels(negative) if given else PRESET_LABELS
    marker, _stick, labels = mmd.add_markerstick(
        polar_ax, m, markerstick_style={"marker": shape, **size_style}, label_info=labels_info
    )
    gaps = _label_gaps_pt(polar_ax, marker, labels)
    expected = (0.4 if given else 0.1) * 10  # offset x fontsize
    beside = 2 if negative else 3  # clockwise gap if MB/SDQ < 0
    measured = [gaps[0, 0], gaps[1, 1], gaps[2, beside], gaps[3, beside]]
    assert measured == pytest.approx([expected] * 4, abs=0.1)
