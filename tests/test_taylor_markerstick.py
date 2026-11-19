import warnings

import numpy as np
import pytest

import multimetricdiagram as mmd
from multimetricdiagram.alt import taylor
from multimetricdiagram.markerstick import MarkerStick

OBS = np.array([1.0, 2.5, 3.0, 4.5, 6.0, 5.5, 7.0, 8.5])
NOISE = np.array([0.3, -0.2, 0.1, -0.4, 0.2, 0.0, -0.1, 0.3])
UNNORMALIZED = {"normalized": False, "units": "m/s", "osd": 2.0, "r_max": 5.0}


def _tm(msd, osd, cc, mb, n=50, ddof=0):
    """Self-consistent taylor.calculate_metrics()-style dict."""
    urmsd = np.sqrt(msd**2 + osd**2 - 2 * msd * osd * cc)
    rmsd = np.sqrt(urmsd**2 * (n - ddof) / n + mb**2)
    return {
        "MSD": msd,
        "OSD": osd,
        "CC": cc,
        "RMSD": rmsd,
        "URMSD": urmsd,
        "MB": mb,
        "n": n,
        "ddof": ddof,
        "MSDonOSD": msd / osd,
        "RMSDonOSD": rmsd / osd,
        "URMSDonOSD": urmsd / osd,
        "MBonOSD": mb / osd,
    }


def _xy(theta, r):
    return np.array([r * np.cos(theta), r * np.sin(theta)])


def _marker_and_stick_end(result):
    marker, stick, _labels = result
    (t0, t1), (r0, r1) = stick.get_data()
    assert tuple(marker.get_offsets()[0]) == (t0, r0)  # the stick starts at the marker
    return _xy(t0, r0), _xy(t1, r1)


def test_returns_a_markerstick_of_the_drawn_artists():
    _fig, ax, _l = taylor.make_fig_axes()
    result = taylor.add_markerstick(ax, _tm(1.2, 2.0, 0.8, 0.3), label_info={"text": "A"})
    assert isinstance(result, MarkerStick)
    assert result.marker in ax.collections and result.stick in ax.lines
    assert isinstance(result.labels, list) and len(result.labels) == 1
    assert result.labels[0] in ax.patches
    marker, stick, labels = result
    assert (marker, stick, labels) == (result.marker, result.stick, result.labels)


@pytest.mark.parametrize("options", [{}, UNNORMALIZED], ids=["normalized", "unnormalized"])
def test_marker_at_msd_and_arccos_cc(options):
    _fig, ax, _l = taylor.make_fig_axes(**options)
    marker, _s, _l = taylor.add_markerstick(ax, _tm(1.2, 2.0, 0.8, 0.3))
    theta, r = marker.get_offsets()[0]
    assert theta == pytest.approx(np.arccos(0.8), abs=1e-12)
    assert r == pytest.approx(1.2 if options else 0.6, abs=1e-12)


# (MSD, CC) on an OSD = 1 diagram: off the star, above it, below and left of it
POSITIONS = {"above star": (1.25, 0.8), "inside": (0.7, 0.9), "outside": (1.4, 0.6)}


@pytest.mark.parametrize("quadrants", [2])
@pytest.mark.parametrize("mb", [0.3, -0.3], ids=["MB>0", "MB<0"])
@pytest.mark.parametrize("msd, cc", list(POSITIONS.values()), ids=list(POSITIONS))
def test_stick_tangent_to_star_circle_clockwise_for_positive_mb(quadrants, msd, cc, mb):
    _fig, ax, _l = taylor.make_fig_axes(quadrants=quadrants)
    p, end = _marker_and_stick_end(taylor.add_markerstick(ax, _tm(msd, 1.0, cc, mb)))
    radial, stick = p - (1.0, 0.0), end - p
    assert np.dot(radial, stick) == pytest.approx(0, abs=1e-12)  # tangent
    assert np.hypot(*stick) == pytest.approx(abs(mb), abs=1e-12)
    cross = radial[0] * stick[1] - radial[1] * stick[0]
    assert bool(cross < 0) is (mb > 0)  # clockwise about the star if MB > 0


@pytest.mark.parametrize("options", [{}, UNNORMALIZED], ids=["normalized", "unnormalized"])
@pytest.mark.parametrize("mb", [0.4, -0.4], ids=["MB>0", "MB<0"])
def test_marker_at_star_has_horizontal_stick(options, mb):
    _fig, ax, _l = taylor.make_fig_axes(**options)
    osd = options.get("osd", 1.0)
    p, end = _marker_and_stick_end(taylor.add_markerstick(ax, _tm(osd, osd, 1.0, mb * osd)))
    assert p == pytest.approx([osd, 0], abs=1e-12)
    assert end == pytest.approx([osd + mb * osd if options else osd + mb, 0], abs=1e-12)


MODELS = {
    "scalar": (1.3 * OBS + NOISE + 0.5, OBS),
    "scalar MB<0": (0.6 * OBS + NOISE - 1.0, OBS),
    "scalar CC<0": (-0.8 * OBS + NOISE + 12, OBS),
    "vector": (
        np.column_stack([1.2 * OBS + NOISE, 0.3 * OBS[::-1] + 0.4]),
        np.column_stack([OBS, 0.4 * OBS[::-1]]),
    ),
}


@pytest.mark.parametrize("normalized", [True, False])
@pytest.mark.parametrize("model, observation", list(MODELS.values()), ids=list(MODELS))
def test_stick_end_on_rmsd_circle_with_ddof_0(model, observation, normalized):
    m = taylor.calculate_metrics(model, observation)
    options = {} if normalized else {"normalized": False, "units": "u", "osd": m["OSD"]}
    if not normalized:
        options["r_max"] = 3 * m["OSD"]
    _fig, ax, _l = taylor.make_fig_axes(quadrants=2, **options)
    _p, end = _marker_and_stick_end(taylor.add_markerstick(ax, m))
    star = 1.0 if normalized else m["OSD"]
    rmsd = m["RMSDonOSD"] if normalized else m["RMSD"]
    assert np.hypot(end[0] - star, end[1]) == pytest.approx(rmsd, abs=1e-9)


def test_negative_cc_on_one_quadrant_raises():
    _fig, ax, _l = taylor.make_fig_axes()
    with pytest.raises(ValueError, match=r"CC=-0.5 < 0 .* use make_fig_axes\(quadrants=2\)"):
        taylor.add_markerstick(ax, _tm(1.0, 1.0, -0.5, 0.1))
    assert not ax.collections
    _fig, ax, _l = taylor.make_fig_axes(quadrants=2)
    marker, _s, _l = taylor.add_markerstick(ax, _tm(1.0, 1.0, -0.5, 0.1))
    assert marker.get_offsets()[0][0] == pytest.approx(np.arccos(-0.5))


def test_osd_mismatch_raises():
    _fig, ax, _l = taylor.make_fig_axes(**UNNORMALIZED)
    taylor.add_markerstick(ax, _tm(1.0, 2.0 * (1 + 1e-9), 0.8, 0.1))  # within np.isclose
    with pytest.raises(ValueError, match=r"metrics\['OSD'\]=2.5 differs from the diagram's"):
        taylor.add_markerstick(ax, _tm(1.0, 2.5, 0.8, 0.1))


def test_axes_not_from_taylor_raises():
    _fig, ax, _l = mmd.make_fig_axes()
    with pytest.raises(ValueError, match=r"taylor.make_fig_axes\(\)"):
        taylor.add_markerstick(ax, _tm(1.0, 1.0, 0.8, 0.1))


def test_missing_metric_raises():
    _fig, ax, _l = taylor.make_fig_axes()
    m = _tm(1.0, 1.0, 0.8, 0.1)
    del m["URMSDonOSD"]
    with pytest.raises(ValueError, match=r"Missing metrics: \{'URMSDonOSD'\}"):
        taylor.add_markerstick(ax, m)


@pytest.mark.parametrize(
    "options, msd, osd, warns",
    [
        ({}, 1.6, 1.0, True),  # normalized, r_max 1.5
        ({}, 1.5, 1.0, False),  # exactly at r_max
        ({}, 1.2, 1.0, False),
        (UNNORMALIZED, 5.5, 2.0, True),  # unnormalized, r_max 5
        (UNNORMALIZED, 5.0, 2.0, False),
        (UNNORMALIZED, 3.0, 2.0, False),
    ],
    ids=["norm-beyond", "norm-at", "norm-inside", "data-beyond", "data-at", "data-inside"],
)
def test_marker_beyond_r_max_warns(options, msd, osd, warns):
    _fig, ax, _l = taylor.make_fig_axes(**options)
    m = _tm(msd, osd, 0.9, 0.1)
    if warns:
        with pytest.warns(
            UserWarning, match=r"beyond r_max=.*re-run make_fig_axes\(\) and pass a larger r_max"
        ):
            marker, _s, _l = taylor.add_markerstick(ax, m)
        assert marker in ax.collections  # still drawn
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            taylor.add_markerstick(ax, m)


def test_patternonly_metrics_on_patternonly_diagram_draw_no_stick():
    m = taylor.calculate_metrics(OBS + NOISE, OBS, patternonly=True)
    _fig, ax, _ = taylor.make_fig_axes(metrics=[m], patternonly=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _marker, stick, _labels = taylor.add_markerstick(ax, m)
    x, y = stick.get_data()
    assert x[0] == pytest.approx(x[1]) and y[0] == pytest.approx(y[1])


def test_biased_metrics_on_patternonly_diagram_raise():
    m = taylor.calculate_metrics(OBS + NOISE + 0.5, OBS)
    with pytest.raises(ValueError, match=r"patternonly=True"):
        taylor.make_fig_axes(metrics=[m], patternonly=True)
    _fig, ax, _ = taylor.make_fig_axes(patternonly=True)
    with pytest.raises(ValueError, match=r"pattern-only"):
        taylor.add_markerstick(ax, m)


def test_patternonly_metrics_on_default_diagram_warn():
    m = taylor.calculate_metrics(OBS + NOISE, OBS, patternonly=True)
    _fig, ax, _ = taylor.make_fig_axes()
    with pytest.warns(UserWarning, match=r"make_fig_axes\(patternonly=True\)"):
        taylor.add_markerstick(ax, m)
