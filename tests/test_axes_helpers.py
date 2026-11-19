import matplotlib.pyplot as plt
import numpy as np
import pytest

import multimetricdiagram as mmd
from multimetricdiagram.axes import _cc, _fmt_tick_k, _vd

OBS = np.array([1.0, 2.5, 3.0, 4.5, 6.0, 5.5, 7.0, 8.5])
NOISE = np.array([0.3, -0.2, 0.1, -0.4, 0.2, 0.0, -0.1, 0.3])
OBS_V = np.column_stack([OBS, 0.4 * OBS[::-1]])


def _vec(a, b):
    return np.column_stack([a, b])


# name -> (model, observation, expected regime: "<1", "==1" or ">1", MSD > OSD?)
CASES = {
    "scalar r<1 MSD>OSD": (1.3 * OBS + NOISE, OBS, "<1", True),
    "scalar r<1 MSD<OSD": (0.6 * OBS + NOISE, OBS, "<1", False),
    "scalar r>1 MSD>OSD": (-1.3 * OBS + NOISE + 20, OBS, ">1", True),
    "scalar r>1 MSD<OSD": (-0.6 * OBS + NOISE + 10, OBS, ">1", False),
    "vector r<1 MSD>OSD": (_vec(1.3 * OBS + NOISE, 0.5 * OBS[::-1]), OBS_V, "<1", True),
    "vector r<1 MSD<OSD": (_vec(0.6 * OBS + NOISE, 0.3 * OBS[::-1]), OBS_V, "<1", False),
    "vector r>1 MSD>OSD": (_vec(-1.3 * OBS + NOISE + 20, -0.5 * OBS[::-1] + 9), OBS_V, ">1", True),
    "vector r>1 MSD<OSD": (
        _vec(-0.6 * OBS + NOISE + 10, -0.3 * OBS[::-1] + 5),
        OBS_V,
        ">1",
        False,
    ),
    # orthogonal anomalies: CC == 0 exactly and URMSD/SDQ == 1.0 exactly
    "scalar r==1 MSD>OSD": (
        np.array([2.0, 2.0, -2.0, -2.0]),
        np.array([1.0, -1.0, 1.0, -1.0]),
        "==1",
        True,
    ),
    "scalar r==1 MSD<OSD": (
        np.array([1.0, -1.0, 1.0, -1.0]),
        np.array([2.0, 2.0, -2.0, -2.0]),
        "==1",
        False,
    ),
}
OFF_UNIT_CIRCLE = [name for name in CASES if CASES[name][2] != "==1"]


def _marker_position_and_metrics(name):
    model, observation, regime, msd_above = CASES[name]
    m = mmd.calculate_metrics(model, observation)
    r_expected = {
        "<1": m["URMSDonSDQ"] < 1,
        "==1": m["URMSDonSDQ"] == 1,
        ">1": m["URMSDonSDQ"] > 1,
    }
    assert r_expected[regime] and (m["MSD"] > m["OSD"]) is msd_above  # the case is what it says
    ax = plt.figure().add_subplot(projection="polar")
    marker, _stick, _labels = mmd.add_markerstick(ax, m)
    theta, r = marker.get_offsets()[0]
    return np.rad2deg(theta), r, m


@pytest.mark.parametrize("name", list(CASES))
def test_cc_mapping_matches_marker_cc(name):
    theta_deg, r, m = _marker_position_and_metrics(name)
    assert float(_cc(theta_deg, r)) == pytest.approx(m["CC"], abs=1e-9)


@pytest.mark.parametrize("name", OFF_UNIT_CIRCLE)  # r == 1 is a singularity of _vd(), see below
def test_vd_mapping_matches_marker_vd(name):
    theta_deg, r, m = _marker_position_and_metrics(name)
    assert float(_vd(theta_deg, r)) == pytest.approx(1 - m["VS"], abs=1e-9)  # |VD| = 1 - VS


def _ring_theta_deg(m):
    """Marker angle on the CC = 0 ring: -45 + 135*VS, mirrored if MSD < OSD."""
    theta = -45 + 135 * m["VS"]
    return theta if m["MSD"] >= m["OSD"] else 225 - (theta + 45)


def test_constant_vector_model_marker_is_finite():
    model = np.tile([0.3, -0.2], (10, 1))
    observation = np.random.default_rng(0).normal(0.0, 0.1, (10, 2))
    m = mmd.calculate_metrics(model, observation)
    assert abs(m["CC"]) < 1e-15  # CC ~ -1.6e-17, URMSD/SDQ = 1 - 1.1e-16: signs inconsistent
    ax = plt.figure().add_subplot(projection="polar")
    marker, _stick, _labels = mmd.add_markerstick(ax, m)
    theta, r = np.ma.filled(marker.get_offsets(), np.nan)[0]
    assert np.isfinite(theta) and np.isfinite(r)
    assert np.rad2deg(theta) == pytest.approx(_ring_theta_deg(m), abs=1e-9)  # 225 (VS = 0)


@pytest.mark.parametrize(
    "k, label",
    [
        (np.inf, "∞"),
        (1.0, "1.00"),
        (0.382, "0.38"),
        (0.5, "0.50"),
        (2.618, "2.62"),
        (10.43, "10.4"),
        (12.34, "12.3"),
        (123.4, "123"),
        (1234.4, "1234"),
    ],
)
def test_fmt_tick_k(k, label):
    assert _fmt_tick_k(k) == label
