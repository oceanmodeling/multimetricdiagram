import numpy as np
import pytest

import multimetricdiagram as mmd
from multimetricdiagram.alt import taylor

OBS = np.array([1.0, 2.5, 3.0, 4.5, 6.0, 5.5, 7.0, 8.5])
MOD = np.array([1.5, 2.0, 3.8, 4.0, 6.5, 6.2, 6.9, 9.4])
OBS_V = np.column_stack([OBS, 0.4 * OBS[::-1]])
MOD_V = np.column_stack([MOD, 0.5 * MOD[::-1]])
SHARED = ("MSD", "OSD", "CC", "RMSD", "URMSD", "MB", "n", "ddof", "MSDonOSD")


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("vector", [False, True], ids=["scalar", "vector"])
def test_values_equal_top_level_and_ratios_to_osd(ddof, vector):
    model, observation = (MOD_V, OBS_V) if vector else (MOD, OBS)
    top = mmd.calculate_metrics(model, observation, ddof=ddof)
    m = taylor.calculate_metrics(model, observation, ddof=ddof)
    assert set(m) == set(SHARED) | {"RMSDonOSD", "URMSDonOSD", "MBonOSD", "period"}
    for key in SHARED:
        assert m[key] == top[key]
    for key in ("RMSD", "URMSD", "MB"):
        assert m[f"{key}onOSD"] == top[key] / top["OSD"]
    assert m["MSDonOSD"] == top["MSD"] / top["OSD"]


def test_constant_observation_raises():
    mmd.calculate_metrics(MOD, np.full_like(OBS, 3.0))  # allowed at the top level
    with pytest.raises(ValueError, match=r"OSD = 0"):
        taylor.calculate_metrics(MOD, np.full_like(OBS, 3.0))


@pytest.mark.parametrize("ddof", [0, 1])
def test_patternonly_zero_bias(ddof):
    m = taylor.calculate_metrics(MOD, OBS, ddof=ddof)
    p = taylor.calculate_metrics(MOD, OBS, ddof=ddof, patternonly=True)
    assert m["MB"] != 0
    assert p["MB"] == p["MBonOSD"] == 0
    scale = np.sqrt((m["n"] - ddof) / m["n"])
    assert p["RMSD"] == pytest.approx(m["URMSD"] * scale)
    assert p["RMSDonOSD"] == pytest.approx(m["URMSDonOSD"] * scale)
    for key in ("MSD", "OSD", "CC", "URMSD", "MSDonOSD", "URMSDonOSD"):
        assert p[key] == m[key]
