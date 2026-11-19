import numpy as np
import pytest

import multimetricdiagram as mmd

OBS = np.array([1.0, 2.5, 3.0, 4.5, 6.0, 5.5, 7.0, 8.5])
MOD = np.array([1.5, 2.0, 3.8, 4.0, 6.5, 6.2, 6.9, 9.4])


def _expected_scalar(mod, obs, ddof):
    """Metrics computed directly from their definitions, independent of calculate_metrics()."""
    n = len(obs)
    am = mod - np.mean(mod)
    ao = obs - np.mean(obs)
    msd = np.sqrt(np.sum(am**2) / (n - ddof))
    osd = np.sqrt(np.sum(ao**2) / (n - ddof))
    sdq = np.sqrt(msd**2 + osd**2)
    rmsd = np.sqrt(np.mean((mod - obs) ** 2))
    urmsd = np.sqrt(np.sum((am - ao) ** 2) / (n - ddof))
    mb = np.mean(mod) - np.mean(obs)
    cc = np.sum(am * ao) / np.sqrt(np.sum(am**2) * np.sum(ao**2))
    vs = 2 * msd * osd / (msd**2 + osd**2)
    return {
        "RMSD": rmsd,
        "URMSD": urmsd,
        "MB": mb,
        "MSD": msd,
        "OSD": osd,
        "SDQ": sdq,
        "CC": cc,
        "VS": vs,
        "VD": np.sign(msd - osd) * (1 - vs),
        "MSDonOSD": msd / osd,
        "RMSDonSDQ": rmsd / sdq,
        "URMSDonSDQ": urmsd / sdq,
        "MBonSDQ": mb / sdq,
        "DIAS": vs * cc - (mb / sdq) ** 2,
    }


@pytest.mark.parametrize("ddof", [0, 1])
def test_scalar_matches_definitions(ddof):
    m = mmd.calculate_metrics(MOD, OBS, ddof=ddof)
    for key, value in _expected_scalar(MOD, OBS, ddof).items():
        assert m[key] == pytest.approx(value, rel=1e-12, abs=1e-12), key
    assert m["n"] == len(OBS)
    assert m["ddof"] == ddof and type(m["ddof"]) is int


@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("seed", [None, 3])
def test_scalar_identities(ddof, seed):
    if seed is None:
        mod, obs = MOD, OBS
    else:
        rng = np.random.default_rng(seed)
        obs = rng.normal(5.0, 2.0, 50)
        mod = 0.7 * obs + rng.normal(1.0, 1.5, 50)
    m = mmd.calculate_metrics(mod, obs, ddof=ddof)
    f = m["n"] / (m["n"] - ddof)
    assert m["RMSDonSDQ"] ** 2 == pytest.approx(m["URMSDonSDQ"] ** 2 / f + m["MBonSDQ"] ** 2)
    assert m["URMSDonSDQ"] ** 2 == pytest.approx(1 - m["VS"] * m["CC"])
    assert m["DIAS"] == pytest.approx(m["VS"] * m["CC"] - m["MBonSDQ"] ** 2)


@pytest.mark.parametrize("ddof", [0, 1])
def test_scalar_perfect_agreement(ddof):
    m = mmd.calculate_metrics(OBS, OBS, ddof=ddof)
    assert m["RMSD"] == pytest.approx(0.0, abs=1e-12)
    assert m["CC"] == pytest.approx(1.0)
    assert m["VS"] == pytest.approx(1.0)
    assert m["DIAS"] == pytest.approx(1.0)


@pytest.mark.parametrize("ddof", [0, 1])
def test_vector_with_zero_second_component_matches_scalar(ddof):
    assert np.mean(OBS) > 0  # needed for vector MB sign to match scalar MB sign
    zeros = np.zeros_like(OBS)
    vec = mmd.calculate_metrics(
        np.column_stack([MOD, zeros]), np.column_stack([OBS, zeros]), ddof=ddof
    )
    sca = mmd.calculate_metrics(MOD, OBS, ddof=ddof)
    for key in ("RMSD", "URMSD", "MSD", "OSD", "CC", "MB"):
        assert vec[key] == pytest.approx(sca[key], rel=1e-12, abs=1e-12), key


@pytest.mark.parametrize(
    "model, observation, kwargs, match",
    [
        (np.arange(5.0), np.arange(6.0), {}, "same shape"),
        (np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, 3.0]), {}, "model contains NaN"),
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, np.nan, 3.0]), {}, "observation contains NaN"),
        (np.ones((4, 2, 1)), np.ones((4, 2, 1)), {}, "1-D"),
        (np.arange(15.0).reshape(5, 3), np.arange(15.0).reshape(5, 3), {}, "exactly 2 components"),
        (np.array([1.0]), np.array([2.0]), {}, "at least 2 elements"),
        (np.ones(5), 2 * np.ones(5), {}, "at least two different values"),
        (MOD, OBS, {"ddof": 2}, "ddof must be 0 or 1"),
    ],
    ids=[
        "shape-mismatch",
        "nan-model",
        "nan-observation",
        "ndim-3",
        "vector-3-components",
        "n-less-than-2",
        "both-constant",
        "ddof-2",
    ],
)
def test_input_validation(model, observation, kwargs, match):
    with pytest.raises(ValueError, match=match):
        mmd.calculate_metrics(model, observation, **kwargs)


def _vector(x):
    return np.column_stack([x, np.zeros_like(x)])


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.parametrize(
    "model, observation",
    [
        (np.array([0.0, 1e-200]), np.array([0.0, 2e-200])),  # squares underflow: SDQ == 0
        (np.array([0.0, 1e200]), np.array([0.0, 2e200])),  # squares overflow
        (_vector(np.array([0.0, 1e-200])), _vector(np.array([0.0, 2e-200]))),
        (_vector(np.array([0.0, 1e200])), _vector(np.array([0.0, 2e200]))),
    ],
    ids=["scalar-underflow", "scalar-overflow", "vector-underflow", "vector-overflow"],
)
def test_unrepresentable_spread_raises_value_error(model, observation):
    with pytest.raises(ValueError, match="too small or too large"):
        mmd.calculate_metrics(model, observation)


@pytest.mark.parametrize("bad", [np.inf, -np.inf], ids=["inf", "-inf"])
@pytest.mark.parametrize("which", ["model", "observation"])
def test_inf_input_raises(bad, which):
    model, observation = MOD.copy(), OBS.copy()
    (model if which == "model" else observation)[2] = bad
    with pytest.raises(ValueError, match=f"{which} contains inf"):
        mmd.calculate_metrics(model, observation)


@pytest.mark.filterwarnings("ignore:overflow encountered:RuntimeWarning")
@pytest.mark.parametrize("vector", [False, True], ids=["scalar", "vector"])
def test_overflowing_rmsd_raises_naming_it(vector):
    prep = _vector if vector else (lambda x: x)
    # constant (spread check passes) and a power of 2 (~4.8e198, exact mean); RMSD overflows
    model = np.full_like(OBS, 2.0**660)
    with pytest.raises(ValueError, match=r"not finite: .*\bRMSD\b"):
        mmd.calculate_metrics(prep(model), prep(OBS))


@pytest.mark.parametrize("vector", [False, True], ids=["scalar", "vector"])
def test_default_ddof_is_0(vector):
    prep = _vector if vector else (lambda x: x)
    default = mmd.calculate_metrics(prep(MOD), prep(OBS))
    assert default["ddof"] == 0
    assert default == mmd.calculate_metrics(prep(MOD), prep(OBS), ddof=0)


@pytest.mark.parametrize("vector", [False, True], ids=["scalar", "vector"])
def test_ddof_0_rmsd_pythagorean_identity_is_exact(vector):
    rng = np.random.default_rng(1)
    obs = rng.normal(5.0, 2.0, (40, 2) if vector else 40)
    mod = 0.7 * obs + rng.normal(1.0, 1.5, obs.shape)
    m = mmd.calculate_metrics(mod, obs, ddof=0)
    lhs = m["RMSDonSDQ"] ** 2
    assert lhs == pytest.approx(m["URMSDonSDQ"] ** 2 + m["MBonSDQ"] ** 2, rel=1e-12, abs=1e-12)
