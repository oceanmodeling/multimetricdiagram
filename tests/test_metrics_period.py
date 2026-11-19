import warnings
from pathlib import Path

import numpy as np
import pytest

import multimetricdiagram as mmd
from multimetricdiagram.alt import taylor

_RNG = np.random.default_rng(7)
_N = 120
# never-wrapped series in degrees, crossing 180
TRUTH_OBS = 170 + 25 * np.sin(np.linspace(0, 6, _N)) + _RNG.normal(0, 5, _N)
TRUTH_MOD = 0.8 * (TRUTH_OBS - 170) + 175 + _RNG.normal(0, 6, _N)


def _wrap(x, period):
    return (x + period / 2) % period - period / 2


def _assert_metrics_close(a, b):
    keys = [k for k in a if isinstance(a[k], float)]
    assert keys and set(keys) == {k for k in b if isinstance(b[k], float)}
    for key in keys:
        assert a[key] == pytest.approx(b[key], rel=1e-10, abs=1e-10), key


@pytest.mark.parametrize("period", [360, 2 * np.pi, 24])
def test_matches_short_way_reference(period):
    obs = _wrap(TRUTH_OBS * period / 360, period)
    mod = _wrap(TRUTH_MOD * period / 360, period)
    assert obs.min() < -0.4 * period and obs.max() > 0.4 * period  # crosses +-period/2
    d = _wrap(mod - obs, period)
    m = mmd.calculate_metrics(mod, obs, period=period)
    assert m["MB"] == pytest.approx(np.mean(d), rel=1e-12, abs=1e-12)
    assert m["RMSD"] == pytest.approx(np.sqrt(np.mean(d**2)), rel=1e-12)
    assert m["URMSD"] == pytest.approx(np.sqrt(np.mean((d - d.mean()) ** 2)), rel=1e-12)
    assert m["period"] == period


def test_whole_periods_and_common_shift_change_nothing():
    rng = np.random.default_rng(11)
    obs, mod = _wrap(TRUTH_OBS, 360), _wrap(TRUTH_MOD, 360)
    ref = mmd.calculate_metrics(mod, obs, period=360)
    obs2 = obs + 360 * rng.integers(-3, 4, _N)
    mod2 = mod + 360 * rng.integers(-3, 4, _N)
    _assert_metrics_close(mmd.calculate_metrics(mod2, obs, period=360), ref)
    _assert_metrics_close(mmd.calculate_metrics(mod, obs2, period=360), ref)
    _assert_metrics_close(mmd.calculate_metrics(mod + 123.4, obs + 123.4, period=360), ref)


def test_wrapped_input_equals_unwrapped_truth():
    m = mmd.calculate_metrics(_wrap(TRUTH_MOD, 360), _wrap(TRUTH_OBS, 360), period=360)
    _assert_metrics_close(m, mmd.calculate_metrics(TRUTH_MOD, TRUTH_OBS))


def test_no_wrapping_equals_period_none():
    rng = np.random.default_rng(5)
    obs = rng.uniform(80, 120, 50)
    mod = np.clip(obs + rng.normal(2, 5, 50), 80, 120)
    _assert_metrics_close(
        mmd.calculate_metrics(mod, obs, ddof=1, period=360),
        mmd.calculate_metrics(mod, obs, ddof=1),
    )


@pytest.mark.parametrize(
    "period, vector, match",
    [
        (360, True, "scalar data only"),
        *((p, False, "finite real number > 0") for p in (0, -1, np.nan, np.inf, True, "360")),
    ],
)
def test_invalid_period_raises(period, vector, match):
    model, observation = TRUTH_MOD[:10], TRUTH_OBS[:10]
    if vector:
        model, observation = np.column_stack([model, observation]), np.column_stack([model, model])
    with pytest.raises(ValueError, match=match):
        mmd.calculate_metrics(model, observation, period=period)


@pytest.mark.parametrize(
    "model, observation, match",
    [
        ([0, 10, 20, 175], [0, 10, 20, 0], "differences are near half a period"),
        ([0, 0, 0, 170], [0, 0, 0, 170], "opposite side of their circular mean"),
    ],
)
def test_warning_near_half_a_period(model, observation, match):
    with pytest.warns(UserWarning, match=match) as record:
        mmd.calculate_metrics(np.array(model), np.array(observation), period=360)
    assert len(record) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mmd.calculate_metrics(_wrap(TRUTH_MOD, 360), _wrap(TRUTH_OBS, 360), period=360)


def test_taylor_passes_period_through():
    mod, obs = _wrap(TRUTH_MOD, 360), _wrap(TRUTH_OBS, 360)
    m = taylor.calculate_metrics(mod, obs, period=360)
    top = mmd.calculate_metrics(mod, obs, period=360)
    assert m["period"] == 360 and taylor.calculate_metrics(mod, obs)["period"] is None
    for key in ("MSD", "OSD", "CC", "RMSD", "URMSD", "MB"):
        assert m[key] == top[key]
    with pytest.raises(ValueError, match="finite real number"):
        taylor.calculate_metrics(mod, obs, period=-360)
    # the period warning points at the caller's line, through either function
    for func in (taylor.calculate_metrics, mmd.calculate_metrics):
        with pytest.warns(UserWarning, match="half a period") as record:
            func(np.array([0.0, 10, 20, 175]), np.array([0.0, 10, 20, 0]), period=360)
        assert Path(record[0].filename).resolve() == Path(__file__).resolve(), func.__module__


def test_add_markerstick_accepts_period_metrics():
    mod, obs = _wrap(TRUTH_MOD, 360), _wrap(TRUTH_OBS, 360)
    _, ax, _ = mmd.make_fig_axes()
    mmd.add_markerstick(ax, mmd.calculate_metrics(mod, obs, period=360))
    m = taylor.calculate_metrics(mod, obs, period=360)
    _, ax, _ = taylor.make_fig_axes(metrics=[m])
    taylor.add_markerstick(ax, m)
