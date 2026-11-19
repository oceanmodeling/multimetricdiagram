"""Model-observation metrics: calculate_metrics()."""

import math
import numbers
import warnings

import numpy as np

_WARN_FRACTION = 5 / 12  # of the period: warn beyond this (150 for 360)


def calculate_metrics(
    model: np.ndarray,
    observation: np.ndarray,
    ddof: int = 0,
    period: float | None = None,
) -> dict[str, float | int | None]:
    """
    Calculate metrics comparing model with observations, for scalar or 2-D vector data.

    Inputs must contain no NaNs; remove them beforehand.

    Args:
        model: Shape (n,) for scalar data, or (n, 2) for n 2-component vectors (e.g. current
            velocity u, v).
        observation: Same shape as model.
        ddof: Delta degrees of freedom for MSD, OSD and URMSD: 0 (population, default) or 1
            (sample). RMSD and MB always divide by n.
        period: None (default) for ordinary data; or, for a circular scalar variable such as
            a phase angle, its period, a finite number > 0 (360 for degrees, 2*pi for
            radians, 24 for hour of day). Scalar data only; see Notes.

    Returns:
        dict with keys:
            RMSD: root-mean-square difference between model and observation.
            URMSD: unbiased root-mean-square difference.
            MB: mean bias. Scalar: mean(model) - mean(observation); vector: see Notes.
            MSD, OSD: model and observation standard deviations.
            MSDonOSD: MSD/OSD (+inf if OSD == 0).
            SDQ: standard deviations in quadrature, sqrt(MSD² + OSD²).
            RMSDonSDQ, URMSDonSDQ, MBonSDQ: RMSD, URMSD and MB divided by SDQ.
            VS: variance similarity, 2·MSD·OSD/(MSD² + OSD²), 0 to 1.
            VD: variance dissimilarity, sign(MSD - OSD)·(1 - VS), -1 to +1.
            CC: Pearson correlation coefficient, -1 to +1 (0 if MSD or OSD is 0).
            DIAS: Diagnostic Index of Agreement with Symmetry, VS·CC - (MB/SDQ)², -inf to 1
                (1 for perfect agreement).
            n: sample size, len(model) (int).
            ddof: the ddof used, 0 or 1 (int).
            period: the period given (None by default).

    Raises:
        ValueError: ddof is not 0 or 1; period is not None or a finite real number > 0, or is
            given with vector data; shapes differ or are not (n,) or (n, 2); an input
            contains NaN or inf; n < 2; both inputs are constant (with period: after
            unwrapping); the standard deviations are too small or too large to compute with
            (e.g. spreads near 1e-200 or 1e200, whose squares underflow or overflow); or a
            returned metric other than MSDonOSD is not finite (e.g. RMSD for a constant model
            near 1e200; named in the message).

    Warns:
        UserWarning: with period, some observations lie more than 5/12 of the period (150
            for period 360) from their circular mean, or some short-way model-observation
            differences exceed 5/12 of the period; results there depend on which way around
            they are counted (see Notes).

    Notes:
        Vector case, with VM, VO the per-sample model and observation vectors, <x> the mean
        over samples, ||x|| the Euclidean norm, "." the dot product, and nu = n - ddof:
            RMSD = <||VM - VO||²>^(1/2)
            URMSD = (sum ||(VM - <VM>) - (VO - <VO>)||² / nu)^(1/2)
            MSD = (sum ||VM - <VM>||² / nu)^(1/2), and OSD likewise
            CC = (sum (VM - <VM>).(VO - <VO>) / nu) / (MSD·OSD)
            MB = sign[(<VM> - <VO>).<VO>]·||<VM> - <VO>||, with sign +1 if the dot product is 0

        Circular scalar case (period given): the series are unwrapped, then the scalar metrics
        are computed from the unwrapped series as usual. With wrap(x) = (x + period/2) mod
        period - period/2 (x moved by whole periods into [-period/2, period/2)), mu the
        circular mean of the observations, the observations are taken as angles from mu and
        the model as those plus the short-way model-observation differences:
            mu = period/(2·pi)·arctan2(<sin a>, <cos a>), with a = 2·pi·observation/period
            obs_lin = wrap(observation - mu)
            model_lin = obs_lin + wrap(model - observation)
        So MB is the mean short-way difference, RMSD its RMS and URMSD its centered RMS;
        MSD, OSD and CC come from the unwrapped series; all are in the input's units and the
        identities of ordinary scalar data hold. Any input range works (e.g. -180 to 180,
        0 to 360 or unwrapped degrees give the same metrics). Limitations: differences over
        half a period, and observations more than half a period from their circular mean, are
        counted the short way; and the circular mean is unstable when the observations
        spread nearly all the way around the circle.
    """
    return _calculate_metrics(model, observation, ddof, period, stacklevel=3)


def _calculate_metrics(
    model: np.ndarray,
    observation: np.ndarray,
    ddof: int,
    period: float | None,
    stacklevel: int,
) -> dict[str, float | int | None]:
    """calculate_metrics(); a period warning uses stacklevel, counted from this function."""
    model = np.asarray(model, dtype=float)
    observation = np.asarray(observation, dtype=float)

    if ddof not in (0, 1):
        raise ValueError(f'ddof must be 0 or 1, got {ddof}')
    if period is not None and (
        isinstance(period, (bool, np.bool_))
        or not isinstance(period, numbers.Real)
        or not (math.isfinite(period) and period > 0)
    ):
        raise ValueError(f'period must be None or a finite real number > 0, got {period!r}')
    if model.shape != observation.shape:
        raise ValueError(
            'model and observation must have the same shape, '
            f'got {model.shape} and {observation.shape}'
        )
    if model.ndim not in (1, 2):
        raise ValueError(
            'model and observation must be 1-D (scalar) or 2-D (vector) arrays, '
            f'got shape {model.shape}'
        )
    if model.ndim == 2 and model.shape[1] != 2:
        raise ValueError(
            'vector model/observation must have exactly 2 components (shape (n, 2)), '
            f'got shape {model.shape}'
        )
    if model.ndim == 2 and period is not None:
        raise ValueError('period applies to scalar data only, not to vector (n, 2) data')
    if np.any(np.isnan(model)):
        raise ValueError('model contains NaN values')
    if np.any(np.isnan(observation)):
        raise ValueError('observation contains NaN values')
    if np.any(np.isinf(model)):
        raise ValueError('model contains inf values')
    if np.any(np.isinf(observation)):
        raise ValueError('observation contains inf values')
    n = model.shape[0]
    if n < 2:
        raise ValueError(f'model and observation must have at least 2 elements, got {n}')
    if period is not None:
        model, observation = _unwrap(model, observation, period, stacklevel + 1)
    if np.all(model == model[0]) and np.all(observation == observation[0]):
        raise ValueError(
            'at least one of model and observation must have at least two different values'
        )

    if model.ndim == 2:
        # vector case (Notes); nu cancels in CC, so CC is ddof-invariant like np.corrcoef()
        nu = n - ddof
        model_mean = np.mean(model, axis=0)
        obs_mean = np.mean(observation, axis=0)
        model_anom = model - model_mean
        obs_anom = observation - obs_mean
        msd = float(np.sqrt(np.sum(model_anom**2) / nu))
        osd = float(np.sqrt(np.sum(obs_anom**2) / nu))
        _check_spread(msd, osd)
        diff = model - observation
        rmsd = float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        urmsd = float(np.sqrt(np.sum((model_anom - obs_anom) ** 2) / nu))
        cc = (
            float((np.sum(model_anom * obs_anom) / nu) / (msd * osd))
            if msd > 0 and osd > 0
            else 0.0
        )
        mean_diff = model_mean - obs_mean
        mb_mag = float(np.linalg.norm(mean_diff))
        mb_sign_arg = float(np.dot(mean_diff, obs_mean))
        mb_sign = 1.0 if mb_sign_arg == 0 else float(np.sign(mb_sign_arg))
        mb = mb_sign * mb_mag
    else:
        msd = float(np.std(model, ddof=ddof))
        osd = float(np.std(observation, ddof=ddof))
        _check_spread(msd, osd)
        cc = float(np.corrcoef(model, observation)[0, 1]) if msd > 0 and osd > 0 else 0.0
        diff = model - observation
        rmsd = float(np.sqrt(np.mean(diff**2)))
        mb = float(np.mean(diff))
        urmsd = float(np.std(diff, ddof=ddof))

    msd_on_osd = msd / osd if osd > 0 else float('inf')
    sdq = float(np.sqrt(msd**2 + osd**2))
    vs = float(2 * msd * osd / sdq**2)
    vd = float(np.sign(msd - osd) * (1 - vs))
    rmsds = rmsd / sdq
    mbs = mb / sdq
    urmsds = urmsd / sdq
    dias = cc * vs - mbs * mbs  # not **, which raises OverflowError instead of giving inf

    metrics = {
        'RMSDonSDQ': rmsds,
        'URMSDonSDQ': urmsds,
        'MBonSDQ': mbs,
        'RMSD': rmsd,
        'URMSD': urmsd,
        'MB': mb,
        'CC': cc,
        'VS': vs,
        'VD': vd,
        'MSD': msd,
        'OSD': osd,
        'MSDonOSD': msd_on_osd,
        'SDQ': sdq,
        'DIAS': dias,
        'n': n,
        'ddof': int(ddof),
        'period': period,
    }
    # MSDonOSD = +inf for OSD == 0 is documented; any other non-finite value is an overflow
    not_finite = [
        key
        for key, value in metrics.items()
        if key != 'MSDonOSD' and isinstance(value, float) and not np.isfinite(value)
    ]
    if not_finite:
        raise ValueError(
            f'metrics not finite: {", ".join(not_finite)} (input values too large to compute)'
        )
    return metrics


def _unwrap(
    model: np.ndarray, observation: np.ndarray, period: float, stacklevel: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (model_lin, obs_lin) of a circular scalar variable (Notes of calculate_metrics)."""
    p = float(period)

    def wrap(x):
        return (x + p / 2) % p - p / 2

    a = 2 * np.pi * observation / p
    mu = p / (2 * np.pi) * np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))
    obs_lin = wrap(observation - mu)
    diff = wrap(model - observation)
    limit = _WARN_FRACTION * p
    far = [
        text
        for values, text in (
            (obs_lin, 'some observations lie near the opposite side of their circular mean'),
            (diff, 'some model-observation differences are near half a period'),
        )
        if np.any(np.abs(values) > limit)
    ]
    if far:
        warnings.warn(
            f'{" and ".join(far)} (beyond 5/12 of period={p:g}, i.e. {limit:g}); results '
            'there depend on which way around the circle they are counted (here the short way)',
            UserWarning,
            stacklevel=stacklevel,
        )
    return obs_lin + diff, obs_lin


def _check_spread(msd: float, osd: float) -> None:
    """Raise ValueError unless SDQ**2 = MSD**2 + OSD**2 is finite and > 0 (it is divided by)."""
    sdq_sq = msd * msd + osd * osd  # not **, which raises OverflowError instead of giving inf
    if not (np.isfinite(sdq_sq) and sdq_sq > 0):
        raise ValueError(
            f'model/observation standard deviations are too small or too large to compute the '
            f'metrics (MSD={msd!r}, OSD={osd!r}); rescale the data'
        )
