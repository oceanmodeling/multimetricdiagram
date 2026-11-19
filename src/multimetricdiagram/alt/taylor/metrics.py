"""Taylor diagram metrics: calculate_metrics()."""

import numpy as np

from ...metrics import _calculate_metrics

_KEPT = ('MSD', 'OSD', 'CC', 'RMSD', 'URMSD', 'MB', 'n', 'ddof', 'period')


def calculate_metrics(
    model: np.ndarray,
    observation: np.ndarray,
    ddof: int = 0,
    period: float | None = None,
    patternonly: bool = False,
) -> dict[str, float | int | None]:
    """
    Calculate the metrics a Taylor diagram shows, for scalar or 2-D vector data.

    Computed by the top-level calculate_metrics() (same inputs, validation, vector definitions,
    ddof and period handling; see its docstring), then divided by OSD.

    Args:
        model: Shape (n,) for scalar data, or (n, 2) for n 2-component vectors (e.g. current
            velocity u, v).
        observation: Same shape as model.
        ddof: Delta degrees of freedom for MSD, OSD and URMSD: 0 (population, default) or 1
            (sample). RMSD and MB always divide by n.
        period: None (default), or the period of a circular scalar variable (360 for degrees,
            2*pi for radians, ...); passed to the top-level calculate_metrics() (see its Notes).
        patternonly: False (default) or True: MB and MBonOSD are set to 0 and RMSD and
            RMSDonOSD to their zero-bias values, URMSD and URMSDonOSD times sqrt((n - ddof)/n)
            (equal to them with ddof = 0); the other metrics are unchanged. For a pattern-only
            diagram (no sticks): use with make_fig_axes(patternonly=True).

    Returns:
        dict with keys:
            MSD, OSD: model and observation standard deviations.
            CC: correlation coefficient, -1 to +1.
            RMSD, URMSD, MB: root-mean-square, unbiased root-mean-square and mean
                differences, as in the top-level calculate_metrics().
            n: sample size (int).
            ddof: the ddof used, 0 or 1 (int).
            period: the period given (None by default).
            MSDonOSD, RMSDonOSD, URMSDonOSD, MBonOSD: MSD, RMSD, URMSD and MB divided by OSD.
            With patternonly=True, MB and MBonOSD are 0 and RMSD and RMSDonOSD the zero-bias
            values (see patternonly).

    Raises:
        ValueError: any error of the top-level calculate_metrics(); or the observations are
            constant (OSD = 0, so nothing can be divided by it); or a ratio to OSD is not
            finite (OSD too small for the differences; the message names the ratio(s)).
        TypeError: patternonly is not a bool.

    Warns:
        UserWarning: as the top-level calculate_metrics(), when period is given.
    """
    if not isinstance(patternonly, (bool, np.bool_)):
        raise TypeError(f'patternonly must be a bool, got {patternonly!r}')
    metrics = _calculate_metrics(model, observation, ddof, period, stacklevel=3)
    osd = metrics['OSD']
    if not osd > 0:  # pylint: disable=unnecessary-negation  # also catches NaN
        raise ValueError('observation is constant (OSD = 0): a Taylor diagram needs OSD > 0')
    result = {key: metrics[key] for key in _KEPT}
    result['MSDonOSD'] = metrics['MSDonOSD']
    for key in ('RMSD', 'URMSD', 'MB'):
        result[f'{key}onOSD'] = metrics[key] / osd
    not_finite = [
        key
        for key in ('MSDonOSD', 'RMSDonOSD', 'URMSDonOSD', 'MBonOSD')
        if not np.isfinite(result[key])
    ]
    if not_finite:
        raise ValueError(
            f'metrics not finite: {", ".join(not_finite)} (OSD too small to divide by)'
        )
    if patternonly:
        # zero bias: RMSD (divided by n) from URMSD (divided by n - ddof)
        scale = np.sqrt((result['n'] - result['ddof']) / result['n'])
        result['MB'] = result['MBonOSD'] = 0.0
        result['RMSD'] = result['URMSD'] * scale
        result['RMSDonOSD'] = result['URMSDonOSD'] * scale
    return result
