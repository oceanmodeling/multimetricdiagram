"""Markersticks on the Taylor diagram: add_markerstick()."""

import warnings
from typing import Any

import numpy as np

from ...markerstick import MarkerStick, _draw_markerstick, _label_dicts
from .axes import _OPTIONS_ATTR

_REQUIRED = {
    'MSD',
    'OSD',
    'CC',
    'RMSD',
    'URMSD',
    'MB',
    'n',
    'ddof',
    'MSDonOSD',
    'RMSDonOSD',
    'URMSDonOSD',
    'MBonOSD',
}


def add_markerstick(
    ax: Any,
    metrics: dict[str, float],
    *,
    markerstick_style: dict[str, Any] | None = None,
    label_info: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> MarkerStick:
    """
    Add a markerstick (marker, stick, optional text labels) for one metrics dict to a Taylor
    diagram.

    The marker is at radius MSD/OSD (MSD with normalized=False) and angle arccos(CC), so its
    distance from the star is URMSD/OSD (URMSD). The stick, of length |MB/OSD| (|MB|), is
    tangent to the circle about the star through the marker: clockwise about the star if
    MB > 0 (rightward for a marker above the star), counterclockwise if MB < 0; with ddof = 0
    it ends on the circle of RMSD/OSD (RMSD). For a marker at the star (URMSD = 0) the stick
    is horizontal, rightward if MB > 0. The marker is rotated so its top points away from the
    star, and labels are rotated parallel to the stick.

    Args:
        ax: Polar axes from taylor.make_fig_axes().
        metrics: Dict returned by taylor.calculate_metrics().
        markerstick_style, label_info: As in the top-level add_markerstick() (same defaults),
            with label 'position' 'outside' (default) away from the star, 'inside' toward it
            and 'beside' opposite the stick; nonzero offsets are gaps from the marker's edge
            (presets 0.1 x fontsize).

    Returns:
        MarkerStick(marker, stick, labels): the PathCollection from ax.scatter(), the Line2D
        from ax.plot(), and a list of PathPatch labels (one per label dict). It unpacks like the
        earlier 3-tuple: marker, stick, labels = taylor.add_markerstick(...).

    Raises:
        ValueError: ax is not from taylor.make_fig_axes(); metrics lacks a required key, its
            'ddof' is not 0 or 1, or its values violate the metric identities (named in the
            message); CC < 0 on a 1-quadrant diagram (make the figure with
            make_fig_axes(quadrants=2), or with metrics= so 2 quadrants are chosen); with
            normalized=False, metrics['OSD'] differs from the diagram's osd; the diagram is
            pattern-only (make_fig_axes(patternonly=True)) but MB is not 0 (metrics not from
            calculate_metrics(patternonly=True)); a label dict has no 'text' key; or a label's
            'position' is not 'outside', 'inside' or 'beside'.
        TypeError: label_info is not None, a dict or a list, or a list item is not a dict.

    Warns:
        UserWarning: the marker's radius is beyond the diagram's r_max (it is still drawn;
            re-run make_fig_axes() with a larger r_max). Sticks may extend past the arc.
            MB is 0 on a diagram that is not pattern-only (no stick is drawn; for metrics from
            calculate_metrics(patternonly=True) use make_fig_axes(patternonly=True)).
    """
    options = getattr(ax, _OPTIONS_ATTR, None)
    if options is None:
        raise ValueError("ax must be the polar axes returned by taylor.make_fig_axes()")
    missing = _REQUIRED - set(metrics.keys())
    if missing:
        raise ValueError(f'Missing metrics: {missing}')
    if metrics['ddof'] not in (0, 1):
        raise ValueError(f"metrics['ddof'] must be 0 or 1, got {metrics['ddof']!r}")
    f = metrics['n'] / (metrics['n'] - metrics['ddof'])
    k = metrics['MSDonOSD']
    consistency_pairs = (
        (metrics['RMSDonOSD'] ** 2, metrics['URMSDonOSD'] ** 2 / f + metrics['MBonOSD'] ** 2),
        (metrics['URMSDonOSD'] ** 2, k**2 + 1 - 2 * k * metrics['CC']),
        *(
            (metrics[f'{key}onOSD'] * metrics['OSD'], metrics[key])
            for key in ('MSD', 'RMSD', 'URMSD', 'MB')
        ),
    )
    if any(not np.isclose(a, b) for a, b in consistency_pairs):
        raise ValueError(
            'Inconsistent metrics: must satisfy (RMSD/OSD)² = (URMSD/OSD)²/f + (MB/OSD)² '
            '(f = n/(n-ddof)), (URMSD/OSD)² = (MSD/OSD)² + 1 - 2*(MSD/OSD)*CC, and '
            'XonOSD = X/OSD for X = MSD, RMSD, URMSD, MB'
        )
    if options['patternonly'] and metrics['MBonOSD'] != 0:
        raise ValueError(
            "the diagram is pattern-only (make_fig_axes(patternonly=True)) but metrics has "
            "MB != 0; compute them with calculate_metrics(..., patternonly=True)"
        )
    if not options['patternonly'] and metrics['MBonOSD'] == 0:
        warnings.warn(
            "metrics has MB = 0 (as from calculate_metrics(patternonly=True)), so there is no "
            "stick; for a pattern-only diagram use make_fig_axes(patternonly=True)",
            UserWarning,
            stacklevel=2,
        )
    label_dicts = _label_dicts(label_info)
    cc = metrics['CC']
    if options['quadrants'] == 1 and cc < 0:
        raise ValueError(
            f"CC={cc!r} < 0 is outside a 1-quadrant Taylor diagram; use make_fig_axes(quadrants=2)"
            " or pass the metrics to make_fig_axes(metrics=...) to choose quadrants from them"
        )
    if options['normalized']:
        r, mb = metrics['MSDonOSD'], metrics['MBonOSD']
    else:
        if not np.isclose(metrics['OSD'], options['osd']):
            raise ValueError(
                f"metrics['OSD']={metrics['OSD']!r} differs from the diagram's "
                f"osd={options['osd']!r}"
            )
        r, mb = metrics['MSD'], metrics['MB']
    star_r = options['star_r']
    if r > options['r_max'] * (1 + 1e-9):
        warnings.warn(
            f"marker at radius {r:.3g} is beyond r_max={options['r_max']:g}; re-run "
            "make_fig_axes() and pass a larger r_max value",
            UserWarning,
            stacklevel=2,
        )

    # marker position; unit vector (ex, ey) from the star to the marker
    theta = np.arccos(np.clip(cc, -1.0, 1.0))
    x0, y0 = r * np.cos(theta), r * np.sin(theta)
    d = np.hypot(x0 - star_r, y0)
    ex, ey = (0.0, 1.0) if np.isclose(d / star_r, 0.0) else ((x0 - star_r) / d, y0 / d)
    # stick: tangent to the circle about the star, clockwise about it if MB > 0
    x1, y1 = x0 + mb * ey, y0 - mb * ex
    return _draw_markerstick(
        ax,
        (theta, r),
        (np.arctan2(y1, x1), np.hypot(x1, y1)),
        np.degrees(np.arctan2(ey, ex)) - 90,
        mb < 0,
        markerstick_style,
        label_dicts,
    )
