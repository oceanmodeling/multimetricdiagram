"""Markersticks on the multimetric diagram: add_markerstick(), _resolve_markerstick_style()."""

from typing import Any, NamedTuple

import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.colors import is_color_like, to_rgba
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D

from ._defaults import (
    ALPHA,
    LABEL_FONTSIZE,
    LABEL_POSITION_PRESETS,
    MARKER_FACECOLOR,
    MARKER_SHAPE,
    MARKER_SIZE,
    MARKERSTICK_COLOR,
    STICK_LINEWIDTH,
)
from ._text import text_polar

# |1 - r²| (= VS*CC) below this counts as r = 1: the marker is on the CC = 0 ring
_RING_TOL = 1e-12


class MarkerStick(NamedTuple):
    """The artists add_markerstick() draws: the marker, the stick and the labels."""

    marker: PathCollection
    stick: Line2D
    labels: list[PathPatch]


def add_markerstick(
    ax: Any,
    metrics: dict[str, float],
    *,
    markerstick_style: dict[str, Any] | None = None,
    label_info: dict[str, Any] | list[dict[str, Any]] | None = None,
) -> MarkerStick:
    """
    Add a markerstick (marker, stick, optional text labels) for one metrics dict to polar axes.

    On the CC = 0 ring (|1 - (URMSD/SDQ)²| < 1e-12), theta = -45 + 135*VS deg (or mirrored).

    Args:
        ax: Polar axes (e.g. from make_fig_axes()).
        metrics: Dict returned by calculate_metrics().
        markerstick_style: Optional dict styling marker, stick and label color; keys are routed
            by _resolve_markerstick_style(). Defaults: color 'r'; open (unfilled) 'o' marker,
            size 14; stick linewidth 1.5; alpha 0.6 (marker and stick); zorder 2. A flat
            'zorder' sets marker and stick; 'zorder' in the 'marker' or 'stick' sub-dict sets
            that part alone.
        label_info: None (default, no labels), a label dict, or a list of label dicts. Each
            dict needs 'text' (any object, converted with str(); None gives ''); labels are
            rotated parallel to the stick. Optional keys:
            - 'position' preset (offsets in multiples of fontsize; unlisted offsets are 0):
                'outside' (default; away from origin): va='bottom', ha='center', out_offset=0.1
                'inside' (toward origin): va='top', ha='center', out_offset=-0.1
                'beside' (opposite the stick): va='center'; if MB/SDQ >= 0, ha='right',
                    cw_offset=-0.1, else ha='left', cw_offset=0.1
            - 'va', 'ha', 'out_offset' (+ outward), 'cw_offset' (+ clockwise): override the
              preset. A nonzero offset is a gap from the marker's edge as drawn (shape, size and
              half the edge linewidth) to the label's va/ha point; 0 keeps that point at the
              marker center.
            - 'fontsize': default 10.
            - 'color': default the marker's edge color; if that is 'none' or fully transparent
              as drawn (after the marker alpha), the marker's face color if visible, else 'k'.
            - 'alpha', 'zorder': default the marker's.
            - Other keys are passed to text_polar() (e.g. 'fontweight').

    Returns:
        MarkerStick(marker, stick, labels): the PathCollection from ax.scatter(), the Line2D
        from ax.plot(), and a list of PathPatch labels (one per label dict). It unpacks like the
        earlier 3-tuple: marker, stick, labels = add_markerstick(...).

    Raises:
        ValueError: metrics lacks a required key, its 'ddof' is not 0 or 1, or its values
            violate the metric identities (named in the message); or the marker angle is not
            finite (CC's sign inconsistent with URMSD/SDQ); or a label dict has no 'text' key;
            or a label's 'position' is not 'outside', 'inside' or 'beside'.
        TypeError: label_info is not None, a dict or a list, or a list item is not a dict.

    Examples:
        Default style (red open circle and stick), no labels:
            mmd.add_markerstick(ax, metrics)
        Blue, with label "Case A" outward from the marker:
            mmd.add_markerstick(ax, metrics, markerstick_style={'color': 'blue'},
                                label_info={'text': 'Case A'})
        Square marker, thicker stick, two labels:
            mmd.add_markerstick(ax, metrics, markerstick_style={'marker': 's', 'linewidth': 3},
                                label_info=[{'text': 'Case A'},
                                            {'text': 'Run 1', 'position': 'inside'}])
        Thin marker edge ('linewidths'), thick stick ('linewidth'):
            mmd.add_markerstick(ax, metrics, markerstick_style={'linewidths': 0.5, 'linewidth': 4})
    """
    required_metrics = {
        'RMSDonSDQ',
        'URMSDonSDQ',
        'MBonSDQ',
        'CC',
        'VS',
        'MSD',
        'OSD',
        'n',
        'ddof',
    }
    missing = required_metrics - set(metrics.keys())
    if missing:
        raise ValueError(f'Missing metrics: {missing}')
    if metrics['ddof'] not in (0, 1):
        raise ValueError(f"metrics['ddof'] must be 0 or 1, got {metrics['ddof']!r}")
    f = metrics['n'] / (metrics['n'] - metrics['ddof'])
    consistency_pairs = (
        (metrics['RMSDonSDQ'] ** 2, metrics['URMSDonSDQ'] ** 2 / f + metrics['MBonSDQ'] ** 2),
        (metrics['URMSDonSDQ'] ** 2, 1 - metrics['VS'] * metrics['CC']),
        (
            metrics['VS'],
            2 * metrics['MSD'] * metrics['OSD'] / (metrics['MSD'] ** 2 + metrics['OSD'] ** 2),
        ),
    )
    if any(not np.isclose(a, b) for a, b in consistency_pairs):
        raise ValueError(
            'Inconsistent metrics: must satisfy (RMSD/SDQ)² = (URMSD/SDQ)²/f + (MB/SDQ)² '
            '(f = n/(n-ddof)), (URMSD/SDQ)² = 1 - VS*CC, and VS = 2*MSD*OSD/(MSD² + OSD²)'
        )
    label_dicts = _label_dicts(label_info)

    # marker position: r = URMSD/SDQ; theta from CC, or from VS on the CC = 0 ring (r ~ 1)
    r = metrics["URMSDonSDQ"]
    if np.isclose(r, 0.0):
        theta_deg = 90.0
    elif abs(1 - r * r) < _RING_TOL:
        theta_deg = -45 + 135 * metrics["VS"]
    elif r < 1:
        theta_deg = -45 + 135 * np.log(metrics["CC"]) / np.log(1 - r * r)
    elif r < np.sqrt(2):
        theta_deg = -45 + 135 * np.log(-metrics["CC"]) / np.log(r * r - 1)
    else:
        theta_deg = 90.0
    if not np.isfinite(theta_deg):
        raise ValueError(
            f"marker angle is not finite: CC={metrics['CC']!r} is inconsistent with "
            f"URMSD/SDQ={r!r} (CC must be > 0 where URMSD/SDQ < 1, and < 0 where > 1)"
        )
    if metrics["MSD"] < metrics["OSD"]:  # mirror into the left half
        theta_deg = 225 - (theta_deg + 45)
    theta = np.deg2rad(theta_deg)
    # stick: tangent to the circle at the marker, length |MB/SDQ|, clockwise if MB/SDQ > 0
    x0 = r * np.cos(theta)
    y0 = r * np.sin(theta)
    tx = -np.sin(theta)
    ty = np.cos(theta)
    stick_len = -metrics["MBonSDQ"]
    x1 = x0 + stick_len * tx
    y1 = y0 + stick_len * ty
    r_end = np.sqrt(x1**2 + y1**2)
    theta_end = np.angle(x1 + y1 * 1j)
    return _draw_markerstick(
        ax,
        (theta, r),
        (theta_end, r_end),
        theta_deg - 90,
        metrics['MBonSDQ'] < 0,
        markerstick_style,
        label_dicts,
    )


def _draw_markerstick(
    ax: Any,
    marker_xy: tuple[float, float],
    stick_end_xy: tuple[float, float],
    rotation_deg: float,
    mb_negative: bool,
    markerstick_style: dict[str, Any] | None,
    label_dicts: list[dict[str, Any]],
) -> MarkerStick:
    """
    Draw a markerstick from its geometry, for add_markerstick() of either diagram.

    marker_xy and stick_end_xy are (theta, r). rotation_deg turns the marker about its center
    and sets the labels' rotation: the marker's top, and the labels' out_offset, point away
    from the diagram's reference point (origin or star), and cw_offset runs clockwise about it.
    mb_negative mirrors the 'beside' preset (labels go opposite the stick).

    Returns:
        MarkerStick(marker, stick, labels), as add_markerstick().

    Raises:
        ValueError: a label's 'position' is not 'outside', 'inside' or 'beside'.
    """
    theta, r = marker_xy
    theta_end, r_end = stick_end_xy
    marker_overrides, stick_overrides, shared_color, marker_shape = _resolve_markerstick_style(
        markerstick_style
    )

    stick_zorder = stick_overrides.pop('zorder', 2)
    stick_props = {
        'linewidth': STICK_LINEWIDTH,
        'linestyle': '-',
        'color': shared_color,
        'alpha': ALPHA,
        **stick_overrides,
    }
    (stick,) = ax.plot(
        [theta, theta_end], [r, r_end], clip_on=False, zorder=stick_zorder, **stick_props
    )

    marker_zorder = marker_overrides.pop('zorder', 2)
    color_override = marker_overrides.pop('color', None)
    marker_defaults = {
        's': MARKER_SIZE**2,
        'facecolors': MARKER_FACECOLOR,
        'alpha': ALPHA,
        'edgecolors': color_override if color_override is not None else shared_color,
    }
    marker_props = {**marker_defaults, **marker_overrides}
    # rotate the marker about its centroid so its top points away from the reference point
    base_path = (
        MarkerStyle(marker_shape).get_path().transformed(MarkerStyle(marker_shape).get_transform())
    )
    centroid = _marker_centroid(base_path)
    m = MarkerStyle(
        marker_shape, transform=Affine2D().translate(*(-centroid)).rotate_deg(rotation_deg)
    )
    marker_artist = ax.scatter(
        [theta], [r], marker=m, clip_on=False, zorder=marker_zorder, **marker_props
    )
    # in the unrotated marker frame +y is outward and +x clockwise, as for the labels
    ccw_edge, cw_edge, in_edge, out_edge = _marker_extents_pt(
        base_path.transformed(Affine2D().translate(*(-centroid))),
        np.sqrt(marker_artist.get_sizes()[0]),
        marker_artist.get_linewidths()[0],
    )

    labels_artists = []
    if label_dicts:
        label_color = _default_label_color(
            marker_props['edgecolors'], marker_props['facecolors'], marker_props['alpha']
        )
        default_label_style = {
            'position': 'outside',
            'fontsize': LABEL_FONTSIZE,
            'color': label_color,
            'zorder': marker_zorder,
            'alpha': marker_props['alpha'],
        }
        for label_spec in label_dicts:
            label_spec = {**default_label_style, **label_spec.copy()}
            position = label_spec.pop('position', None)
            if position is not None:
                if position not in LABEL_POSITION_PRESETS:
                    raise ValueError(
                        "label_info 'position' must be 'outside', 'inside', or 'beside', "
                        f"got {position!r}"
                    )
                pos_props = dict(LABEL_POSITION_PRESETS[position])
                if position == 'beside' and mb_negative:  # opposite the stick
                    pos_props.update(ha='left', cw_offset=-pos_props['cw_offset'])
                label_spec = {**pos_props, **label_spec}  # label_spec keys win
            label_props = {**label_spec, 'rotation': rotation_deg}
            label_props.pop('text', None)
            fontsize = label_props['fontsize']
            for key, low, high in (
                ('out_offset', in_edge, out_edge),
                ('cw_offset', ccw_edge, cw_edge),
            ):
                shift = _edge_shift_pt(label_props.get(key, 0), fontsize, low, high)
                label_props[key] = shift / fontsize
            labels_artists.append(text_polar(ax, theta, r, label_spec['text'], **label_props))

    return MarkerStick(marker_artist, stick, labels_artists)


def _marker_centroid(path: Path) -> np.ndarray:
    """Center of a marker path: the area centroid of its outline if it is closed (last code
    CLOSEPOLY), else the center of its extents; (0, 0) for an empty path."""
    if not len(path.vertices):
        return np.zeros(2)
    if path.codes is not None and len(path.codes) and path.codes[-1] == Path.CLOSEPOLY:
        # shoelace formula over the outline, curves flattened at 1000x to follow them closely
        area2, moment = 0.0, np.zeros(2)
        for polygon in path.to_polygons(Affine2D().scale(1000)):
            p, q = polygon, np.roll(polygon, -1, axis=0)
            cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
            area2 += cross.sum()
            moment += ((p + q) * cross[:, None]).sum(axis=0)
        if area2:
            return moment / (3 * area2) / 1000
    ext = path.get_extents()
    return np.array([(ext.x0 + ext.x1) / 2, (ext.y0 + ext.y1) / 2])


def _marker_extents_pt(path: Path, size: float, edgewidth: float) -> tuple[float, ...]:
    """(x0, x1, y0, y1) of a marker as drawn about its point, in points, edge included: path in
    marker units (as drawn about the point), size and edgewidth in points."""
    pad = edgewidth / 2
    if not len(path.vertices):
        return -pad, pad, -pad, pad
    ext = path.get_extents()  # of the curves, not their control points
    return ext.x0 * size - pad, ext.x1 * size + pad, ext.y0 * size - pad, ext.y1 * size + pad


def _edge_shift_pt(offset: float, fontsize: float, low: float, high: float) -> float:
    """A label's shift from the marker center, in points, for an offset in multiples of fontsize:
    a nonzero offset is a gap beyond the marker's edge, at low (< 0) or high (> 0) points."""
    if offset > 0:
        return offset * fontsize + high
    if offset < 0:
        return offset * fontsize + low
    return 0.0


def _label_dicts(label_info: Any, where: str = 'label_info') -> list[dict[str, Any]]:
    """
    label_info (None, a dict or a list of dicts) as a list of dicts, each with a 'text' key.

    Raises:
        TypeError: label_info is not None, a dict or a list, or a list item is not a dict.
        ValueError: a dict has no 'text' key.
    """
    if label_info is None:
        return []
    if isinstance(label_info, dict):
        named = [(where, label_info)]
    elif isinstance(label_info, list):
        named = [(f'{where}[{i}]', item) for i, item in enumerate(label_info)]
    else:
        raise TypeError(
            f"{where} must be None, a dict or a list of dicts, got {type(label_info).__name__}"
        )
    for name, item in named:
        if not isinstance(item, dict):
            raise TypeError(f"{name} must be a dict, got {type(item).__name__}")
        if 'text' not in item:
            raise ValueError(f"{name} dict needs a 'text' key")
    return [item for _, item in named]


def _default_label_color(edgecolor: Any, facecolor: Any, alpha: float | None = None) -> Any:
    """
    Default marker-label color for add_markerstick() and add_labeledmarker_legend().

    The marker's edge color; if that is 'none' or fully transparent, its face color if
    visible; else 'k'. Visibility is judged as drawn: a non-None marker alpha replaces each
    color's own alpha.
    """
    edgecolor, facecolor = _single_color(edgecolor), _single_color(facecolor)
    if isinstance(edgecolor, str) and edgecolor == 'face':
        edgecolor = facecolor  # scatter's 'face' edge is drawn in the face color
    for color in (edgecolor, facecolor):
        if color is not None and to_rgba(color, alpha)[3] > 0:
            return color
    return 'k'


def _single_color(color: Any) -> Any:
    """A marker's one color: the first entry of a color sequence (e.g. ['g'] or a 1x4 RGBA array),
    since each markerstick is a single-point scatter; None for an empty sequence."""
    if color is None or isinstance(color, str) or is_color_like(color):
        return color
    return color[0] if len(color) else None


def _resolve_markerstick_style(
    markerstick_style: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """
    Split a markerstick_style dict into marker (ax.scatter) and stick (ax.plot) kwargs.

    Args:
        markerstick_style: Optional dict. Flat keys are routed:
            - marker only: 'marker' (shape string), 's', 'markersize', 'linewidths',
              'facecolors', 'edgecolors', 'c', 'cmap', 'norm', 'vmin', 'vmax', 'colorizer';
            - stick only: 'linestyle'/'ls', 'linewidth'/'lw', 'dashes', 'drawstyle'/'ds',
              'fillstyle', 'gapcolor', 'markevery', 'dash_capstyle', 'solid_capstyle',
              'dash_joinstyle', 'solid_joinstyle';
            - all others (e.g. 'color', 'alpha', 'zorder'): both.
            'marker' and 'stick' sub-dicts are merged last, overriding flat keys. 'markersize'
            (points) becomes s = markersize**2 unless 's' is also given, in which case it is
            dropped.

    Returns:
        (marker_props, stick_props, shared_color, marker_shape): kwargs for ax.scatter(),
        kwargs for ax.plot(), the flat 'color' (default 'r'), and the marker shape (default
        'o').
    """
    # marker-describing keys and scatter-only kwargs (Line2D rejects e.g. 'edgecolors', 'c')
    marker_only_keys = frozenset(
        {
            'marker',
            's',
            'markersize',
            'linewidths',
            'facecolors',
            'edgecolors',
            'c',
            'cmap',
            'norm',
            'vmin',
            'vmax',
            'colorizer',
        }
    )
    # 'linewidth' is the stick width; 'linewidths' (above) is the marker edge width
    stick_only_keys = frozenset(
        {
            'linestyle',
            'ls',
            'linewidth',
            'lw',
            'dashes',
            'drawstyle',
            'ds',
            'fillstyle',
            'gapcolor',
            'markevery',
            'dash_capstyle',
            'solid_capstyle',
            'dash_joinstyle',
            'solid_joinstyle',
        }
    )
    style = markerstick_style.copy() if markerstick_style else {}
    explicit_marker_ov: dict[str, Any] = (
        style.pop('marker', {}) if isinstance(style.get('marker'), dict) else {}
    )
    explicit_stick_ov: dict[str, Any] = (
        style.pop('stick', {}) if isinstance(style.get('stick'), dict) else {}
    )
    shared_color = style.get('color', MARKERSTICK_COLOR)
    marker_props: dict[str, Any] = {}
    stick_props: dict[str, Any] = {}
    for k, v in style.items():
        if k in marker_only_keys:
            marker_props[k] = v
        elif k in stick_only_keys:
            stick_props[k] = v
        else:
            marker_props[k] = v
            stick_props[k] = v
    marker_props.update(explicit_marker_ov)
    stick_props.update(explicit_stick_ov)
    if 'markersize' in marker_props and 's' not in marker_props:
        marker_props['s'] = float(marker_props.pop('markersize')) ** 2
    else:
        marker_props.pop('markersize', None)
    marker_shape = marker_props.pop('marker', MARKER_SHAPE)
    return marker_props, stick_props, shared_color, marker_shape
