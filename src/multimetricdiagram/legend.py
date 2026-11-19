"""Labeled-marker legend: add_labeledmarker_legend() and its legend handler."""

from typing import Any

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.legend import Legend
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.text import Text

from ._defaults import (
    ALPHA,
    LABEL_FONTSIZE,
    LABEL_POSITION_PRESETS,
    LEGEND_MARKER_EDGEWIDTH,
    MARKER_FACECOLOR,
    MARKER_SIZE,
)
from .markerstick import (
    _default_label_color,
    _edge_shift_pt,
    _label_dicts,
    _marker_extents_pt,
    _resolve_markerstick_style,
    _single_color,
)

# least gap (legend fontsize units) from a marker label to another row or the frame
_LABEL_CLEARANCE = 0.4
_MEASURE_DPI = 300  # text is measured at this dpi, whatever the figure's


def _resolve_legend_label_props(
    lspec: dict[str, Any], shared_color: str
) -> tuple[str, float, float, float, dict[str, Any], str]:
    """
    Apply a 'position' preset to one legend label spec; split off its positioning values.

    Offsets mean what they do in add_markerstick()'s label_info, except that out_offset is
    measured upward and cw_offset rightward (a legend marker has no stick). 'rotation' is
    dropped.

    Returns:
        (text, fontsize, out_offset, cw_offset, remaining_props, position); remaining_props are
        Text kwargs, with 'color' defaulting to shared_color.

    Raises:
        ValueError: 'position' is not 'outside', 'inside' or 'beside'.
    """
    lspec = lspec.copy()
    position = lspec.pop('position', 'outside')
    if position not in LABEL_POSITION_PRESETS:
        raise ValueError(
            f"label_info 'position' must be 'outside', 'inside', or 'beside', got {position!r}"
        )
    lspec = {**LABEL_POSITION_PRESETS[position], **lspec}
    fontsize = lspec.pop('fontsize', LABEL_FONTSIZE)
    out_offset = lspec.pop('out_offset', 0)
    cw_offset = lspec.pop('cw_offset', 0)
    lspec.pop('rotation', None)  # rotation not meaningful for a legend label
    text = lspec.pop('text')
    lspec.setdefault('color', shared_color)
    return text, fontsize, out_offset, cw_offset, lspec, position


class _LabeledMarkerHandler(HandlerLine2D):
    """
    HandlerLine2D that also draws a legend row's marker labels.

    label_specs: (text, fontsize, cw_offset, out_offset, text_props) per label; offsets in
    multiples of fontsize, rightward and upward, nonzero ones from the drawn marker's edge.
    marker_xy: marker position in points from the handle box's left edge and the row's baseline
    (None: HandlerLine2D's). Each use sets box, the handle box's (xdescent, ydescent, width,
    height, fontsize) in points; label_shifts, each label's (dx, dy) in points from the marker
    center; and labels, the row's label Texts.
    """

    def __init__(self, label_specs=()):
        super().__init__()
        self.label_specs = list(label_specs)
        self.marker_xy: tuple[float, float] | None = None
        self.box: tuple | None = None
        self.label_shifts: list[tuple[float, float]] = []
        self.labels: list[Text] = []

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        self.box = (xdescent, ydescent, width, height, fontsize)
        (legline,) = super().create_artists(
            legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
        )
        # box coordinates in points, y = 0 at the row's text baseline
        x0, y0 = -xdescent + width / 2, (height - ydescent) / 2  # HandlerLine2D's marker spot
        x, y = (
            (x0, y0)
            if self.marker_xy is None
            else (-xdescent + self.marker_xy[0], self.marker_xy[1])
        )
        if self.get_numpoints(legend) == 1:
            legline.set_data([x], [y])  # the marker alone: legend rows draw no line
            legline.set_markevery(None)
        else:
            legline.set_data(
                np.asarray(legline.get_xdata()) + (x - x0),
                np.asarray(legline.get_ydata()) + (y - y0),
            )
        left, right, bottom, top = _line_marker_extents_pt(legline)  # markerscale applied
        self.label_shifts = [
            (_edge_shift_pt(cw, size, left, right), _edge_shift_pt(out, size, bottom, top))
            for _text, size, cw, out, _props in self.label_specs
        ]
        self.labels = [
            Text(x + dx, y + dy, text, fontsize=size, transform=trans, clip_on=False, **props)
            for (text, size, _cw, _out, props), (dx, dy) in zip(
                self.label_specs, self.label_shifts
            )
        ]
        return [legline, *self.labels]


def _text_renderer() -> Any:
    """A renderer for measuring text without drawing; its fixed dpi keeps the layout independent
    of the figure's dpi."""
    return RendererAgg(1, 1, _MEASURE_DPI)


def _text_extent_pt(text: Text, fig: Any, renderer: Any) -> tuple[float, float, float, float]:
    """(x0, x1, y0, y1) of a Text's box about its anchor, in points, as matplotlib lays it out."""
    text.set_figure(fig)  # needed to measure; the Text is not added to the figure
    to_pt = 1 / renderer.points_to_pixels(1.0)
    bbox = text.get_window_extent(renderer, dpi=72 / to_pt)  # layout at the renderer's dpi
    return bbox.x0 * to_pt, bbox.x1 * to_pt, bbox.y0 * to_pt, bbox.y1 * to_pt


def _line_marker_extents_pt(line: Line2D) -> tuple[float, ...]:
    """(x0, x1, y0, y1) of a Line2D's marker about its point, including its edge, in points."""
    style = MarkerStyle(line.get_marker())
    return _marker_extents_pt(
        style.get_path().transformed(style.get_transform()),
        line.get_markersize(),
        line.get_markeredgewidth(),
    )


def _marker_half_extent_pt(line: Line2D) -> tuple[float, float]:
    """Half width and half height of a Line2D's marker, including its edge, in points."""
    x0, x1, y0, y1 = _line_marker_extents_pt(line)
    return max(-x0, x1), max(-y0, y1)


def _kwarg_or_rc(legend_kwargs: dict[str, Any], name: str) -> float:
    value = legend_kwargs.get(name)
    return mpl.rcParams[f'legend.{name}'] if value is None else value


def _blank_title(height: float, fig: Any, renderer: Any) -> tuple[float, float]:
    """(fontsize, row height in points) of the shortest blank legend title row at least height
    points tall; matplotlib's smallest font size is 1 pt."""
    size = 1.0
    for _ in range(20):
        _, _, y0, y1 = _text_extent_pt(Text(text=' ', fontsize=size), fig, renderer)
        if y1 - y0 >= height:
            break
        size *= max(height / (y1 - y0), 1.01)
    return size, y1 - y0


def _layout_for_labels(
    ax: Any, handles: list, labels: list, handlers: list, legend_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """
    Legend kwargs that give the marker labels room (legend fontsize units; values passed are
    minimums); sets each handler's marker_xy.

    Labels and markers may reach below a row (row gap, border padding), above it (row gap; above
    the top row, a blank title row or a larger borderpad, whichever adds less to height plus
    width) and left of the handle box (border padding or column gap); they need only clear what
    they overlap horizontally. With one column (rows in entry order) a row need only clear the
    rows below it, over the row gaps between, and only the top row needs room above, the bottom
    row room below; with several columns any row may sit above any row. Markers stay level with
    their row's text. Handle box size and font come from a probe Legend that is never drawn.
    """
    probe = Legend(
        ax, handles, labels, **{k: v for k, v in legend_kwargs.items() if k != 'draggable'}
    )
    _xdescent, ydescent, width, height, fontsize = handlers[0].box
    fig = ax.get_figure()
    renderer = _text_renderer()
    font = probe.get_texts()[0].get_fontproperties()
    _, _, lp_bottom, lp_top = _text_extent_pt(
        Text(text='lp', fontproperties=font, va='baseline'), fig, renderer
    )
    text_widths = []
    for label in labels:
        x0, x1, _, _ = _text_extent_pt(Text(text=label, fontproperties=font), fig, renderer)
        text_widths.append(x1 - x0)
    marker_y = (height - ydescent) / 2  # HandlerLine2D's marker height
    # per row: marker and label boxes (x0, x1, y0, y1), x from the marker center, y from baseline
    rows = []
    for line, handler in zip(probe.legend_handles, handlers):
        half_w, half_h = _marker_half_extent_pt(line)
        items = [(-half_w, half_w, marker_y - half_h, marker_y + half_h)]
        for (text, size, _cw, _out, props), (dx, dy) in zip(
            handler.label_specs, handler.label_shifts
        ):
            x0, x1, y0, y1 = _text_extent_pt(
                Text(text=text, fontsize=size, **props), fig, renderer
            )
            items.append((dx + x0, dx + x1, marker_y + dy + y0, marker_y + dy + y1))
        rows.append(items)
    left = max(-item[0] for items in rows for item in items)
    right = max(item[1] for items in rows for item in items)
    ncols = legend_kwargs.get('ncols') or legend_kwargs.get('ncol') or 1
    # one column keeps the entry order: only the top row needs room above, the bottom row below
    top_rows, bottom_rows = (rows[:1], rows[-1:]) if ncols == 1 else (rows, rows)
    top = max(item[3] for items in top_rows for item in items)
    bottom = min(item[2] for items in bottom_rows for item in items)

    clear = _LABEL_CLEARANCE * fontsize
    textpad = _kwarg_or_rc(legend_kwargs, 'handletextpad') * fontsize
    markerfirst = legend_kwargs.get('markerfirst', True)
    column_gap = _kwarg_or_rc(legend_kwargs, 'columnspacing') * fontsize if ncols > 1 else np.inf
    # a row reaches at least as far as its handle box and its text's "lp" height
    row_top, row_bottom = max(height - ydescent, lp_top), min(-ydescent, lp_bottom)
    row_height = row_top - row_bottom

    def place(borderpad):
        """Handle box width, marker x and labelspacing for this borderpad."""
        # room left of the handle box: border padding (first column), column gap (others)
        overhang = max(0.0, min(borderpad * fontsize, column_gap) - clear) if markerfirst else 0.0
        box_width = max(width, left + right - overhang)
        marker_x = min(max(box_width / 2, left - overhang), box_width - right)
        # per row, x from the handle box's left edge: marker and labels, then the text
        placed = []
        for items, text_width in zip(rows, text_widths):
            text_x = box_width + textpad if markerfirst else -textpad - text_width
            placed.append(
                [(marker_x + x0, marker_x + x1, y0, y1) for x0, x1, y0, y1 in items]
                + [(text_x, text_x + text_width, row_bottom, row_top)]
            )
        # row pitch: items of two rows (except text with text) stay `clear` apart where they
        # overlap horizontally; one column: row a above row b (a < b), b - a pitches apart;
        # several columns: any row above any row, itself included
        distance = row_height
        for a, upper in enumerate(placed):
            for b, lower in enumerate(placed):
                if ncols == 1 and b <= a:
                    continue
                steps = b - a if ncols == 1 else 1
                for i, p in enumerate(upper):
                    for j, q in enumerate(lower):
                        both_text = i == len(upper) - 1 and j == len(lower) - 1
                        if not both_text and p[0] < q[1] + clear and q[0] < p[1] + clear:
                            distance = max(distance, (q[3] + clear - p[2]) / steps)
        labelspacing = max(
            _kwarg_or_rc(legend_kwargs, 'labelspacing'), (distance - row_height) / fontsize
        )
        return box_width, marker_x, labelspacing

    borderpad = max(
        _kwarg_or_rc(legend_kwargs, 'borderpad'), (row_bottom - bottom + clear) / fontsize
    )
    box_width, marker_x, labelspacing = place(borderpad)
    kwargs: dict[str, Any] = {}
    room = top - row_top + clear  # needed above the top row's box
    if legend_kwargs.get('title'):  # the title sits a labelspacing above the top row
        labelspacing = max(labelspacing, room / fontsize)
    elif room > borderpad * fontsize:
        title_size, title_height = _blank_title(
            room - (borderpad + labelspacing) * fontsize, fig, renderer
        )
        # the title row adds height only; a larger borderpad adds to all four sides
        if title_height + labelspacing * fontsize <= 4 * (room - borderpad * fontsize):
            kwargs.update(title=' ', title_fontsize=title_size, title_fontproperties=None)
        else:
            borderpad = room / fontsize
            box_width, marker_x, labelspacing = place(borderpad)
    for handler in handlers:
        handler.marker_xy = (marker_x, marker_y)
    kwargs.update(
        handlelength=box_width / fontsize, borderpad=borderpad, labelspacing=labelspacing
    )
    return kwargs


def add_labeledmarker_legend(
    ax: Any,
    entries: dict[str, Any] | list[dict[str, Any]] | tuple[dict[str, Any], ...],
    *,
    loc: str = 'best',
    bbox_to_anchor: tuple | None = None,
    **legend_kwargs: Any,
) -> tuple[Any, list[Any], list[list[Any]]]:
    """
    Add a legend of add_markerstick()-style markers (no sticks), optionally with marker labels.

    matplotlib's legends cannot put text on a marker; without marker labels, ax.legend() is
    simpler. Each row's marker and labels are drawn by a legend handler and laid out at draw
    time, so the legend can be added at any point and the figure saved at any dpi.
    handlelength, labelspacing and borderpad are enlarged to fit the labels (values passed are
    minimums); rows are evenly spaced. Without a title, a blank title (' ') may be added for
    room above the top row: get_title() then returns ' ', a later set_title() may overlap the
    top row's labels (pass title= here instead), and a title_fontproperties given without a
    title is ignored.

    It calls ax.legend(), which replaces any earlier ax.legend() legend. To keep another legend
    on the same axes, make it first and add it with ax.add_artist(), e.g.
        leg = ax.legend(handles, labels, loc='center right')
        ax.add_artist(leg)
        add_labeledmarker_legend(ax, entries, loc='center left')
    (https://matplotlib.org/stable/users/explain/axes/legend_guide.html, "Multiple legends on
    the same Axes").

    Args:
        ax: Axes to hold the legend.
        entries: One dict, or a list of dicts, one per legend row, top to bottom:
            - 'label': the legend text, any object, converted with str(); None gives ''
              (matplotlib's default color, black).
            - 'markerstick_style': optional, as in add_markerstick() (same defaults); only
              marker keys are used, and 'zorder' is ignored.
            - 'label_info': optional (default None, no labels), as in add_markerstick(). Its
              'position' 'outside' (default), 'inside' or 'beside' puts a label above, below or
              left of the marker; offsets are in multiples of the label's fontsize, and a
              nonzero one is a gap from the legend marker's edge (default 0.1). 'alpha'
              defaults to the marker's; 'color' to the marker's edge color, or, if that is
              'none' or fully transparent as drawn (after the marker alpha), its face color if
              visible, else 'k'. 'rotation' is ignored.
        loc, bbox_to_anchor, **legend_kwargs: Passed to ax.legend() ('numpoints' defaults to
            1). 'handlelength', 'labelspacing' and 'borderpad' are minimums.

    Returns:
        (legend_artist, marker_handles, labels_artists): the Legend from ax.legend(); its row
        marker artists (Line2D), one per entry in entry order, not linked to any plotted
        markerstick; and one list of label Text artists per entry (empty if no label_info).

    Raises:
        ValueError: an entry has no 'label' key, a label dict has no 'text' key, or a label's
            'position' is invalid.
        TypeError: entries is not a dict, a list or a tuple, or an item of it is not a dict; a
            'label_info' is not None, a dict or a list, or one of its list items is not a dict;
            or dpi is passed (no longer accepted; the layout does not depend on dpi).
    """
    if 'dpi' in legend_kwargs:
        raise TypeError(
            "add_labeledmarker_legend() no longer accepts dpi: the legend layout does not depend "
            "on dpi; remove the dpi= argument"
        )
    if isinstance(entries, dict):
        named = [('entries', entries)]
    elif isinstance(entries, (list, tuple)):
        named = [(f'entries[{i}]', entry) for i, entry in enumerate(entries)]
    else:
        raise TypeError(
            f"entries must be a dict or a list (or tuple) of dicts, got {type(entries).__name__}"
        )
    handles = []
    labels = []
    label_specs = []
    for name, entry in named:
        if not isinstance(entry, dict):
            raise TypeError(f"{name} must be a dict, got {type(entry).__name__}")
        if 'label' not in entry:
            raise ValueError(f"{name} dict needs a 'label' key")
        label_dicts = _label_dicts(entry.get('label_info'), f"{name}['label_info']")
        marker_overrides, _stick_overrides, shared_color, marker_shape = (
            _resolve_markerstick_style(entry.get('markerstick_style'))
        )
        marker_overrides.pop('zorder', None)  # zorder not used for legend markers
        color_override = marker_overrides.pop('color', None)
        s = marker_overrides.pop('s', MARKER_SIZE**2)
        # Line2D takes one color, not a sequence
        facecolor = _single_color(marker_overrides.pop('facecolors', MARKER_FACECOLOR))
        edgecolor = _single_color(
            marker_overrides.pop(
                'edgecolors', color_override if color_override is not None else shared_color
            )
        )
        edgewidth = marker_overrides.pop('linewidths', LEGEND_MARKER_EDGEWIDTH)
        alpha = marker_overrides.pop('alpha', ALPHA)
        handles.append(
            Line2D(
                [],
                [],
                marker=marker_shape,
                linestyle='none',
                markerfacecolor=facecolor,
                markeredgecolor=edgecolor,
                markeredgewidth=edgewidth,
                markersize=np.sqrt(s),
                alpha=alpha,
                **marker_overrides,
            )
        )
        labels.append(entry['label'])

        specs = []
        for lspec in label_dicts:
            text, size, out_offset, cw_offset, props, _position = _resolve_legend_label_props(
                lspec, _default_label_color(edgecolor, facecolor, alpha)
            )
            props.setdefault('alpha', alpha)
            specs.append((text, size, cw_offset, out_offset, props))
        label_specs.append(specs)

    handlers = [_LabeledMarkerHandler(specs) for specs in label_specs]
    legend_kwargs['handler_map'] = {
        **(legend_kwargs.get('handler_map') or {}),
        **dict(zip(handles, handlers)),
    }
    legend_kwargs.setdefault('numpoints', 1)
    if handles:
        legend_kwargs.update(_layout_for_labels(ax, handles, labels, handlers, legend_kwargs))
    legend_artist = ax.legend(
        handles, labels, loc=loc, bbox_to_anchor=bbox_to_anchor, **legend_kwargs
    )
    return legend_artist, legend_artist.legend_handles, [handler.labels for handler in handlers]
