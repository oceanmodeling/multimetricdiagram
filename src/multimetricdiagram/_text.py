"""text_polar(): rotated text on polar axes (used by axes and markerstick; also public)."""

from typing import Any

import numpy as np
from matplotlib.artist import allow_rasterization
from matplotlib.font_manager import FontProperties
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D, Bbox, ScaledTranslation, Transform


def text_polar(
    ax: Any,
    theta: float,
    r: float,
    text: Any,
    *,
    color: str = 'k',
    rotation: float = 0,
    fontsize: float = 10,
    fontweight: str = 'normal',
    out_offset: float = 0,
    cw_offset: float = 0,
    ha: str = 'center',
    va: str = 'center',
    zorder: float = 2,
    **kwargs,
) -> PathPatch:
    """
    Draw rotated text on polar axes with precise alignment and offsets.

    Drawn as a PathPatch of a TextPath, because pyplot.text() with rotation is unreliable on
    polar axes.

    Args:
        ax: Polar axes.
        theta: Angular position, radians.
        r: Radial position, data units.
        text: Text to draw: any object, converted with str(); None gives '' (draws nothing).
        color: Text color. Default 'k'.
        rotation: Rotation, degrees counterclockwise. Default 0.
        fontsize: Font size, points. Default 10.
        fontweight: E.g. 'normal' (default) or 'bold'.
        out_offset: Shift perpendicular to the baseline, in multiples of fontsize; positive is
            toward the top of the text (radially outward when rotation = theta - 90 deg, as
            for markerstick labels). Default 0.
        cw_offset: Shift along the baseline, in multiples of fontsize; positive is toward the
            end of the text (clockwise when rotation = theta - 90 deg). Default 0.
        ha: Point of the text's ink box placed at (theta, r) horizontally: 'left' edge,
            'center' (default), or 'right' edge.
        va: Same, vertically: 'bottom' edge, 'center' (default), or 'top' edge.
        zorder: Default 2.
        **kwargs: Passed to PathPatch, e.g. alpha=0.5 (clip_on defaults to False).

    Returns:
        The PathPatch, for later changes, e.g. h = text_polar(...); h.set_facecolor('red').

    Raises:
        ValueError: ha or va is not one of the values above.
    """
    if ha not in ("center", "left", "right"):
        raise ValueError(f"ha must be 'center', 'left', or 'right', got {ha!r}")
    if va not in ("center", "bottom", "top"):
        raise ValueError(f"va must be 'center', 'bottom', or 'top', got {va!r}")
    clip_on = kwargs.pop('clip_on', False)
    text = '' if text is None else str(text)  # as matplotlib's Text.set_text()
    if text:
        path = TextPath((0, 0), text, size=1, prop=FontProperties(weight=fontweight))
    else:  # TextPath can't make an empty path; draw nothing
        path = Path(np.empty((0, 2)))
    patch = _PolarTextPatch(
        path,
        ax=ax,
        theta=theta,
        r=r,
        rotation=rotation,
        fontsize=fontsize,
        out_offset=out_offset,
        cw_offset=cw_offset,
        ha=ha,
        va=va,
        lw=0,
        zorder=zorder,
        clip_on=clip_on,
        **kwargs,
    )
    patch.set_color(color)  # set here, not in draw(), so later color changes persist
    # pylint: disable-next=protected-access,attribute-defined-outside-init
    patch._text = text  # e.g. for adjust_markerstick_labels()'s warning

    autoscale_was_on = ax.get_autoscale_on()
    ax.set_autoscale_on(False)
    ax.add_patch(patch)
    ax.set_autoscale_on(autoscale_was_on)

    return patch


class _PolarTextPatch(PathPatch):
    """
    text_polar()'s patch. Its transform is recomputed at each draw, since it depends on figure
    size, axes position and r-limits; an optional offset in inches is applied last.
    """

    def __init__(
        self,
        path: TextPath,
        *,
        ax: Any,
        theta: float,
        r: float,
        rotation: float,
        fontsize: float,
        out_offset: float,
        cw_offset: float,
        ha: str,
        va: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(path, **kwargs)
        self._ax = ax
        # empty text: a zero-size box at the anchor keeps the transform finite
        self._bb = path.get_extents() if len(path.vertices) else Bbox.from_bounds(0, 0, 0, 0)
        self._theta = theta
        self._r = r
        self._rotation = rotation
        self._fontsize = fontsize
        self._out_offset = out_offset
        self._cw_offset = cw_offset
        self._ha = ha
        self._va = va
        self._offset_inches = (0.0, 0.0)

    def _polar_transform(self) -> Affine2D:
        ax = self._ax
        fontsize = self._fontsize
        bb = self._bb
        cx = bb.x0 + 0.5 * (bb.x1 - bb.x0)
        cy = bb.y0 + 0.5 * (bb.y1 - bb.y0)
        half_width = 0.5 * (bb.x1 - bb.x0)
        half_height = 0.5 * (bb.y1 - bb.y0)
        # Cartesian-data-to-display part of polar transData (equal to transData._b)
        to_display = ax.transProjectionAffine + ax.transWedge + ax.transAxes
        unit_px = np.hypot(*(to_display.transform((1.0, 0.0)) - to_display.transform((0.0, 0.0))))
        pt_per_data = unit_px * 72.0 / ax.get_figure().dpi
        scale = fontsize / pt_per_data
        if self._ha == "left":
            ha_shift = half_width
        elif self._ha == "right":
            ha_shift = -half_width
        else:
            ha_shift = 0.0
        if self._va == "bottom":
            va_shift = half_height
        elif self._va == "top":
            va_shift = -half_height
        else:
            va_shift = 0.0
        x_offset = (self._cw_offset * fontsize) / pt_per_data
        y_offset = (self._out_offset * fontsize) / pt_per_data
        x = self._r * np.cos(self._theta)
        y = self._r * np.sin(self._theta)
        return (
            Affine2D()
            .translate(-cx + ha_shift, -cy + va_shift)  # 1. center glyphs, apply ha/va (em units)
            .scale(scale)  # 2. scale to data units
            .translate(x_offset, y_offset)  # 3. shift in local text frame (data units)
            .rotate_deg(self._rotation)  # 4. rotate into position
            .translate(x, y)  # 5. move to polar location
            + to_display
        )

    def _full_transform(self) -> Transform:
        transform = self._polar_transform()
        if self._offset_inches != (0.0, 0.0):
            # as offset_copy(..., units='inches'): scaled by the figure dpi at draw time
            dpi_scale_trans = self._ax.get_figure().dpi_scale_trans
            transform = transform + ScaledTranslation(*self._offset_inches, dpi_scale_trans)
        return transform

    def _set_offset_inches(self, dx: float, dy: float) -> None:
        """Set the extra (dx, dy) offset in inches, updating the transform immediately."""
        self._offset_inches = (dx, dy)
        self.set_transform(self._full_transform())

    @allow_rasterization  # lets set_rasterized(True) take effect
    def draw(self, renderer: Any) -> None:
        self.set_transform(self._full_transform())
        super().draw(renderer)
