"""Taylor diagram with the multimetric diagram's markersticks, labels and legend.

Each model is a marker at radius MSD/OSD and angle arccos(CC); the observations are the star at
radius 1 and CC = 1, so a marker's distance from the star is URMSD/OSD (Taylor 2001,
https://doi.org/10.1029/2000JD900719). The stick, of length |MB/OSD|, is tangent to the circle
about the star through the marker, clockwise about the star if MB > 0, so with ddof = 0 it ends
on the circle of RMSD/OSD. Gray gridlines: radials of constant CC (dotted), circles about the
origin of constant MSD/OSD (dashed), circles about the star of constant RMSD/OSD = URMSD/OSD
(solid).

Functions
    calculate_metrics()
        Metrics of a model-observation pair: the top-level ones plus their ratios to OSD
    make_fig_axes()
        Figure with the diagram's polar axes and an empty legend axes above it
    add_markerstick()
        Add a markerstick (marker, stick, optional labels) for one metrics dict
    adjust_markerstick_labels(), add_labeledmarker_legend(), text_polar()
        The top-level functions themselves

make_fig_axes() options: quadrants=None (default: 2 if any dict passed as metrics= has CC < 0,
else 1), or 1 (CC 0 to 1) or 2 (CC -1 to 1) to override; normalized=True (radii in units of
OSD, default) or False (data units: then units and osd are needed, and r_max or metrics);
r_max=None (default: fits the largest MSD in metrics=, with a labeled outer arc; else 1.5);
grid_info (gridline styles, outer arc); pad_t_b_l_r_inch (room for labels); patternonly=False
(True: axis titles without RMSD and "(marker)", for metrics from
calculate_metrics(patternonly=True), which sets MB to 0 and RMSD to its zero-bias value, so
there are no sticks).

Example:
    >>> import numpy as np
    >>> import multimetricdiagram as mmd
    >>> taylor = mmd.alt.taylor
    >>> rng = np.random.default_rng(0)
    >>> obs = rng.normal(10.0, 2.0, 200)
    >>> metrics = {}
    >>> for name, gain, bias in [("A", 0.9, 0.3), ("B", 1.2, -0.4)]:
    ...     model = gain * (obs - 10) + 10 + bias + rng.normal(0, 0.8, obs.size)
    ...     metrics[name] = taylor.calculate_metrics(model, obs)
    >>> fig, ax, ax_legend = taylor.make_fig_axes(metrics=list(metrics.values()))
    >>> results = [taylor.add_markerstick(ax, m, label_info={"text": name})
    ...            for name, m in metrics.items()]
    >>> moved = taylor.adjust_markerstick_labels(ax, results)
    >>> fig.savefig("taylor.png", dpi=300)
"""

from ..._text import text_polar
from ...labels import adjust_markerstick_labels
from ...legend import add_labeledmarker_legend
from .axes import make_fig_axes
from .markerstick import add_markerstick
from .metrics import calculate_metrics

__all__ = [  # noqa: RUF022 (kept in workflow order)
    "calculate_metrics",
    "make_fig_axes",
    "add_markerstick",
    "adjust_markerstick_labels",
    "add_labeledmarker_legend",
    "text_polar",
]
