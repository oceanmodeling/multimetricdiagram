"""Multimetric diagram: 5-metric alternative, that rewards realistic variance, to Taylor diagram

Metrics: total, bias, pattern, amplitude, phase.

Associated manuscript:
Codiga, Daniel L., "Improved Evaluation and Communication of Model–Observation Comparisons: The
Diagnostic Index of Agreement with Symmetry, and the Multimetric Diagram as an Alternative to the
Taylor Diagram", submitted manuscript, 2026.

Main functions
    calculate_metrics()
        Compute metrics from model-observation pairs (scalar or vector)
    make_fig_axes()
        Generate figure, main polar axes, legend container axes above
    add_markerstick()
        Add markerstick to polar axes, metrics for a model-observation pair
    adjust_markerstick_labels()
        Reposition markerstick labels to minimize overlap
    add_labeledmarker_legend()
        Add a marker legend, optionally with text labels on markers

See README.md for the explanation of diagram elements, the purposes of all
functions (including helper functions), usage examples, and notes about the code.

D. Codiga, with Claude agents, 11/2026
"""

from . import alt as alt  # pylint: disable=useless-import-alias  # explicit re-export
from ._text import text_polar
from .axes import make_fig_axes
from .labels import adjust_markerstick_labels
from .legend import add_labeledmarker_legend
from .markerstick import add_markerstick
from .metrics import calculate_metrics

__version__ = "0.1.0"

__all__ = [  # noqa: RUF022 (kept in workflow order)
    "calculate_metrics",
    "make_fig_axes",
    "add_markerstick",
    "adjust_markerstick_labels",
    "add_labeledmarker_legend",
    "text_polar",
]
