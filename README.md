[![Available on pypi](https://img.shields.io/pypi/v/multimetricdiagram.svg)](https://pypi.org/project/multimetricdiagram/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/multimetricdiagram.svg)](https://anaconda.org/conda-forge/multimetricdiagram)
[![CI](https://github.com/oceanmodeling/multimetricdiagram/actions/workflows/run_tests.yml/badge.svg)](https://github.com/oceanmodeling/multimetricdiagram/actions/workflows/run_tests.yml)
[![Pylint](https://github.com/oceanmodeling/multimetricdiagram/actions/workflows/pylint.yml/badge.svg?branch=main)](https://github.com/oceanmodeling/multimetricdiagram/actions/workflows/pylint.yml)
[![codecov](https://codecov.io/gh/oceanmodeling/multimetricdiagram/branch/main/graph/badge.svg)](https://codecov.io/gh/oceanmodeling/multimetricdiagram)

![multimetricdiagram logo](https://raw.githubusercontent.com/oceanmodeling/multimetricdiagram/main/docs/logo.png)

# multimetricdiagram

This package enables creation of multimetric diagrams. A multimetric diagram is a polar plot to visualize quantified differences/similarities of paired model and observation scalar or vector arrays (e.g., an array of scalar or vector model values and an array of associated observations; can be time series, spatial data, or any index). This includes (for each pair, through one markerstick on the plot) the total root mean square difference (RMSD) normalized by the model and observation standard deviations in quadrature (SDQ), its bias and pattern components, and the amplitude and phase contributions to the pattern component. (Amplitude and phase are used here, for convenience, in the broad sense that does not assume periodic signals: amplitude refers to the strength of variability, as measured by the standard deviation or variance, and phase refers to the degree to which model and observations vary together, as measured by the correlation coefficient.) The markerstick tip-ends are ordered by RMSD/SDQ and hence by their values of the Diagnostic Index of Agreement with Symmetry (DIAS). The multimetric diagram and DIAS are framed here in terms of model-observation pairs but they are also applicable to model-model comparisons or evaluations of paired observations collected by different measurement methods. The multimetric diagram is similar to a normalized Taylor diagram that includes bias sticks, with the main exceptions being that (1) the perfection star is co-located with the polar plot origin, and (2) it orders the model-observation comparisons, indicated by the distance between stick-ends of the markersticks and the 'perfection' star, by DIAS (not Nash-Sutcliffe efficiency, as in a Taylor diagram) and therefore rewards realistic variance (unlike the reward of NSE for damped variance).

## Installation

Requires Python 3.11 or later. Dependencies: numpy 1.26 or later, matplotlib 3.8 or later.

### PyPI

    pip install multimetricdiagram

### Conda / Mamba

    mamba install -c conda-forge multimetricdiagram

### From source (local, editable)

From this folder:

    pip install -e .

With conda: install the dependencies from conda-forge, then the package without dependencies:

    conda install -c conda-forge numpy matplotlib
    pip install --no-deps -e .

The usage examples below assume the package is imported as:

    import multimetricdiagram as mmd

## Explanation of diagram elements

![Definitions figure: (a) an example model and observation time series, (b) the corresponding metrics, and (c) the resulting markerstick (orange); the Pythagorean triangle (blue) relating RMSD/SDQ (total), MB/SDQ (bias) and URMSD/SDQ (pattern); and green labels marking where amplitude or phase contributions to the pattern difference are dominant.](https://raw.githubusercontent.com/oceanmodeling/multimetricdiagram/main/docs/definitions_mmd.png)

*Definitions figure: (a) an example model and observation time series, (b) the corresponding metrics, and (c) the resulting markerstick (orange); the Pythagorean triangle (blue) relating RMSD/SDQ (total), MB/SDQ (bias) and URMSD/SDQ (pattern); and green labels marking where amplitude or phase contributions to the pattern difference are dominant.*

* Each model-observation comparison is shown as a single markerstick. A markerstick is a marker symbol with a 'stick', which is a line tangential to the circular gridline through the marker location, emanating from the marker.

* Perfect model-observation agreement corresponds to a zero-length stick with marker positioned at the origin, indicated by the star.

* The distance of the tip or stick-end of the markerstick (end opposite to the marker) from the origin is an indication of the total difference metric: RMSD/SDQ, which is the Root-Mean-Square Difference RMSD normalized by SDQ, the Standard Deviations in Quadrature (Model Standard Deviation MSD, and Observation Standard Deviation OSD, in quadrature). The standard deviations are defined in the standard ways for scalars (square root of summed squared anomalies relative to mean, normalized by the number of values n) and vectors (square root of summed squared vector anomalies relative to mean, normalized by n).

* The length of the stick (distance between marker and stick-end) indicates the normalized mean bias metric MB/SDQ (Mean Bias normalized by SDQ), positive/negative for stick emanating from the marker in the clockwise/counterclockwise direction. For vectors, MB is the magnitude of the difference between the mean model vector and the mean observed vector, positive when that difference has a component along the mean observed vector and negative when it has a component opposite to it (positive for the edge case of the difference vector perpendicular to mean observation vector).

* The distance of the marker from the origin is an indication of the pattern difference metric: URMSD/SDQ (unbiased RMSD normalized by SDQ), which by definition has maximum value sqrt(2).

* The right-angle triangle geometry of (1) the hypotenuse distance from the origin to the stick-end (RMSD/SDQ), (2) the distance from the origin to the marker (URMSD/SDQ), and (3) the length of the stick (MB/SDQ) embodies the "total = bias + pattern" Pythagorean relationship (RMSD/SDQ)^2 = (URMSD/SDQ)^2 + (MB/SDQ)^2. Thus it presents the relative proportions of RMSD/SDQ, URMSD/SDQ, and MB/SDQ, and equivalently the relative proportions of the un-normalized RMSD, URMSD, and MB.

* The position of the marker relative to the grayscale shading (see grayscale bar key below bottom of polar plot) indicates MSD/OSD, the metric for the amplitude contribution to the pattern difference. The vertical axis of the plot is where MSD=OSD, perfect agreement. MSD/OSD changes with distance from there, increasing to the right and decreasing to the left, asymmetrically; the Variance Similarity VS = 2·MSD·OSD/(MSD² + OSD²) = MSD·OSD/(0.5·SDQ²) (range 0 to 1) is also 1 on the vertical axis, and decreases symmetrically with marker distance from there rightward and leftward. There is a one to one relationship between VS and MSD/OSD values: VS = 2·(MSD/OSD) / [(MSD/OSD)² + 1].

* For scalars, the Pearson correlation coefficient CC (range 1 to -1), the metric for phase agreement, is indicated by the position of the marker relative to the white contours. CC is 1 at the origin and along the radial boundaries of the polar plot at theta=-45 and -135 deg inside the unit circle, and -1 along those boundaries outside it (where theta is defined positive counter-clockwise from the rightward direction); CC is positive (dotted and solid white contours) inside the unit circle, zero at the unit circle, and negative (dashed white contours) outside the unit circle. For vectors, CC is the vector correlation coefficient based on the dot product of the vector anomalies.

* The location of the marker relative to the gray shading and the white contours for CC embodies the definitional relationship (URMSD/SDQ)^2 = 1 - VS\*CC among the pattern difference and its amplitude and phase contributions, in which there are two contributions to increasing URMSD/SDQ: (a) the extent to which VS is less than 1 (amplitude), and (b) the extent to which CC is less than 1 (phase).

* A marker located on the dotted black radial grid lines (at 67.5 degrees leftward and rightward from the vertically upward axis) indicates equal contributions to URMSD/SDQ, per (URMSD/SDQ)^2 = 1 - VS\*CC, by (a) the extent to which VS is less than 1 (amplitude), and (b) the extent to which CC is less than 1 (phase).

* A marker located outside that sector indicates the main contributor to URMSD/SDQ is the extent to which VS is less than 1 (amplitude); conversely, marker located in upper central sector of polar plot between the two dotted black radial grid lines indicates the main contributor to URMSD/SDQ is the extent to which CC is less than 1 (phase).

## Usage examples:

Save figures at dpi=300 or more for clear images (the layout does not depend on dpi; matplotlib's default is 100 dpi).

The examples use these lists, one entry per station: stnslist (station names), depth (bottom depths), TempSurfaceMod/TempSurfaceObs and TempBottomMod/TempBottomObs (temperature time series, arrays of shape (n,)), and CurrentSurfaceMod/CurrentSurfaceObs (velocity time series, arrays of shape (n, 2)). Runnable versions are in the usage_examples folder: example1.py to example6.py, with synthetic data from example_data.py. Run them from that folder, e.g. python example1.py; each saves its plot there (example1.png, ...).

Ex 1: One group of model-observation pairs; default marker style, no marker labels; legend in the legend axes above the diagram; legend by Matplotlib's built-in ax.legend(), simpler and more direct than add_labeledmarker_legend() because no marker labels are needed.

    fig, ax_polar, ax_legend = mmd.make_fig_axes()
    for mod, obs in zip(TempSurfaceMod, TempSurfaceObs):
        ms = mmd.add_markerstick(ax_polar, mmd.calculate_metrics(mod, obs))
    ax_legend.legend([ms.marker], ['Near-Surface Temperature'], loc='center')
    fig.savefig('example1.png', dpi=300)

Ex 2: Same as Ex 1, but: one label per markerstick; labeledmarker legend.

    fig, ax_polar, ax_legend = mmd.make_fig_axes()
    for stn, mod, obs in zip(stnslist, TempSurfaceMod, TempSurfaceObs):
        mmd.add_markerstick(ax_polar, mmd.calculate_metrics(mod, obs), label_info={'text': stn})
    legend_entry = {'label': 'Near-Surface Temperature', 'label_info': {'text': 'StnID'}}
    mmd.add_labeledmarker_legend(ax_legend, legend_entry, loc='center')
    fig.savefig('example2.png', dpi=300)

Ex 3: Same as Ex 2, but: near-bottom temperature; two labels per markerstick; custom markerstick style; legend at lower left on polar axes.

    fig, ax_polar, _ = mmd.make_fig_axes()  # legend axes not used
    style = {'marker': 's', 'color': 'b'}
    for stn, z, mod, obs in zip(stnslist, depth, TempBottomMod, TempBottomObs):
        labels = [{'text': stn, 'position': 'beside'}, {'text': z, 'position': 'inside'}]
        mmd.add_markerstick(ax_polar, mmd.calculate_metrics(mod, obs),
                            markerstick_style=style, label_info=labels)
    legend_labels = [{'text': 'StnID', 'position': 'beside'},
                     {'text': 'Depth [m]', 'position': 'inside'}]
    legend_entry = {'label': 'Near-Bottom Temperature', 'markerstick_style': style,
                    'label_info': legend_labels}
    # legend inside the diagram: its upper-left corner at (0.02, 0.42), in fractions of ax_polar
    mmd.add_labeledmarker_legend(ax_polar, legend_entry,
                                 loc='upper left', bbox_to_anchor=(0.02, 0.42))
    fig.savefig('example3.png', dpi=300)

Ex 4: Groups of Ex 2 and Ex 3 superposed on one diagram; labeledmarker legend with two items (one add_labeledmarker_legend() call, 2-item list passed); legend at lower left on polar axes.

    fig, ax_polar, _ = mmd.make_fig_axes()  # legend axes not used
    style = {'marker': 's', 'color': 'b'}
    for i, stn in enumerate(stnslist):
        surface = mmd.calculate_metrics(TempSurfaceMod[i], TempSurfaceObs[i])
        mmd.add_markerstick(ax_polar, surface, label_info={'text': stn})
        bottom = mmd.calculate_metrics(TempBottomMod[i], TempBottomObs[i])
        labels = [{'text': stn, 'position': 'beside'}, {'text': depth[i], 'position': 'inside'}]
        mmd.add_markerstick(ax_polar, bottom, markerstick_style=style, label_info=labels)
    legend_labels = [{'text': 'StnID', 'position': 'beside'},
                     {'text': 'Depth [m]', 'position': 'inside'}]
    mmd.add_labeledmarker_legend(ax_polar, [
        {'label': 'Near-Surface Temperature', 'label_info': {'text': 'StnID'}},
        {'label': 'Near-Bottom Temperature', 'markerstick_style': style, 'label_info': legend_labels},
    ], loc='upper left', bbox_to_anchor=(0.02, 0.42))
    fig.savefig('example4.png', dpi=300)

Ex 5: Vector inputs (near-surface current); six model-observation pairs whose markersticks crowd together, labels repositioned by adjust_markerstick_labels() to avoid overlap.

    fig, ax_polar, _ = mmd.make_fig_axes()  # legend axes not used
    style = {'marker': '^'}
    results = []
    for stn, mod, obs in zip(stnslist, CurrentSurfaceMod, CurrentSurfaceObs):
        metrics = mmd.calculate_metrics(mod, obs)
        results.append(mmd.add_markerstick(ax_polar, metrics, markerstick_style=style,
                                           label_info={'text': stn}))
    mmd.adjust_markerstick_labels(ax_polar, results)
    legend_entry = {'label': 'Near-Surface Current', 'markerstick_style': style,
                    'label_info': {'text': 'StnID'}}
    mmd.add_labeledmarker_legend(ax_polar, legend_entry,
                                 loc='upper left', bbox_to_anchor=(0.02, 0.42))
    fig.savefig('example5.png', dpi=300)

Ex 6: Same as Ex 2, but on a Taylor diagram (multimetricdiagram.alt.taylor). The metrics are passed to make_fig_axes() so it can choose one- or two-quadrant variant, and the outer radius. Two example figures: patternonly=True, bias set to zero, only markers with no bias sticks, example6_patternonly.png; then default (patternonly=False) so bias sticks are shown, example6.png. In the latter, note the stations appear in different order, by distance of stick end to star, compared to Ex 2; the ordering is by NSE here and by DIAS in Ex 2.

    taylor = mmd.alt.taylor
    for patternonly in (True, False):
        metrics = [taylor.calculate_metrics(mod, obs, patternonly=patternonly)
                   for mod, obs in zip(TempSurfaceMod, TempSurfaceObs)]
        fig, ax_polar, ax_legend = taylor.make_fig_axes(metrics=metrics, patternonly=patternonly)
        for stn, m in zip(stnslist, metrics):
            taylor.add_markerstick(ax_polar, m, label_info={'text': stn})
        legend_entry = {'label': 'Near-Surface Temperature', 'label_info': {'text': 'StnID'}}
        taylor.add_labeledmarker_legend(ax_legend, legend_entry, loc='center')
        fig.savefig('example6_patternonly.png' if patternonly else 'example6.png', dpi=300)

## Purposes of functions

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
  
The text_polar() helper function (see full list of helpers below) is public on purpose, as matplotlib's ax.text() with rotation is unreliable on polar axes. The package draws its rotated text (markerstick labels and the text on and around the polar axes) with text_polar() instead. It is public so that users can add their own rotated annotations to the diagram the same way.

calculate_metrics(model, observation, period=None, ...) takes arrays of shape (n,) for scalars or (n, 2) for vectors (e.g. east and north components), of equal shape and with no NaN or inf values, and returns a dict of metrics; see its docstring for the full list and definitions. Circular scalar variables such as a phase angle can be handled (for any input range e.g. -180 to 180 deg, 0 to 360 deg, 1 to 12 months) by passing the period (period=360 for degrees, 2*np.pi for radians, 24 for hour of day), observations are taken as angles from their circular mean and the model-observation differences use within-cycle distances.

make_fig_axes(**overrides) accepts settings that replace its defaults: figsize_inch, margins_t_b_l_r_inch, vgap_legend_rel_polar_inch, fontsize, and the dicts grid_info and vdcc_info (e.g. line widths of gridlines and CC contours). See its docstring for the valid keys; an unknown name raises an error that lists them. With return_artists=True it also returns a dict of the background artists, for fine-tuning (see Notes about the code).

## Notes about the code

The user must condition the incoming data to exclude NaN and inf values.

Background elements of the plot are all in black/white or grayscale, including the main polar axes with shading to indicate MSD/OSD values and white contours to indicate CC values, and the MSD/OSD grayscale bar key at the bottom below the polar axes. This is intentional, such that the markersticks that indicate model-observation comparisons can be colored so they are more easily distinguished against the background. Using black or gray markersticks is possible but not recommended.

The plot layout, spacing of graphical elements, and default fontsizes are tailored to work well for the default figure size of a full 8.5x11 inch page (US standard), with 1" margins at top and bottom and 0.5" left and right margins. In order to allow for multiple model-observation pairs to be presented on the diagram such that all remain legible, the plot elements nearly always will require most of the full page, to allow the polar axes to be sufficiently large. Changing the figure size, for example making it smaller by ~50% such that more than one multimetric diagram could be arranged on a single page, may yield legible results for simple cases with few model-observation pairs; however if attempting this be aware that it will mean some background elements are less legible and/or overlap, and will require some potentially extensive fine-tuning. The first tools for that are make_fig_axes()'s settings (fontsize, margins_t_b_l_r_inch, grid_info, vdcc_info). make_fig_axes(return_artists=True) also returns a dict of the background artists, grouped by kind (e.g. 'circular_gridlines', 'shading', 'cc_contours', 'markerstick_keys', 'grayscale_bar_ax'; see its docstring for all keys), so they can be restyled or removed. add_markerstick(), add_labeledmarker_legend() and text_polar() return the artists they draw (add_markerstick() as a named tuple, MarkerStick(marker, stick, labels), which unpacks like a 3-tuple), and adjust_markerstick_labels() returns the labels it moved, each with its leader line.

The layout does not depend on dpi. plt.savefig() without dpi uses matplotlib's default resolution (100 dpi: an 850x1100-pixel PNG at the default page size); pass dpi=300 or more for clear images, as the usage examples do.

Label text (the 'text' in label_info, and legend entry labels) can be any object, such as a number; it is converted with str(), and None gives no label. Crowding of labeled markersticks can be mitigated by adjusting the label positions using the 'position' key in label_info, and/or by adjusting the markerstick styles to make them more distinguishable, for example by using different colors and transparencies, marker shapes, and/or stick linestyles. Crowding can also be reduced by calling adjust_markerstick_labels(). This function is meant to be called after all add_markerstick() calls are complete (calling it again starts over from the preset positions). It leaves labels that overlap nothing at their preset positions, moves each overlapping label to a nearby clear spot (preserving its stick-aligned rotation), and draws a thin leader line back to the marker wherever a label is moved. Its placement approach was originally built on textalloc by Christoffer Kjellson (https://github.com/ckjellson/textalloc).

Matplotlib's own built-in legend functions (e.g. ax.legend(), fig.legend()) can be used instead of add_labeledmarker_legend(), and are simpler and more direct whenever the legend does not need to include text labels drawn on the markersticks themselves (e.g. all keys in the legend are unadorned markers). Use add_labeledmarker_legend() specifically when the markersticks on the polar plot must carry their own text labels (via add_markerstick()'s label_info) and the legend needs to explain what those labels mean, annotating a representative marker's label placement/meaning within each legend entry (which is beyond what matplotlib's built-in legend functions can do).

## Package layout

    src/multimetricdiagram/
        __init__.py      public API: the five main functions, plus text_polar()
        metrics.py       calculate_metrics()
        axes.py          make_fig_axes()
        markerstick.py   add_markerstick(), _resolve_markerstick_style()
        labels.py        adjust_markerstick_labels() and geometry helpers
        legend.py        add_labeledmarker_legend() and label-position helpers
        _text.py         text_polar() (helper; also public)
        _defaults.py     shared default style values
        alt/             other material, including Taylor diagrams

Outside src/, usage_examples/ holds the runnable usage examples and docs/ the README's images; neither is installed with the package.

## Helper functions

    Grouped by module; see Package layout.
        _text.py
            text_polar()
                Rotated text at a polar position, drawn as a path; public (see the note under Purposes of functions); used by add_markerstick, make_fig_axes
            _PolarTextPatch (class)
                The patch text_polar() returns; recomputes its transform at each draw
        metrics.py
            _check_spread()
                Check that SDQ can be computed (no underflow or overflow); used by calculate_metrics
        axes.py
            _check_axes_sizes(), _vd(), _cc(), _fmt_tick_k()
                Axes-size check, |VD| and CC at diagram positions, grayscale bar tick labels; used by make_fig_axes
            _draw_polar_grid_and_reference_lines(), _draw_sector_text_and_cc_caption(), _draw_vd_shading_and_cc_contours(), _draw_markerstick_keys(), _draw_msdosd_grayscale_bar()
                Draw the diagram's background elements and MSD/OSD grayscale bar; used by make_fig_axes
        markerstick.py
            MarkerStick (class)
                Named tuple (marker, stick, labels) that add_markerstick returns
            _draw_markerstick()
                Draw a marker, its stick and its labels at given positions; used by add_markerstick
            _resolve_markerstick_style()
                Handle defaults and overrides for markerstick styling; used by add_markerstick, add_labeledmarker_legend
            _default_label_color(), _single_color()
                Default marker-label color (marker edge color; face color if the edge is invisible; else black); used by add_markerstick, add_labeledmarker_legend
            _label_dicts()
                Check label_info and return it as a list of label dicts; used by add_markerstick, add_labeledmarker_legend
            _marker_centroid()
                Center of a marker shape (area centroid if closed, else center of its extents); used by add_markerstick, make_fig_axes
            _marker_extents_pt(), _edge_shift_pt()
                Marker extents in points, and label shifts measured as gaps from the marker edge; used by add_markerstick, add_labeledmarker_legend
        labels.py (all used by adjust_markerstick_labels)
            _obstacles(), _Scene (class), _scene(), _sector_test()
                The diagram's fixed elements that labels avoid, and the plotted sector labels must stay inside
            _label_quad(), _hits(), _overlaps(), _fewest_overlaps(), _clear_labels()
                A label's rotated box, overlap tests, and the labels that already overlap nothing (left in place)
            _candidate_shifts(), _ring_shifts(), _best_spot()
                Candidate spots for a moved label (rings around its marker, then a grid) and the choice among them
            _has_leader(), _leader(), _leader_crosses(), _along_own_stick(), _exit_fraction()
                Leader lines: whether one is needed, where it starts, what it crosses, and keeping it clear of its own stick
            _grown(), _convex_hull(), _convex_overlap()
                Geometry for the overlap tests
        legend.py (all used by add_labeledmarker_legend)
            _resolve_legend_label_props()
                Apply label position presets
            _LabeledMarkerHandler (class), _text_renderer(), _text_extent_pt(), _line_marker_extents_pt(), _marker_half_extent_pt(), _kwarg_or_rc(), _blank_title(), _layout_for_labels()
                Legend handler that draws each row's marker labels, text measurement without drawing, and legend spacing that fits the labels

## Acknowledgments

Label placement in adjust_markerstick_labels() was originally built on textalloc by Christoffer Kjellson (ckjellson), https://github.com/ckjellson/textalloc.

## License

BSD 3-Clause License. See `LICENSE`.

## Citation

If you use this package, please cite:

Codiga, Daniel L., "Improved Evaluation and Communication of Model–Observation Comparisons: The Diagnostic Index of Agreement with Symmetry, and the Multimetric Diagram as an Alternative to the Taylor Diagram", submitted manuscript, 2026.

The same citation is in `CITATION.cff`, from which GitHub's "Cite this repository" button gives it in APA and BibTeX formats.

D. Codiga, with Claude agents, 11/2026
