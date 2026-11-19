# Ex 6: Same as Ex 2, but on a Taylor diagram (multimetricdiagram.alt.taylor). The metrics are
# passed to make_fig_axes() so it can choose one- or two-quadrant variant, and the outer radius.
# Two example figures: patternonly=True, bias set to zero, only markers with no bias sticks,
# example6_patternonly.png; then default (patternonly=False) so bias sticks are shown,
# example6.png. In the latter, note the stations appear in different order, by distance of stick
# end to star, compared to Ex 2; the ordering is by NSE here and by DIAS in Ex 2.

"""Ex 6 of the README usage examples, with synthetic data (example_data.py)."""

from example_data import TempSurfaceMod, TempSurfaceObs, stnslist

import multimetricdiagram as mmd

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
