# Ex 4: Groups of Ex 2 and Ex 3 superposed on one diagram; labeledmarker legend with two items
# (one add_labeledmarker_legend() call, 2-item list passed); legend at lower left on polar axes.

"""Ex 4 of the README usage examples, with synthetic data (example_data.py)."""

from example_data import (
    TempBottomMod,
    TempBottomObs,
    TempSurfaceMod,
    TempSurfaceObs,
    depth,
    stnslist,
)

import multimetricdiagram as mmd

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
