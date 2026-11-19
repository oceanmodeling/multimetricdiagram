# Ex 2: Same as Ex 1, but: one label per markerstick; labeledmarker legend.

"""Ex 2 of the README usage examples, with synthetic data (example_data.py)."""

from example_data import TempSurfaceMod, TempSurfaceObs, stnslist

import multimetricdiagram as mmd

fig, ax_polar, ax_legend = mmd.make_fig_axes()
for stn, mod, obs in zip(stnslist, TempSurfaceMod, TempSurfaceObs):
    mmd.add_markerstick(ax_polar, mmd.calculate_metrics(mod, obs), label_info={'text': stn})
legend_entry = {'label': 'Near-Surface Temperature', 'label_info': {'text': 'StnID'}}
mmd.add_labeledmarker_legend(ax_legend, legend_entry, loc='center')
fig.savefig('example2.png', dpi=300)
