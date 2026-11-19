# Ex 1: One group of model-observation pairs; default marker style, no marker labels; legend in
# the legend axes above the diagram; legend by Matplotlib's built-in ax.legend(), simpler and
# more direct than add_labeledmarker_legend() because no marker labels are needed.

"""Ex 1 of the README usage examples, with synthetic data (example_data.py)."""

from example_data import TempSurfaceMod, TempSurfaceObs

import multimetricdiagram as mmd

fig, ax_polar, ax_legend = mmd.make_fig_axes()
for mod, obs in zip(TempSurfaceMod, TempSurfaceObs):
    ms = mmd.add_markerstick(ax_polar, mmd.calculate_metrics(mod, obs))
ax_legend.legend([ms.marker], ['Near-Surface Temperature'], loc='center')
fig.savefig('example1.png', dpi=300)
