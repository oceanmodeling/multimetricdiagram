# Ex 5: Vector inputs (near-surface current); six model-observation pairs whose markersticks
# crowd together, labels repositioned by adjust_markerstick_labels() to avoid overlap.

"""Ex 5 of the README usage examples, with synthetic data (example_data.py)."""

from example_data import CurrentSurfaceMod, CurrentSurfaceObs, stnslist

import multimetricdiagram as mmd

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
