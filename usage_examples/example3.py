# Ex 3: Same as Ex 2, but: near-bottom temperature; two labels per markerstick; custom
# markerstick style; legend at lower left on polar axes.

"""Ex 3 of the README usage examples, with synthetic data (example_data.py)."""

from example_data import TempBottomMod, TempBottomObs, depth, stnslist

import multimetricdiagram as mmd

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
