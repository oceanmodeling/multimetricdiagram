"""Default style values shared by add_markerstick() and add_labeledmarker_legend()."""

MARKERSTICK_COLOR = 'r'
MARKER_SHAPE = 'o'
MARKER_SIZE = 14  # points; scatter size s = MARKER_SIZE**2
MARKER_FACECOLOR = 'none'  # open (unfilled) marker
ALPHA = 0.6  # marker and stick
STICK_LINEWIDTH = 1.5
LEGEND_MARKER_EDGEWIDTH = 1.5
LABEL_FONTSIZE = 10

# label 'position' presets; offsets are gaps from the marker's edge, in multiples of fontsize;
# add_markerstick() mirrors 'beside' to ha='left', cw_offset=+0.1 when MB/SDQ < 0
LABEL_POSITION_PRESETS = {
    'outside': {'va': 'bottom', 'ha': 'center', 'out_offset': 0.1, 'cw_offset': 0},
    'inside': {'va': 'top', 'ha': 'center', 'out_offset': -0.1, 'cw_offset': 0},
    'beside': {'va': 'center', 'ha': 'right', 'out_offset': 0, 'cw_offset': -0.1},
}
