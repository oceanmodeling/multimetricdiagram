import io

import numpy as np
import pytest

import multimetricdiagram as mmd

NAMES = [
    "calculate_metrics",
    "make_fig_axes",
    "add_markerstick",
    "adjust_markerstick_labels",
    "add_labeledmarker_legend",
    "text_polar",
]


def test_reachable_from_top_level_import():
    assert mmd.alt.taylor.__all__ == NAMES
    for name in NAMES:
        assert callable(getattr(mmd.alt.taylor, name))
    assert not any(name.startswith(("alt", "taylor")) for name in mmd.__all__)


@pytest.mark.parametrize("name", NAMES[3:])
def test_reexports_are_the_top_level_functions(name):
    assert getattr(mmd.alt.taylor, name) is getattr(mmd, name)


@pytest.mark.parametrize("quadrants", [1, 2])
def test_adjust_labels_and_labeled_legend_on_taylor_figure(quadrants):
    taylor = mmd.alt.taylor
    rng = np.random.default_rng(3)
    obs = rng.normal(10.0, 2.0, 200)
    fig, ax, ax_legend = taylor.make_fig_axes(quadrants=quadrants)
    results = []
    cases = [(0.9, 0.3, 0.6), (1.1, -0.4, 0.8), (0.8, 0.2, 1.0), (1.0, 0.0, 0.3)]
    if quadrants == 2:
        cases.append((-0.5, 0.2, 1.5))
    for i, (gain, bias, noise) in enumerate(cases):
        model = gain * (obs - 10) + 10 + bias + rng.normal(0, noise, 200)
        m = taylor.calculate_metrics(model, obs)
        results.append(taylor.add_markerstick(ax, m, label_info={"text": f"Model {i}"}))
    fig.draw_without_rendering()
    before = [label.get_window_extent().extents for _m, _s, (label,) in results]
    taylor.adjust_markerstick_labels(ax, results)
    leaders = [ln for ln in ax.lines if ln.get_gid() == "multimetricdiagram-leader"]
    after = [label.get_window_extent().extents for _m, _s, (label,) in results]
    moved = sum(not np.allclose(a, b) for a, b in zip(after, before))
    assert len(leaders) <= moved < len(results)  # leader lines only for moved labels
    _legend, handles, labels = taylor.add_labeledmarker_legend(
        ax_legend,
        [{"label": "Models", "label_info": {"text": "Mi"}}, {"label": "Observations"}],
        loc="center",
    )
    assert len(handles) == 2 and len(labels[0]) == 1
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=50)
    assert buf.tell() > 0
