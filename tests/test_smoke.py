import matplotlib
import numpy as np

matplotlib.use("Agg")

import multimetricdiagram as mmd


def test_public_api():
    for name in mmd.__all__:
        assert callable(getattr(mmd, name))


def test_full_diagram_renders(tmp_path):
    rng = np.random.default_rng(1)
    fig, ax_polar, ax_legend = mmd.make_fig_axes()
    results = []
    for i in range(4):
        obs = rng.normal(10.0, 2.0, 100)
        mod = obs + rng.normal(0.3 * i, 1.0 + 0.2 * i, 100)
        metrics = mmd.calculate_metrics(mod, obs)
        results.append(mmd.add_markerstick(ax_polar, metrics, label_info={"text": f"S{i}"}))
    mmd.adjust_markerstick_labels(ax_polar, results)
    mmd.add_labeledmarker_legend(
        ax_legend, [{"label": "Test group", "label_info": {"text": "StnID"}}]
    )
    fig.savefig(tmp_path / "diagram.png", dpi=100)
    assert (tmp_path / "diagram.png").stat().st_size > 0
