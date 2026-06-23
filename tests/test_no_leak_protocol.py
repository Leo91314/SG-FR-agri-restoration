from pathlib import Path
import inspect

import numpy as np

from cea_plus.degradation import build_degradation_plan
from cea_plus.synthesis import make_synthetic_agri_sample


def test_no_leak_restore_api_excludes_oracle_inputs():
    from cea_plus.no_leak_restoration import restore_no_leak_methods

    signature = inspect.signature(restore_no_leak_methods)

    assert "mask" not in signature.parameters
    assert "gt_mask" not in signature.parameters
    assert "degradation_name" not in signature.parameters


def test_no_leak_semantic_restore_does_not_paint_pseudo_mask_green():
    from cea_plus.no_leak_restoration import semantic_frequency_no_leak_components

    low_res = np.full((16, 16, 3), 0.42, dtype=np.float32)
    pseudo_mask = np.zeros((64, 64), dtype=bool)
    pseudo_mask[16:48, 16:48] = True

    components = semantic_frequency_no_leak_components(
        low_res,
        output_shape=(64, 64),
        pseudo_mask=pseudo_mask,
    )

    inside = components.restored[pseudo_mask]
    outside = components.restored[~pseudo_mask]
    assert inside[:, 1].mean() <= inside[:, 0].mean() + 0.03
    assert abs(float(inside.mean()) - float(outside.mean())) < 0.02


def test_no_leak_dataset_experiment_writes_only_no_leak_methods(tmp_path: Path):
    from cea_plus.no_leak_experiment import run_no_leak_dataset_experiment

    samples = [make_synthetic_agri_sample(seed=41, size=64).with_name("sample_0")]
    summary = run_no_leak_dataset_experiment(
        output_dir=tmp_path / "no_leak",
        samples=samples,
        degradation_plan=build_degradation_plan(scales=(2,), modes=("fog",)),
        seed=19,
    )

    metrics = summary.metrics_csv.read_text(encoding="utf-8")

    assert "semantic_frequency_no_leak" in metrics
    assert "semantic_frequency," not in metrics
    assert "metrics.csv" == summary.metrics_csv.name
