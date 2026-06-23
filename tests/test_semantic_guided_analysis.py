import csv
import json
from pathlib import Path

import torch

from cea_plus.degradation import build_degradation_plan
from cea_plus.synthesis import make_synthetic_agri_sample


def test_guide_quality_analysis_reports_pseudo_mask_quality_without_training_leak(tmp_path: Path):
    from cea_plus.semantic_guided_analysis import run_guide_quality_analysis

    train_samples = [make_synthetic_agri_sample(seed=5200 + idx, size=32).with_name(f"train_{idx}") for idx in range(3)]
    test_samples = [make_synthetic_agri_sample(seed=5300, size=32).with_name("heldout_0")]

    summary = run_guide_quality_analysis(
        output_dir=tmp_path / "guide_quality",
        train_samples=train_samples,
        test_samples=test_samples,
        degradation_plan=build_degradation_plan(scales=(2,), modes=("fog", "mixed")),
        guide_architecture="tiny",
        guide_steps=2,
        guide_crop_size=32,
        eval_size=32,
        seed=51,
        device=torch.device("cpu"),
    )

    rows = list(csv.DictReader(summary.metrics_csv.open("r", encoding="utf-8")))
    summary_json = json.loads(summary.summary_json.read_text(encoding="utf-8"))
    report = summary.report.read_text(encoding="utf-8")

    assert len(rows) == 2
    assert rows[0]["sample"] == "heldout_0"
    assert "pseudo_mask_iou" in rows[0]
    assert "mean_confidence" in rows[0]
    assert "pseudo_positive_rate" in rows[0]
    assert "foreground_probability_mean" in rows[0]
    assert summary_json["guide_train_source"] == "clean_train_only"
    assert summary_json["sample_pairs"] == 2
    assert 0.0 <= summary_json["mean_pseudo_mask_iou"] <= 1.0
    assert 0.0 <= summary_json["mean_confidence"] <= 1.0
    assert "GT masks are evaluation only" in report
