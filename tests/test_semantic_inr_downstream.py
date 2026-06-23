import json
from pathlib import Path

import pytest
import torch

from cea_plus.degradation import build_degradation_plan
from cea_plus.synthesis import make_synthetic_agri_sample


def test_semantic_inr_frozen_downstream_rejects_green_task_loss(tmp_path: Path):
    from cea_plus.semantic_inr_downstream import run_semantic_inr_frozen_downstream

    train_samples = [make_synthetic_agri_sample(seed=4100, size=32).with_name("train_0")]
    test_samples = [make_synthetic_agri_sample(seed=4200, size=32).with_name("test_0")]

    with pytest.raises(ValueError, match="task_loss_weight=0"):
        run_semantic_inr_frozen_downstream(
            output_dir=tmp_path / "reject_task_loss",
            train_samples=train_samples,
            test_samples=test_samples,
            degradation_plan=build_degradation_plan(scales=(2,), modes=("fog",)),
            segmenter_architecture="tiny",
            segmenter_steps=1,
            segmenter_crop_size=32,
            eval_size=32,
            inr_steps=1,
            task_loss_weight=0.2,
            device=torch.device("cpu"),
        )


def test_semantic_inr_frozen_downstream_uses_clean_segmenter_and_no_leak_restoration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import cea_plus.deep_downstream as deep_downstream
    from cea_plus.semantic_inr_downstream import run_semantic_inr_frozen_downstream

    def fail_if_called(*args, **kwargs):
        raise AssertionError("旧 restore_all_methods 会接收 GT mask 和 degradation_name，可信 downstream 不能调用")

    monkeypatch.setattr(deep_downstream, "restore_all_methods", fail_if_called)

    train_samples = [make_synthetic_agri_sample(seed=4300 + idx, size=32).with_name(f"train_{idx}") for idx in range(2)]
    test_samples = [make_synthetic_agri_sample(seed=4400, size=32).with_name("test_0")]

    summary = run_semantic_inr_frozen_downstream(
        output_dir=tmp_path / "semantic_inr_downstream",
        train_samples=train_samples,
        test_samples=test_samples,
        degradation_plan=build_degradation_plan(scales=(2,), modes=("fog",)),
        segmenter_architecture="tiny",
        segmenter_steps=2,
        segmenter_crop_size=32,
        eval_size=32,
        inr_steps=2,
        hidden_channels=8,
        semantic_loss_weight=0.12,
        seed=31,
        device=torch.device("cpu"),
    )

    metrics = summary.metrics_csv.read_text(encoding="utf-8")
    report = summary.report.read_text(encoding="utf-8")
    summary_json = json.loads(summary.summary_json.read_text(encoding="utf-8"))

    assert summary.sample_count == 1
    assert "deep_frozen_miou" in metrics.splitlines()[0]
    assert "task_miou" not in metrics.splitlines()[0]
    assert "semantic_inr_no_leak" in metrics
    assert "Segmenter training: clean images only" in report
    assert summary_json["segmenter_train_source"] == "clean"
    assert summary_json["task_loss_weight"] == 0.0
    assert summary_json["semantic_loss_weight"] == 0.12


def test_semantic_inr_frozen_downstream_supports_train_only_learned_task_loss(tmp_path: Path):
    from cea_plus.semantic_inr_downstream import run_semantic_inr_frozen_downstream

    train_samples = [make_synthetic_agri_sample(seed=4500 + idx, size=32).with_name(f"train_{idx}") for idx in range(3)]
    test_samples = [make_synthetic_agri_sample(seed=4600, size=32).with_name("test_0")]

    summary = run_semantic_inr_frozen_downstream(
        output_dir=tmp_path / "semantic_inr_learned_task_downstream",
        train_samples=train_samples,
        test_samples=test_samples,
        degradation_plan=build_degradation_plan(scales=(2,), modes=("fog",)),
        segmenter_architecture="tiny",
        segmenter_steps=2,
        segmenter_crop_size=32,
        eval_size=32,
        inr_steps=2,
        hidden_channels=8,
        learned_task_loss_weight=0.25,
        learned_task_segmenter_architecture="tiny",
        learned_task_segmenter_steps=2,
        seed=37,
        device=torch.device("cpu"),
    )

    report = summary.report.read_text(encoding="utf-8")
    summary_json = json.loads(summary.summary_json.read_text(encoding="utf-8"))

    assert summary_json["task_loss_weight"] == 0.0
    assert summary_json["learned_task_loss_weight"] == 0.25
    assert summary_json["learned_task_segmenter_architecture"] == "tiny"
    assert summary_json["learned_task_segmenter_train_source"] == "clean_train_only"
    assert summary_json["learned_task_segmenter_role"] == "train_loss_only"
    assert "learned frozen segmenter task loss" in report
    assert summary.learned_task_segmenter_model_path is not None
    assert summary.learned_task_segmenter_model_path.exists()


def test_semantic_guided_frequency_can_select_params_on_validation_without_test_leak(tmp_path: Path):
    from cea_plus.semantic_inr_downstream import run_semantic_inr_frozen_downstream

    train_samples = [make_synthetic_agri_sample(seed=4900 + idx, size=32).with_name(f"train_{idx}") for idx in range(3)]
    validation_samples = [make_synthetic_agri_sample(seed=5000, size=32).with_name("validation_only")]
    test_samples = [make_synthetic_agri_sample(seed=5100, size=32).with_name("heldout_test")]
    candidates = (
        {"base_strength": 0.0, "inner_boost": 0.0, "outer_boost": 0.0, "uncertain_strength": 0.0},
        {"base_strength": 0.72, "inner_boost": 0.2, "outer_boost": -0.05, "uncertain_strength": None},
        {"base_strength": 0.36, "inner_boost": 0.2, "outer_boost": -0.05, "probability_calibration": "image_minmax"},
        {"base_strength": 0.0, "inner_boost": 0.45, "bright_fog_fallback_strength": 0.0},
        {"base_strength": 0.36, "inner_boost": 0.2, "uniform_mix_weight": 0.5},
    )

    summary = run_semantic_inr_frozen_downstream(
        output_dir=tmp_path / "semantic_guided_validation_tuned",
        train_samples=train_samples,
        guide_validation_samples=validation_samples,
        guide_candidate_params=candidates,
        test_samples=test_samples,
        degradation_plan=build_degradation_plan(scales=(2,), modes=("fog",)),
        segmenter_architecture="tiny",
        segmenter_steps=2,
        segmenter_crop_size=32,
        eval_size=32,
        inr_steps=1,
        hidden_channels=8,
        guide_segmenter_architecture="tiny",
        guide_segmenter_steps=2,
        methods=("bicubic", "semantic_guided_frequency"),
        seed=43,
        device=torch.device("cpu"),
    )

    metrics = summary.metrics_csv.read_text(encoding="utf-8")
    report = summary.report.read_text(encoding="utf-8")
    summary_json = json.loads(summary.summary_json.read_text(encoding="utf-8"))

    assert "heldout_test" in metrics
    assert "validation_only" not in metrics
    assert summary_json["guide_selection_source"] == "clean_validation_only"
    assert summary_json["guide_selection_validation_pairs"] == 1
    assert summary_json["guide_selected_params"] in [dict(item) for item in candidates]
    assert any(
        item["effective_params"]["probability_calibration"] == "image_minmax"
        for item in summary_json["guide_candidate_scores"]
    )
    assert any(
        item["effective_params"]["bright_fog_fallback_strength"] == 0.0
        for item in summary_json["guide_candidate_scores"]
    )
    assert any(
        item["effective_params"]["uniform_mix_weight"] == 0.5
        for item in summary_json["guide_candidate_scores"]
    )
    assert "Guide parameter selection" in report
    assert "clean validation only" in report
