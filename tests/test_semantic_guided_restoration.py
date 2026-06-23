import inspect
import json
from pathlib import Path

import torch

from cea_plus.degradation import build_degradation_plan
from cea_plus.restoration import bicubic_restore, uniform_sharp_restore
from cea_plus.synthesis import make_synthetic_agri_sample


def test_segmenter_guided_frequency_restore_api_excludes_oracle_inputs():
    from cea_plus.semantic_guided_restoration import segmenter_guided_frequency_restore

    signature = inspect.signature(segmenter_guided_frequency_restore)

    assert "mask" not in signature.parameters
    assert "gt_mask" not in signature.parameters
    assert "degradation_name" not in signature.parameters


def test_segmenter_guided_frequency_can_fall_back_on_low_confidence():
    from cea_plus.semantic_guided_restoration import segmenter_guided_frequency_restore

    class UncertainGuide:
        def predict_probability(self, image):
            return torch.full(image.shape[:2], 0.5).numpy().astype("float32")

    sample = make_synthetic_agri_sample(seed=4690, size=48)
    low_res = sample.image[::2, ::2]
    restored = segmenter_guided_frequency_restore(
        low_res,
        sample.image.shape[:2],
        guide_segmenter=UncertainGuide(),
        uncertain_strength=0.0,
    )
    bicubic = bicubic_restore(low_res, sample.image.shape[:2])

    assert restored.shape == sample.image.shape
    assert float(abs(restored - bicubic).mean()) < 1e-6


def test_segmenter_guided_frequency_can_blindly_dampen_scale2_bright_fog():
    from cea_plus.semantic_guided_restoration import segmenter_guided_frequency_restore

    class ConfidentGuide:
        def predict_probability(self, image):
            return torch.ones(image.shape[:2]).numpy().astype("float32")

    low_res = torch.full((24, 24, 3), 0.68).numpy().astype("float32")
    low_res[::2, ::2, 0] = 0.62
    bicubic = bicubic_restore(low_res, (48, 48))
    restored = segmenter_guided_frequency_restore(
        low_res,
        (48, 48),
        guide_segmenter=ConfidentGuide(),
        base_strength=0.72,
        inner_boost=0.2,
        outer_boost=-0.05,
        scale2_fog_fallback_strength=0.0,
        scale2_fog_mean_threshold=0.6,
        scale2_fog_saturation_threshold=0.12,
    )

    assert restored.shape == bicubic.shape
    assert float(abs(restored - bicubic).mean()) < 1e-6


def test_segmenter_guided_frequency_can_blindly_dampen_any_scale_bright_fog():
    from cea_plus.semantic_guided_restoration import segmenter_guided_frequency_restore

    class ConfidentGuide:
        def predict_probability(self, image):
            return torch.ones(image.shape[:2]).numpy().astype("float32")

    low_res = torch.full((12, 12, 3), 0.68).numpy().astype("float32")
    low_res[::2, ::2, 0] = 0.62
    bicubic = bicubic_restore(low_res, (48, 48))
    restored = segmenter_guided_frequency_restore(
        low_res,
        (48, 48),
        guide_segmenter=ConfidentGuide(),
        base_strength=0.72,
        inner_boost=0.2,
        outer_boost=-0.05,
        bright_fog_fallback_strength=0.0,
        bright_fog_mean_threshold=0.6,
        bright_fog_saturation_threshold=0.12,
    )

    assert restored.shape == bicubic.shape
    assert float(abs(restored - bicubic).mean()) < 1e-6


def test_segmenter_guided_frequency_can_calibrate_low_absolute_probabilities():
    from cea_plus.semantic_guided_restoration import segmenter_guided_frequency_restore

    class LowAbsoluteGuide:
        def predict_probability(self, image):
            prob = torch.full(image.shape[:2], 0.10)
            prob[12:36, 12:36] = 0.20
            return prob.numpy().astype("float32")

    sample = make_synthetic_agri_sample(seed=4710, size=48)
    low_res = sample.image[::2, ::2]
    raw = segmenter_guided_frequency_restore(
        low_res,
        sample.image.shape[:2],
        guide_segmenter=LowAbsoluteGuide(),
        base_strength=0.0,
        inner_boost=1.0,
        outer_boost=0.0,
        mask_blur_sigma=0.0,
        probability_calibration="none",
    )
    calibrated = segmenter_guided_frequency_restore(
        low_res,
        sample.image.shape[:2],
        guide_segmenter=LowAbsoluteGuide(),
        base_strength=0.0,
        inner_boost=1.0,
        outer_boost=0.0,
        mask_blur_sigma=0.0,
        probability_calibration="image_minmax",
    )

    assert calibrated.shape == raw.shape
    assert float(abs(calibrated - raw).mean()) > 1e-4


def test_segmenter_guided_frequency_can_mix_back_to_uniform_sharp():
    from cea_plus.semantic_guided_restoration import segmenter_guided_frequency_restore

    class ConfidentGuide:
        def predict_probability(self, image):
            return torch.ones(image.shape[:2]).numpy().astype("float32")

    sample = make_synthetic_agri_sample(seed=4720, size=48)
    low_res = sample.image[::2, ::2]
    restored = segmenter_guided_frequency_restore(
        low_res,
        sample.image.shape[:2],
        guide_segmenter=ConfidentGuide(),
        base_strength=0.0,
        inner_boost=0.0,
        outer_boost=0.0,
        uniform_mix_weight=1.0,
    )
    uniform = uniform_sharp_restore(low_res, sample.image.shape[:2])

    assert restored.shape == uniform.shape
    assert float(abs(restored - uniform).mean()) < 1e-6


def test_downstream_can_evaluate_segmenter_guided_frequency_no_leak(tmp_path: Path):
    from cea_plus.semantic_inr_downstream import run_semantic_inr_frozen_downstream

    train_samples = [make_synthetic_agri_sample(seed=4700 + idx, size=32).with_name(f"train_{idx}") for idx in range(3)]
    test_samples = [make_synthetic_agri_sample(seed=4800, size=32).with_name("test_0")]

    summary = run_semantic_inr_frozen_downstream(
        output_dir=tmp_path / "guided_frequency",
        train_samples=train_samples,
        test_samples=test_samples,
        degradation_plan=build_degradation_plan(scales=(2,), modes=("fog",)),
        segmenter_architecture="tiny",
        segmenter_steps=2,
        segmenter_crop_size=32,
        eval_size=32,
        inr_steps=2,
        hidden_channels=8,
        guide_segmenter_architecture="tiny",
        guide_segmenter_steps=2,
        guide_inner_boost=0.2,
        guide_outer_boost=-0.05,
        methods=("bicubic", "uniform_sharp", "semantic_guided_frequency"),
        seed=41,
        device=torch.device("cpu"),
    )

    metrics = summary.metrics_csv.read_text(encoding="utf-8")
    summary_json = json.loads(summary.summary_json.read_text(encoding="utf-8"))
    report = summary.report.read_text(encoding="utf-8")

    assert "semantic_guided_frequency" in metrics
    assert "Semantic Guided Frequency vs Bicubic" in report
    assert "Semantic Guided Frequency vs Uniform Sharp" in report
    assert summary_json["guide_segmenter_architecture"] == "tiny"
    assert summary_json["guide_segmenter_train_source"] == "clean_train_only"
    assert "semantic_guided_frequency_vs_bicubic_deep_frozen_miou_ttest_p" in summary_json
    assert "semantic_guided_frequency_vs_uniform_sharp_deep_frozen_miou_delta" in summary_json
    assert summary.guide_segmenter_model_path is not None
    assert summary.guide_segmenter_model_path.exists()
