from pathlib import Path

import numpy as np

from cea_plus.degradation import degrade_sample
from cea_plus.metrics import boundary_f_score, proxy_segmentation_miou, psnr, ssim_score
from cea_plus.pipeline import run_experiment
from cea_plus.restoration import bicubic_restore, semantic_frequency_components, semantic_frequency_restore
from cea_plus.statistics import paired_significance
from cea_plus.synthesis import make_synthetic_agri_sample


def test_synthetic_sample_is_deterministic():
    first = make_synthetic_agri_sample(seed=11, size=96)
    second = make_synthetic_agri_sample(seed=11, size=96)

    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.mask, second.mask)
    assert first.image.shape == (96, 96, 3)
    assert first.mask.shape == (96, 96)
    assert first.mask.dtype == bool


def test_degradation_downsamples_and_restoration_returns_hr_size():
    sample = make_synthetic_agri_sample(seed=3, size=96)
    degraded = degrade_sample(sample, scale=4, seed=5)

    assert degraded.low_res.shape == (24, 24, 3)
    assert degraded.gt.shape == sample.image.shape
    assert degraded.mask.shape == sample.mask.shape

    restored = bicubic_restore(degraded.low_res, output_shape=degraded.gt.shape[:2])
    assert restored.shape == degraded.gt.shape
    assert restored.dtype == np.float32
    assert restored.min() >= 0.0
    assert restored.max() <= 1.0


def test_metrics_reward_closer_images_and_boundaries():
    sample = make_synthetic_agri_sample(seed=8, size=96)
    degraded = degrade_sample(sample, scale=4, seed=9)
    bicubic = bicubic_restore(degraded.low_res, degraded.gt.shape[:2])
    semantic = semantic_frequency_restore(degraded.low_res, degraded.mask, degraded.gt.shape[:2])

    assert psnr(degraded.gt, semantic) > psnr(degraded.gt, bicubic)
    assert ssim_score(degraded.gt, semantic) > ssim_score(degraded.gt, bicubic)
    assert boundary_f_score(degraded.gt, semantic) >= boundary_f_score(degraded.gt, bicubic) - 0.02
    assert proxy_segmentation_miou(degraded.mask, semantic) >= proxy_segmentation_miou(degraded.mask, bicubic) - 0.05


def test_semantic_frequency_components_expose_modulation():
    sample = make_synthetic_agri_sample(seed=15, size=96)
    degraded = degrade_sample(sample, scale=4, seed=16)

    components = semantic_frequency_components(degraded.low_res, degraded.mask, degraded.gt.shape[:2])

    assert components.restored.shape == degraded.gt.shape
    assert components.structure.shape == degraded.gt.shape
    assert components.texture.shape == degraded.gt.shape
    assert components.alpha.shape == degraded.mask.shape
    assert components.alpha.min() >= 0.0
    assert components.alpha.max() <= 1.0
    assert components.alpha[degraded.mask].mean() > components.alpha[~degraded.mask].mean() + 0.2


def test_paired_significance_detects_consistent_gain():
    baseline = np.array([20.0, 21.0, 19.5, 22.0, 20.5, 21.5])
    proposed = baseline + np.array([1.0, 1.2, 0.9, 1.1, 1.0, 1.3])

    result = paired_significance(baseline, proposed, seed=13)

    assert result.mean_delta > 0.9
    assert result.ttest_p < 0.01
    assert result.bootstrap_ci_low > 0.0


def test_run_experiment_writes_report(tmp_path: Path):
    output_dir = tmp_path / "smoke"
    summary = run_experiment(output_dir=output_dir, samples=8, seed=17, image_size=96)

    assert (output_dir / "metrics.csv").exists()
    assert (output_dir / "significance_report.md").exists()
    assert (output_dir / "cases" / "case_000.png").exists()
    assert summary.sample_count == 8
    assert summary.best_method == "semantic_frequency"
    assert summary.significant_metrics
    assert "task_miou" in summary.significant_metrics
