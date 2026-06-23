from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

from cea_plus.dataset import load_agri_dataset, load_loveda_dataset, load_weedsgalore_dataset
from cea_plus.degradation import DegradationConfig, build_degradation_plan, degrade_sample
from cea_plus.pipeline import run_dataset_experiment
from cea_plus.restoration import restore_all_methods


def _write_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255)).save(path)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.uint8(mask.astype(np.uint8) * 255)).save(path)


def test_load_agri_dataset_pairs_images_and_masks(tmp_path: Path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    image = np.zeros((32, 48, 3), dtype=np.float32)
    image[..., 1] = 0.6
    mask = np.zeros((32, 48), dtype=bool)
    mask[:, 10:20] = True
    _write_rgb(image_dir / "plot_001.png", image)
    _write_mask(mask_dir / "plot_001.png", mask)

    samples = load_agri_dataset(image_dir=image_dir, mask_dir=mask_dir, limit=4)

    assert len(samples) == 1
    assert samples[0].image.shape == (32, 48, 3)
    assert samples[0].mask.shape == (32, 48)
    assert samples[0].mask.dtype == bool
    assert samples[0].name == "plot_001"


def test_load_weedsgalore_dataset_reads_rgb_bands_and_semantics():
    dataset_root = Path("data/external/weedsgalore-dataset")
    if not dataset_root.exists():
        return

    samples = load_weedsgalore_dataset(dataset_root=dataset_root, split="test", limit=2)

    assert len(samples) == 2
    assert samples[0].image.shape == (600, 600, 3)
    assert samples[0].image.dtype == np.float32
    assert samples[0].image.min() >= 0.0
    assert samples[0].image.max() <= 1.0
    assert samples[0].mask.shape == (600, 600)
    assert samples[0].mask.dtype == bool
    assert samples[0].mask.any()
    assert samples[0].name.startswith("2023-")


def test_load_loveda_dataset_reads_agriculture_masks(tmp_path: Path):
    root = tmp_path / "loveda"
    rural_images = root / "Val" / "Rural" / "images_png"
    rural_masks = root / "Val" / "Rural" / "masks_png"
    urban_images = root / "Val" / "Urban" / "images_png"
    urban_masks = root / "Val" / "Urban" / "masks_png"
    for path in (rural_images, rural_masks, urban_images, urban_masks):
        path.mkdir(parents=True)

    image = np.zeros((48, 64, 3), dtype=np.float32)
    image[..., 1] = 0.5
    labels = np.zeros((48, 64), dtype=np.uint8)
    labels[8:28, 12:30] = 7
    _write_rgb(rural_images / "rural_001.png", image)
    Image.fromarray(labels).save(rural_masks / "rural_001.png")

    empty_labels = np.zeros((48, 64), dtype=np.uint8)
    _write_rgb(rural_images / "rural_002.png", image)
    Image.fromarray(empty_labels).save(rural_masks / "rural_002.png")

    urban_labels = np.zeros((48, 64), dtype=np.uint8)
    urban_labels[4:18, 6:26] = 7
    _write_rgb(urban_images / "urban_001.png", image)
    Image.fromarray(urban_labels).save(urban_masks / "urban_001.png")

    samples = load_loveda_dataset(root, split="val", domain="Rural", target_labels=(7,), limit=4)

    assert len(samples) == 1
    assert samples[0].image.shape == (48, 64, 3)
    assert samples[0].mask.dtype == bool
    assert samples[0].mask.sum() == 20 * 18
    assert samples[0].name == "Val_Rural_rural_001"


def test_load_loveda_dataset_supports_seeded_random_crops(tmp_path: Path):
    root = tmp_path / "loveda"
    image_dir = root / "Val" / "Rural" / "images_png"
    mask_dir = root / "Val" / "Rural" / "masks_png"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    image = np.zeros((96, 96, 3), dtype=np.float32)
    image[..., 0] = np.linspace(0.0, 1.0, 96, dtype=np.float32)[None, :]
    labels = np.zeros((96, 96), dtype=np.uint8)
    labels[70:92, 70:92] = 7
    _write_rgb(image_dir / "tile_001.png", image)
    Image.fromarray(labels).save(mask_dir / "tile_001.png")

    centered = load_loveda_dataset(root, split="Val", domain="Rural", crop_size=48, crop_strategy="mask_center", limit=1)
    random_a = load_loveda_dataset(root, split="Val", domain="Rural", crop_size=48, crop_strategy="random", crop_seed=3, require_mask=False, limit=1)
    random_b = load_loveda_dataset(root, split="Val", domain="Rural", crop_size=48, crop_strategy="random", crop_seed=3, require_mask=False, limit=1)

    assert centered[0].mask.any()
    assert random_a[0].image.shape == (48, 48, 3)
    assert np.array_equal(random_a[0].image, random_b[0].image)
    assert not np.array_equal(centered[0].image, random_a[0].image)


def test_configurable_degradation_plan_covers_scale_and_modes():
    plan = build_degradation_plan(scales=(2, 4), modes=("rain", "fog", "mixed"))

    assert [item.name for item in plan] == ["x2_rain", "x2_fog", "x2_mixed", "x4_rain", "x4_fog", "x4_mixed"]
    assert plan[0].scale == 2
    assert plan[-1].scale == 4
    assert plan[-1].rain_density > 0
    assert plan[-1].fog_strength > 0


def test_degrade_sample_accepts_explicit_config():
    from cea_plus.synthesis import make_synthetic_agri_sample

    sample = make_synthetic_agri_sample(seed=31, size=96)
    config = DegradationConfig(name="x2_noise", scale=2, noise_sigma=0.02, rain_density=0, fog_strength=0.0)
    degraded = degrade_sample(sample, config=config, seed=32)

    assert degraded.low_res.shape == (48, 48, 3)
    assert degraded.degradation_name == "x2_noise"
    assert degraded.scale == 2


def test_restore_all_methods_includes_ablation_variants():
    from cea_plus.synthesis import make_synthetic_agri_sample

    sample = make_synthetic_agri_sample(seed=41, size=96)
    degraded = degrade_sample(sample, config=DegradationConfig(name="x4_mixed", scale=4), seed=42)

    outputs = restore_all_methods(degraded.low_res, degraded.mask, degraded.gt.shape[:2])

    assert set(outputs) == {
        "bicubic",
        "uniform_sharp",
        "semantic_frequency",
        "semantic_edge_aware",
        "semantic_boundary_guard",
        "semantic_no_mod",
        "semantic_fixed_alpha",
        "structure_only",
    }
    assert all(image.shape == degraded.gt.shape for image in outputs.values())


def test_run_dataset_experiment_writes_degradation_and_ablation_rows(tmp_path: Path):
    from cea_plus.synthesis import make_synthetic_agri_sample

    samples = [make_synthetic_agri_sample(seed=50 + idx, size=96).with_name(f"synthetic_{idx}") for idx in range(3)]
    plan = build_degradation_plan(scales=(4,), modes=("rain", "fog"))
    summary = run_dataset_experiment(output_dir=tmp_path / "real_stage", samples=samples, degradation_plan=plan, seed=3)
    metrics_text = (tmp_path / "real_stage" / "metrics.csv").read_text(encoding="utf-8")
    report_text = (tmp_path / "real_stage" / "significance_report.md").read_text(encoding="utf-8")

    assert summary.sample_count == 6
    assert "degradation" in metrics_text.splitlines()[0]
    assert "semantic_no_mod" in metrics_text
    assert "x4_rain" in metrics_text
    assert "消融" in report_text
    assert "分退化统计" in report_text
    assert "x4_fog" in report_text


def test_run_dataset_experiment_flushes_completed_rows_on_failure(tmp_path: Path, monkeypatch):
    import cea_plus.pipeline as pipeline
    from cea_plus.synthesis import make_synthetic_agri_sample

    original_restore_all = pipeline.restore_all_methods
    calls = {"count": 0}

    def fail_after_first(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] > 1:
            raise RuntimeError("simulated matrix interruption")
        return original_restore_all(*args, **kwargs)

    monkeypatch.setattr(pipeline, "restore_all_methods", fail_after_first)
    samples = [make_synthetic_agri_sample(seed=140 + idx, size=64).with_name(f"heldout_{idx}") for idx in range(2)]
    plan = build_degradation_plan(scales=(2,), modes=("fog",))

    with pytest.raises(RuntimeError, match="simulated matrix interruption"):
        run_dataset_experiment(output_dir=tmp_path / "matrix_resume", samples=samples, degradation_plan=plan, seed=9, resume=True)

    metrics = (tmp_path / "matrix_resume" / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert len(metrics) == 9
    assert all(line.startswith("heldout_0,") for line in metrics[1:])
    assert any("semantic_frequency" in line for line in metrics)


def test_run_dataset_experiment_resumes_existing_metrics(tmp_path: Path, monkeypatch):
    import cea_plus.pipeline as pipeline
    from cea_plus.synthesis import make_synthetic_agri_sample

    original_restore_all = pipeline.restore_all_methods
    calls = {"count": 0}

    def counting_restore_all(*args, **kwargs):
        calls["count"] += 1
        return original_restore_all(*args, **kwargs)

    monkeypatch.setattr(pipeline, "restore_all_methods", counting_restore_all)
    samples = [make_synthetic_agri_sample(seed=150 + idx, size=64).with_name(f"heldout_{idx}") for idx in range(2)]
    plan = build_degradation_plan(scales=(2,), modes=("fog",))
    output_dir = tmp_path / "matrix_resume"
    output_dir.mkdir()
    existing_rows = [
        "sample,degradation,scale,method,psnr,ssim,boundary_f,task_miou",
        *[f"heldout_0,x2_fog,2,{method},1,1,1,1" for method in (
            "bicubic",
            "uniform_sharp",
            "semantic_frequency",
            "semantic_edge_aware",
            "semantic_boundary_guard",
            "semantic_no_mod",
            "semantic_fixed_alpha",
            "structure_only",
        )],
    ]
    (output_dir / "metrics.csv").write_text("\n".join(existing_rows) + "\n", encoding="utf-8")

    summary = run_dataset_experiment(output_dir=output_dir, samples=samples, degradation_plan=plan, seed=9, resume=True)

    metrics = (output_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert summary.sample_count == 2
    assert calls["count"] == 1
    assert sum(1 for line in metrics if line.startswith("heldout_0,")) == 8
    assert sum(1 for line in metrics if line.startswith("heldout_1,")) == 8


def test_semantic_frequency_is_degradation_aware_across_modes():
    from cea_plus.metrics import psnr
    from cea_plus.restoration import bicubic_restore, semantic_frequency_restore
    from cea_plus.synthesis import make_synthetic_agri_sample

    sample = make_synthetic_agri_sample(seed=71, size=128)
    blur = degrade_sample(
        sample,
        config=DegradationConfig(name="x2_blur", scale=2, rain_density=0, fog_strength=0.0, blur_sigma=0.95, noise_sigma=0.004),
        seed=72,
    )
    fog = degrade_sample(
        sample,
        config=DegradationConfig(name="x4_fog", scale=4, rain_density=0, fog_strength=0.30, noise_sigma=0.008),
        seed=73,
    )

    blur_bicubic = bicubic_restore(blur.low_res, blur.gt.shape[:2])
    blur_semantic = semantic_frequency_restore(blur.low_res, blur.mask, blur.gt.shape[:2], degradation_name=blur.degradation_name)
    fog_bicubic = bicubic_restore(fog.low_res, fog.gt.shape[:2])
    fog_semantic = semantic_frequency_restore(fog.low_res, fog.mask, fog.gt.shape[:2], degradation_name=fog.degradation_name)

    assert psnr(blur.gt, blur_semantic) >= psnr(blur.gt, blur_bicubic) - 0.5
    assert psnr(fog.gt, fog_semantic) > psnr(fog.gt, fog_bicubic) + 4.5


def test_weedsgalore_nonfog_visual_quality_regression():
    from cea_plus.metrics import boundary_f_score, psnr, ssim_score
    from cea_plus.restoration import bicubic_restore, semantic_frequency_restore

    dataset_root = Path("data/external/weedsgalore-dataset")
    if not dataset_root.exists():
        return

    samples = load_weedsgalore_dataset(dataset_root=dataset_root, split="test", limit=3)
    nonfog_deltas = []
    fog_deltas = []
    for sample in samples:
        for config in build_degradation_plan(scales=(2, 4), modes=("rain", "noise", "blur", "jpeg")):
            degraded = degrade_sample(sample, config=config, seed=23)
            bicubic = bicubic_restore(degraded.low_res, degraded.gt.shape[:2])
            semantic = semantic_frequency_restore(degraded.low_res, degraded.mask, degraded.gt.shape[:2], degradation_name=degraded.degradation_name)
            nonfog_deltas.append(
                (
                    psnr(degraded.gt, semantic) - psnr(degraded.gt, bicubic),
                    ssim_score(degraded.gt, semantic) - ssim_score(degraded.gt, bicubic),
                    boundary_f_score(degraded.gt, semantic) - boundary_f_score(degraded.gt, bicubic),
                )
            )
        for config in build_degradation_plan(scales=(2, 4), modes=("fog", "mixed")):
            degraded = degrade_sample(sample, config=config, seed=23)
            bicubic = bicubic_restore(degraded.low_res, degraded.gt.shape[:2])
            semantic = semantic_frequency_restore(degraded.low_res, degraded.mask, degraded.gt.shape[:2], degradation_name=degraded.degradation_name)
            fog_deltas.append(psnr(degraded.gt, semantic) - psnr(degraded.gt, bicubic))

    nonfog = np.asarray(nonfog_deltas)
    assert nonfog[:, 0].mean() > -0.25
    assert nonfog[:, 1].mean() > -0.006
    assert nonfog[:, 2].mean() > -0.006
    assert np.mean(fog_deltas) > 4.0


def test_weedsgalore_semantic_frequency_keeps_boundary_loss_small():
    from cea_plus.metrics import boundary_f_score, psnr, proxy_segmentation_miou
    from cea_plus.restoration import bicubic_restore, semantic_frequency_restore

    dataset_root = Path("data/external/weedsgalore-dataset")
    if not dataset_root.exists():
        return

    samples = load_weedsgalore_dataset(dataset_root=dataset_root, split="test", limit=3)
    deltas = []
    for sample in samples:
        for config in build_degradation_plan(scales=(2, 4), modes=("rain", "fog", "noise", "blur", "jpeg", "mixed")):
            degraded = degrade_sample(sample, config=config, seed=23)
            bicubic = bicubic_restore(degraded.low_res, degraded.gt.shape[:2])
            semantic = semantic_frequency_restore(degraded.low_res, degraded.mask, degraded.gt.shape[:2], degradation_name=degraded.degradation_name)
            deltas.append(
                (
                    psnr(degraded.gt, semantic) - psnr(degraded.gt, bicubic),
                    boundary_f_score(degraded.gt, semantic) - boundary_f_score(degraded.gt, bicubic),
                    proxy_segmentation_miou(degraded.mask, semantic) - proxy_segmentation_miou(degraded.mask, bicubic),
                )
            )

    values = np.asarray(deltas)
    assert values[:, 0].mean() > 1.3
    assert values[:, 1].mean() > -0.010
    assert values[:, 2].mean() > 0.08


def test_weedsgalore_fog_mixed_supports_paper_strength_claims():
    from cea_plus.metrics import boundary_f_score, proxy_segmentation_miou, psnr, ssim_score
    from cea_plus.restoration import bicubic_restore, restore_all_methods

    dataset_root = Path("data/external/weedsgalore-dataset")
    if not dataset_root.exists():
        return

    samples = load_weedsgalore_dataset(dataset_root=dataset_root, split="test", limit=3)
    deltas = []
    for sample in samples:
        for config in build_degradation_plan(scales=(2, 4), modes=("fog", "mixed")):
            degraded = degrade_sample(sample, config=config, seed=29)
            bicubic = bicubic_restore(degraded.low_res, degraded.gt.shape[:2])
            outputs = restore_all_methods(degraded.low_res, degraded.mask, degraded.gt.shape[:2], degradation_name=degraded.degradation_name)
            semantic = outputs["semantic_frequency"]
            structure = outputs["structure_only"]
            deltas.append(
                (
                    psnr(degraded.gt, semantic) - psnr(degraded.gt, bicubic),
                    ssim_score(degraded.gt, semantic) - ssim_score(degraded.gt, bicubic),
                    boundary_f_score(degraded.gt, semantic) - boundary_f_score(degraded.gt, bicubic),
                    proxy_segmentation_miou(degraded.mask, semantic) - proxy_segmentation_miou(degraded.mask, bicubic),
                    proxy_segmentation_miou(degraded.mask, semantic) - proxy_segmentation_miou(degraded.mask, structure),
                )
            )

    values = np.asarray(deltas)
    assert values[:, 0].mean() > 4.0
    assert values[:, 1].mean() > -0.002
    assert values[:, 2].mean() > -0.008
    assert values[:, 3].mean() > 0.25
    assert values[:, 4].mean() > 0.010


def test_weedsgalore_edge_aware_ablation_exposes_boundary_task_tradeoff():
    from cea_plus.metrics import boundary_f_score, proxy_segmentation_miou

    dataset_root = Path("data/external/weedsgalore-dataset")
    if not dataset_root.exists():
        return

    samples = load_weedsgalore_dataset(dataset_root=dataset_root, split="test", limit=3)
    deltas = []
    for sample in samples:
        for config in build_degradation_plan(scales=(2, 4), modes=("fog", "mixed")):
            degraded = degrade_sample(sample, config=config, seed=31)
            outputs = restore_all_methods(
                degraded.low_res,
                degraded.mask,
                degraded.gt.shape[:2],
                degradation_name=degraded.degradation_name,
            )
            semantic = outputs["semantic_frequency"]
            edge_aware = outputs["semantic_edge_aware"]
            deltas.append(
                (
                    boundary_f_score(degraded.gt, edge_aware) - boundary_f_score(degraded.gt, semantic),
                    proxy_segmentation_miou(degraded.mask, edge_aware) - proxy_segmentation_miou(degraded.mask, semantic),
                )
            )

    values = np.asarray(deltas)
    assert values[:, 0].mean() > 0.004
    assert values[:, 1].mean() > -0.18


def test_weedsgalore_edge_aware_ablation_turns_boundary_positive_over_full_degradation_mix():
    from cea_plus.metrics import boundary_f_score, proxy_segmentation_miou
    from cea_plus.restoration import bicubic_restore

    dataset_root = Path("data/external/weedsgalore-dataset")
    if not dataset_root.exists():
        return

    samples = load_weedsgalore_dataset(dataset_root=dataset_root, split="test", limit=3)
    deltas = []
    for sample in samples:
        for config in build_degradation_plan(scales=(2, 4), modes=("rain", "fog", "noise", "blur", "jpeg", "mixed")):
            degraded = degrade_sample(sample, config=config, seed=37)
            bicubic = bicubic_restore(degraded.low_res, degraded.gt.shape[:2])
            edge_aware = restore_all_methods(
                degraded.low_res,
                degraded.mask,
                degraded.gt.shape[:2],
                degradation_name=degraded.degradation_name,
            )["semantic_edge_aware"]
            deltas.append(
                (
                    boundary_f_score(degraded.gt, edge_aware) - boundary_f_score(degraded.gt, bicubic),
                    proxy_segmentation_miou(degraded.mask, edge_aware) - proxy_segmentation_miou(degraded.mask, bicubic),
                )
            )

    values = np.asarray(deltas)
    assert values[:, 0].mean() > 0.0003
    assert values[:, 1].mean() > 0.03


def test_weedsgalore_boundary_guard_ablation_preserves_boundary_against_bicubic():
    from cea_plus.metrics import boundary_f_score, proxy_segmentation_miou
    from cea_plus.restoration import bicubic_restore

    dataset_root = Path("data/external/weedsgalore-dataset")
    if not dataset_root.exists():
        return

    samples = load_weedsgalore_dataset(dataset_root=dataset_root, split="test", limit=3)
    deltas = []
    for sample in samples:
        for config in build_degradation_plan(scales=(2, 4), modes=("fog", "mixed")):
            degraded = degrade_sample(sample, config=config, seed=41)
            bicubic = bicubic_restore(degraded.low_res, degraded.gt.shape[:2])
            boundary_guard = restore_all_methods(
                degraded.low_res,
                degraded.mask,
                degraded.gt.shape[:2],
                degradation_name=degraded.degradation_name,
            )["semantic_boundary_guard"]
            deltas.append(
                (
                    boundary_f_score(degraded.gt, boundary_guard) - boundary_f_score(degraded.gt, bicubic),
                    proxy_segmentation_miou(degraded.mask, boundary_guard) - proxy_segmentation_miou(degraded.mask, bicubic),
                )
            )

    values = np.asarray(deltas)
    assert values[:, 0].mean() > 0.001
    assert values[:, 1].mean() > 0.010


def test_loveda_boundary_guard_does_not_amplify_boundary_loss():
    from cea_plus.metrics import boundary_f_score, proxy_segmentation_miou
    from cea_plus.restoration import bicubic_restore

    dataset_root = Path("data/external/loveda")
    if not dataset_root.exists():
        return

    samples = load_loveda_dataset(
        dataset_root=dataset_root,
        split="Val",
        domain="Rural",
        target_labels=(7,),
        crop_size=512,
        crop_strategy="random",
        crop_seed=101,
        limit=6,
    )
    deltas = []
    for sample in samples:
        for config in build_degradation_plan(scales=(2, 4), modes=("fog", "mixed")):
            degraded = degrade_sample(sample, config=config, seed=43)
            bicubic = bicubic_restore(degraded.low_res, degraded.gt.shape[:2])
            outputs = restore_all_methods(
                degraded.low_res,
                degraded.mask,
                degraded.gt.shape[:2],
                degradation_name=degraded.degradation_name,
            )
            semantic = outputs["semantic_frequency"]
            boundary_guard = outputs["semantic_boundary_guard"]
            deltas.append(
                (
                    boundary_f_score(degraded.gt, boundary_guard) - boundary_f_score(degraded.gt, bicubic),
                    boundary_f_score(degraded.gt, semantic) - boundary_f_score(degraded.gt, bicubic),
                    proxy_segmentation_miou(degraded.mask, boundary_guard) - proxy_segmentation_miou(degraded.mask, bicubic),
                )
            )

    values = np.asarray(deltas)
    assert values[:, 0].mean() >= values[:, 1].mean() - 0.001
    assert values[:, 2].mean() > -0.001


def test_cli_runs_local_image_mask_dataset(tmp_path: Path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    output_dir = tmp_path / "cli_results"
    image_dir.mkdir()
    mask_dir.mkdir()

    for idx in range(2):
        image = np.zeros((96, 96, 3), dtype=np.float32)
        image[..., 0] = 0.35
        image[..., 1] = 0.24
        image[..., 2] = 0.13
        image[:, 20 + idx : 34 + idx, 1] = 0.65
        mask = np.zeros((96, 96), dtype=bool)
        mask[:, 20 + idx : 34 + idx] = True
        _write_rgb(image_dir / f"plot_{idx}.png", image)
        _write_mask(mask_dir / f"plot_{idx}.png", mask)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cea_plus.run_experiment",
            "--image-dir",
            str(image_dir),
            "--mask-dir",
            str(mask_dir),
            "--out",
            str(output_dir),
            "--limit",
            "2",
            "--scales",
            "4",
            "--modes",
            "fog,mixed",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "samples=4" in result.stdout
    assert (output_dir / "metrics.csv").exists()
    assert "x4_fog" in (output_dir / "metrics.csv").read_text(encoding="utf-8")


def test_cli_runs_weedsgalore_dataset_when_present(tmp_path: Path):
    dataset_root = Path("data/external/weedsgalore-dataset")
    if not dataset_root.exists():
        return

    output_dir = tmp_path / "weedsgalore_results"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cea_plus.run_experiment",
            "--weedsgalore-root",
            str(dataset_root),
            "--split",
            "test",
            "--limit",
            "2",
            "--out",
            str(output_dir),
            "--scales",
            "4",
            "--modes",
            "fog",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "samples=2" in result.stdout
    assert (output_dir / "metrics.csv").exists()
    assert "2023-" in (output_dir / "metrics.csv").read_text(encoding="utf-8")


def test_cli_runs_loveda_dataset(tmp_path: Path):
    root = tmp_path / "loveda"
    image_dir = root / "Val" / "Rural" / "images_png"
    mask_dir = root / "Val" / "Rural" / "masks_png"
    output_dir = tmp_path / "loveda_results"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    for idx in range(2):
        image = np.zeros((64, 64, 3), dtype=np.float32)
        image[..., 0] = 0.22
        image[..., 1] = 0.42
        image[..., 2] = 0.18
        image[:, 16 + idx : 28 + idx, 1] = 0.7
        labels = np.zeros((64, 64), dtype=np.uint8)
        labels[:, 16 + idx : 28 + idx] = 7
        _write_rgb(image_dir / f"tile_{idx}.png", image)
        Image.fromarray(labels).save(mask_dir / f"tile_{idx}.png")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cea_plus.run_experiment",
            "--loveda-root",
            str(root),
            "--split",
            "Val",
            "--domain",
            "Rural",
            "--target-labels",
            "7",
            "--out",
            str(output_dir),
            "--limit",
            "2",
            "--scales",
            "4",
            "--modes",
            "fog",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "samples=2" in result.stdout
    assert (output_dir / "metrics.csv").exists()
    assert "Val_Rural_tile_0" in (output_dir / "metrics.csv").read_text(encoding="utf-8")
