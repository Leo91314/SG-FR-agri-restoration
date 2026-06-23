from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

from cea_plus.deep_downstream import run_deep_downstream_experiment, train_torch_segmenter
from cea_plus.degradation import build_degradation_plan
from cea_plus.synthesis import make_synthetic_agri_sample


def test_torch_segmenter_trains_and_predicts_mask():
    train_samples = [make_synthetic_agri_sample(seed=1200 + idx, size=64) for idx in range(3)]
    test_sample = make_synthetic_agri_sample(seed=1210, size=64)

    segmenter = train_torch_segmenter(
        train_samples,
        architecture="tiny",
        steps=8,
        crop_size=48,
        seed=5,
    )
    pred = segmenter.predict_mask(test_sample.image)
    prob = segmenter.predict_probability(test_sample.image)

    assert pred.shape == test_sample.mask.shape
    assert pred.dtype == bool
    assert prob.shape == test_sample.mask.shape
    assert prob.dtype == np.float32
    assert float(prob.min()) >= 0.0
    assert float(prob.max()) <= 1.0


def test_deep_downstream_experiment_writes_metrics(tmp_path: Path):
    train_samples = [make_synthetic_agri_sample(seed=1230 + idx, size=64) for idx in range(3)]
    test_samples = [make_synthetic_agri_sample(seed=1240, size=64).with_name("heldout")]
    plan = build_degradation_plan(scales=(2,), modes=("fog",))

    summary = run_deep_downstream_experiment(
        output_dir=tmp_path / "deep_downstream",
        train_samples=train_samples,
        test_samples=test_samples,
        degradation_plan=plan,
        architecture="tiny",
        steps=8,
        crop_size=48,
        eval_size=64,
        seed=9,
        methods=("bicubic", "semantic_frequency"),
    )

    metrics = (tmp_path / "deep_downstream" / "deep_downstream_metrics.csv").read_text(encoding="utf-8")
    report = (tmp_path / "deep_downstream" / "deep_downstream_report.md").read_text(encoding="utf-8")

    assert summary.sample_count == 1
    assert "deep_frozen_miou" in metrics.splitlines()[0]
    assert "semantic_frequency" in metrics
    assert "Deep frozen downstream" in report


def test_deep_downstream_default_methods_include_boundary_guard(tmp_path: Path):
    train_samples = [make_synthetic_agri_sample(seed=1270 + idx, size=64) for idx in range(3)]
    test_samples = [make_synthetic_agri_sample(seed=1280, size=64).with_name("heldout")]
    plan = build_degradation_plan(scales=(2,), modes=("fog",))

    run_deep_downstream_experiment(
        output_dir=tmp_path / "deep_downstream_default_methods",
        train_samples=train_samples,
        test_samples=test_samples,
        degradation_plan=plan,
        architecture="tiny",
        steps=2,
        crop_size=48,
        eval_size=64,
        seed=10,
    )

    metrics = (tmp_path / "deep_downstream_default_methods" / "deep_downstream_metrics.csv").read_text(encoding="utf-8")

    assert "semantic_boundary_guard" in metrics


def test_deeplabv3_training_handles_single_image_batches():
    sample = make_synthetic_agri_sample(seed=1250, size=128)

    segmenter = train_torch_segmenter(
        [sample],
        architecture="deeplabv3",
        steps=1,
        crop_size=128,
        input_size=128,
        seed=3,
    )

    pred = segmenter.predict_mask(sample.image)
    assert pred.shape == sample.mask.shape


def test_smp_strong_segmenter_architectures_train_one_step():
    sample = make_synthetic_agri_sample(seed=1260, size=96)

    for architecture in ("deeplabv3plus", "segformer_b0"):
        segmenter = train_torch_segmenter(
            [sample],
            architecture=architecture,
            steps=1,
            crop_size=96,
            input_size=96,
            seed=4,
        )
        pred = segmenter.predict_mask(sample.image)

        assert pred.shape == sample.mask.shape
        assert segmenter.architecture == architecture


def test_deep_downstream_cli_runs_local_dataset(tmp_path: Path):
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    out_dir = tmp_path / "deep_cli"
    image_dir.mkdir()
    mask_dir.mkdir()

    for idx in range(4):
        image = np.zeros((64, 64, 3), dtype=np.float32)
        image[..., 0] = 0.24
        image[..., 1] = 0.34
        image[..., 2] = 0.18
        image[16:44, 14 + idx : 26 + idx, 1] = 0.72
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[16:44, 14 + idx : 26 + idx] = 255
        Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255)).save(image_dir / f"tile_{idx}.png")
        Image.fromarray(mask).save(mask_dir / f"tile_{idx}.png")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cea_plus.run_deep_downstream",
            "--image-dir",
            str(image_dir),
            "--mask-dir",
            str(mask_dir),
            "--out",
            str(out_dir),
            "--architecture",
            "tiny",
            "--steps",
            "2",
            "--crop-size",
            "48",
            "--eval-size",
            "64",
            "--train-limit",
            "2",
            "--test-limit",
            "2",
            "--scales",
            "2",
            "--modes",
            "fog",
            "--methods",
            "bicubic,semantic_frequency",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "samples=2" in result.stdout
    assert (out_dir / "deep_downstream_metrics.csv").exists()
    assert "semantic_frequency" in (out_dir / "deep_downstream_metrics.csv").read_text(encoding="utf-8")
