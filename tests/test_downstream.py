from pathlib import Path
import subprocess
import sys

import numpy as np

from cea_plus.degradation import build_degradation_plan
from cea_plus.downstream import evaluate_frozen_segmenter, load_segmenter, save_segmenter, train_pixel_segmenter
from cea_plus.downstream_eval import run_downstream_experiment
from cea_plus.synthesis import make_synthetic_agri_sample


def test_pixel_segmenter_learns_crop_mask_and_round_trips(tmp_path: Path):
    train_samples = [make_synthetic_agri_sample(seed=200 + idx, size=96) for idx in range(4)]
    test_sample = make_synthetic_agri_sample(seed=240, size=96)

    segmenter = train_pixel_segmenter(train_samples, max_pixels_per_sample=600, seed=5)
    score = evaluate_frozen_segmenter(segmenter, test_sample.image, test_sample.mask)

    model_path = tmp_path / "pixel_segmenter.joblib"
    save_segmenter(segmenter, model_path)
    loaded = load_segmenter(model_path)
    loaded_score = evaluate_frozen_segmenter(loaded, test_sample.image, test_sample.mask)

    assert model_path.exists()
    assert score > 0.75
    assert loaded_score == score


def test_downstream_experiment_writes_frozen_metrics_and_report(tmp_path: Path):
    train_samples = [make_synthetic_agri_sample(seed=300 + idx, size=96) for idx in range(4)]
    test_samples = [make_synthetic_agri_sample(seed=340 + idx, size=96).with_name(f"heldout_{idx}") for idx in range(2)]
    plan = build_degradation_plan(scales=(4,), modes=("fog",))

    summary = run_downstream_experiment(
        output_dir=tmp_path / "downstream",
        train_samples=train_samples,
        test_samples=test_samples,
        degradation_plan=plan,
        max_pixels_per_sample=600,
        seed=11,
    )

    metrics_text = (tmp_path / "downstream" / "downstream_metrics.csv").read_text(encoding="utf-8")
    report_text = (tmp_path / "downstream" / "downstream_report.md").read_text(encoding="utf-8")

    assert summary.sample_count == 2
    assert "frozen_miou" in metrics_text.splitlines()[0]
    assert "semantic_frequency" in metrics_text
    assert "冻结下游分割" in report_text
    assert summary.significant_metrics == ("frozen_miou",)


def test_pixel_segmenter_trains_on_weedsgalore_when_present():
    from cea_plus.dataset import load_weedsgalore_dataset

    dataset_root = Path("data/external/weedsgalore-dataset")
    if not dataset_root.exists():
        return

    train_samples = load_weedsgalore_dataset(dataset_root=dataset_root, split="train", limit=4)
    test_samples = load_weedsgalore_dataset(dataset_root=dataset_root, split="test", limit=2)

    segmenter = train_pixel_segmenter(train_samples, max_pixels_per_sample=1200, seed=13)
    scores = [evaluate_frozen_segmenter(segmenter, sample.image, sample.mask) for sample in test_samples]

    assert np.mean(scores) > 0.45


def test_downstream_cli_runs_weedsgalore_when_present(tmp_path: Path):
    dataset_root = Path("data/external/weedsgalore-dataset")
    if not dataset_root.exists():
        return

    output_dir = tmp_path / "downstream_cli"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cea_plus.run_downstream",
            "--weedsgalore-root",
            str(dataset_root),
            "--train-split",
            "train",
            "--test-split",
            "test",
            "--train-limit",
            "4",
            "--test-limit",
            "2",
            "--out",
            str(output_dir),
            "--scales",
            "4",
            "--modes",
            "fog",
            "--max-pixels-per-sample",
            "600",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "samples=2" in result.stdout
    assert (output_dir / "downstream_metrics.csv").exists()
    assert "frozen_miou" in (output_dir / "downstream_metrics.csv").read_text(encoding="utf-8")
