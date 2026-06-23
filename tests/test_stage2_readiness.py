from pathlib import Path

import numpy as np
from PIL import Image

from cea_plus.stage2 import write_stage2_readiness


def _write_pair(root: Path, name: str = "plot_001") -> None:
    image_dir = root / "dataset" / "images"
    mask_dir = root / "dataset" / "masks"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[..., 1] = 120
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[:, 20:35] = 255
    Image.fromarray(image).save(image_dir / f"{name}.png")
    Image.fromarray(mask).save(mask_dir / f"{name}.png")


def test_stage2_readiness_reports_dataset_sources_and_missing_baselines(tmp_path: Path):
    summary = write_stage2_readiness(root=tmp_path, output_dir=tmp_path / "stage2")

    report = (tmp_path / "stage2" / "stage2_readiness_report.md").read_text(encoding="utf-8")
    sources = (tmp_path / "stage2" / "dataset_sources.md").read_text(encoding="utf-8")
    baselines = (tmp_path / "stage2" / "baseline_status.csv").read_text(encoding="utf-8")

    assert summary.real_dataset_count == 0
    assert "真实数据目录：未发现" in report
    assert "LoveDA" in sources
    assert "Agriculture-Vision" in sources
    assert "WeedsGalore" in sources
    assert "SwinIR" in baselines
    assert "missing" in baselines


def test_stage2_readiness_detects_local_paired_dataset_and_efficiency(tmp_path: Path):
    _write_pair(tmp_path)
    summary = write_stage2_readiness(root=tmp_path, output_dir=tmp_path / "stage2", run_efficiency=True)

    report = (tmp_path / "stage2" / "stage2_readiness_report.md").read_text(encoding="utf-8")
    efficiency = (tmp_path / "stage2" / "efficiency.csv").read_text(encoding="utf-8")

    assert summary.real_dataset_count == 1
    assert "dataset/images" in report
    assert "semantic_frequency" in efficiency
    assert "avg_ms" in efficiency.splitlines()[0]


def test_stage2_readiness_report_reflects_available_official_weight(tmp_path: Path):
    weight_dir = tmp_path / "external_baselines" / "SwinIR" / "model_zoo" / "swinir"
    weight_dir.mkdir(parents=True)
    (weight_dir / "mock.pth").write_bytes(b"weight")

    write_stage2_readiness(root=tmp_path, output_dir=tmp_path / "stage2")

    report = (tmp_path / "stage2" / "stage2_readiness_report.md").read_text(encoding="utf-8")
    assert "SwinIR: available" in report
