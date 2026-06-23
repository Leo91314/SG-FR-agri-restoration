from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from cea_plus.deep_baselines import (
    ensure_swinir_lightweight_weight,
    ensure_swin2sr_classical_weight,
    ensure_hat_imagenet_weight,
    hat_imagenet_weight_info,
    load_official_restormer_baseline,
    load_official_hat_baseline,
    load_official_liif_baseline,
    load_official_lte_baseline,
    load_official_swin2sr_baseline,
    official_baseline_specs,
    load_official_swinir_baseline,
    swin2sr_classical_weight_info,
    run_restormer_baseline_experiment,
    run_hat_baseline_experiment,
    run_deep_baseline_experiment,
    swinir_lightweight_weight_info,
    train_tiny_rescnn_baseline,
)
from cea_plus.degradation import build_degradation_plan, degrade_sample
from cea_plus.metrics import psnr
from cea_plus.restoration import bicubic_restore
from cea_plus.synthesis import make_synthetic_agri_sample


def test_official_baseline_specs_track_required_models():
    specs = {spec.name: spec for spec in official_baseline_specs()}

    assert {"SwinIR", "Swin2SR", "HAT", "Restormer", "LIIF", "LTE"}.issubset(specs)
    assert specs["SwinIR"].official_url.startswith("https://github.com/")
    assert specs["Restormer"].task
    assert specs["LIIF"].local_dir.name == "LIIF"
    assert specs["LTE"].status in {"missing", "code_only", "available"}
    assert specs["Swin2SR"].local_dir.name == "swin2sr"
    assert specs["HAT"].weight_hint


def test_swinir_lightweight_weight_info_uses_official_release_names():
    info = swinir_lightweight_weight_info(scale=4)

    assert info.filename == "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"
    assert info.url == (
        "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/"
        "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"
    )

    with pytest.raises(ValueError):
        swinir_lightweight_weight_info(scale=3)


def test_swin2sr_classical_weight_info_uses_official_release_names():
    info = swin2sr_classical_weight_info(scale=4)

    assert info.filename == "Swin2SR_ClassicalSR_X4_64.pth"
    assert info.url == "https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/Swin2SR_ClassicalSR_X4_64.pth"

    with pytest.raises(ValueError):
        swin2sr_classical_weight_info(scale=3)


def test_hat_imagenet_weight_info_uses_official_google_drive_ids():
    info = hat_imagenet_weight_info(scale=4)

    assert info.filename == "HAT_SRx4_ImageNet-pretrain.pth"
    assert info.gdrive_id == "1cxls85ZE7kalhNy47eBJI_L_Lwf9hxRI"
    assert info.local_path == Path("external_baselines/HAT/experiments/pretrained_models/HAT_SRx4_ImageNet-pretrain.pth")

    with pytest.raises(ValueError):
        hat_imagenet_weight_info(scale=3)


def test_ensure_swinir_weight_reuses_existing_file(tmp_path: Path):
    repo = tmp_path / "SwinIR"
    weight_dir = repo / "model_zoo" / "swinir"
    weight_dir.mkdir(parents=True)
    expected = weight_dir / "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"
    expected.write_bytes(b"already here")

    actual = ensure_swinir_lightweight_weight(repo_dir=repo, scale=2)

    assert actual == expected
    assert expected.read_bytes() == b"already here"


def test_ensure_swinir_weight_falls_back_to_curl_when_requests_fails(tmp_path: Path, monkeypatch):
    import cea_plus.deep_baselines as deep_baselines

    def fail_request(*args, **kwargs):
        raise deep_baselines.requests.exceptions.SSLError("test ssl failure")

    def fake_curl(cmd, check):
        output_path = Path(cmd[cmd.index("-o") + 1])
        output_path.write_bytes(b"downloaded by curl")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(deep_baselines.requests, "get", fail_request)
    monkeypatch.setattr(deep_baselines.subprocess, "run", fake_curl)

    path = ensure_swinir_lightweight_weight(repo_dir=tmp_path / "SwinIR", scale=4)

    assert path.read_bytes() == b"downloaded by curl"


def test_ensure_swin2sr_classical_weight_reuses_existing_file(tmp_path: Path):
    repo = tmp_path / "swin2sr"
    weight_dir = repo / "model_zoo" / "swin2sr"
    weight_dir.mkdir(parents=True)
    expected = weight_dir / "Swin2SR_ClassicalSR_X2_64.pth"
    expected.write_bytes(b"already here")

    actual = ensure_swin2sr_classical_weight(repo_dir=repo, scale=2)

    assert actual == expected
    assert expected.read_bytes() == b"already here"


def test_ensure_hat_imagenet_weight_reuses_existing_file(tmp_path: Path):
    repo = tmp_path / "HAT"
    weight_dir = repo / "experiments" / "pretrained_models"
    weight_dir.mkdir(parents=True)
    expected = weight_dir / "HAT_SRx2_ImageNet-pretrain.pth"
    expected.write_bytes(b"already here")

    actual = ensure_hat_imagenet_weight(repo_dir=repo, scale=2)

    assert actual == expected
    assert expected.read_bytes() == b"already here"


def test_official_swinir_baseline_restores_expected_shape(tmp_path: Path):
    repo = Path("external_baselines/SwinIR")
    if not repo.exists():
        return

    from cea_plus.deep_baselines import _build_swinir_lightweight_model

    weight_path = tmp_path / "mock_swinir_x2.pth"
    model = _build_swinir_lightweight_model(repo_dir=repo, scale=2, device=torch.device("cpu"))
    torch.save({"params": model.state_dict()}, weight_path)

    baseline = load_official_swinir_baseline(
        repo_dir=repo,
        scale=2,
        weight_path=weight_path,
        device=torch.device("cpu"),
        tile=8,
    )
    low_res = np.zeros((8, 8, 3), dtype=np.float32)
    restored = baseline.restore(low_res, output_shape=(16, 16))

    assert restored.shape == (16, 16, 3)
    assert restored.dtype == np.float32
    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0


def test_official_swin2sr_baseline_restores_expected_shape(tmp_path: Path):
    repo = Path("external_baselines/swin2sr")
    if not repo.exists():
        return

    from cea_plus.deep_baselines import _build_swin2sr_classical_model

    weight_path = tmp_path / "mock_swin2sr_x2.pth"
    model = _build_swin2sr_classical_model(repo_dir=repo, scale=2, device=torch.device("cpu"))
    torch.save({"params": model.state_dict()}, weight_path)

    baseline = load_official_swin2sr_baseline(
        repo_dir=repo,
        scale=2,
        weight_path=weight_path,
        device=torch.device("cpu"),
        tile=8,
    )
    low_res = np.zeros((8, 8, 3), dtype=np.float32)
    restored = baseline.restore(low_res, output_shape=(16, 16))

    assert restored.shape == (16, 16, 3)
    assert restored.dtype == np.float32
    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0


def test_official_hat_baseline_restores_expected_shape(tmp_path: Path):
    repo = Path("external_baselines/HAT")
    if not repo.exists():
        return

    from cea_plus.deep_baselines import _build_hat_imagenet_model

    weight_path = tmp_path / "mock_hat_x2.pth"
    model = _build_hat_imagenet_model(repo_dir=repo, scale=2, device=torch.device("cpu"))
    torch.save({"params_ema": model.state_dict()}, weight_path)

    baseline = load_official_hat_baseline(
        repo_dir=repo,
        scale=2,
        weight_path=weight_path,
        device=torch.device("cpu"),
        tile=16,
    )
    low_res = np.zeros((8, 8, 3), dtype=np.float32)
    restored = baseline.restore(low_res, output_shape=(16, 16))

    assert restored.shape == (16, 16, 3)
    assert restored.dtype == np.float32
    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0


def test_official_restormer_baseline_restores_expected_shape(tmp_path: Path):
    repo = Path("external_baselines/Restormer")
    if not repo.exists():
        return

    from cea_plus.deep_baselines import _build_restormer_model

    weight_path = tmp_path / "mock_restormer.pth"
    model = _build_restormer_model(repo_dir=repo, device=torch.device("cpu"))
    torch.save({"params": model.state_dict()}, weight_path)

    baseline = load_official_restormer_baseline(
        repo_dir=repo,
        weight_path=weight_path,
        device=torch.device("cpu"),
        tile=16,
    )
    low_res = np.zeros((8, 8, 3), dtype=np.float32)
    restored = baseline.restore(low_res, output_shape=(32, 32))

    assert restored.shape == (32, 32, 3)
    assert restored.dtype == np.float32
    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0


def test_tiny_rescnn_baseline_learns_synthetic_restoration(tmp_path: Path):
    train_samples = [make_synthetic_agri_sample(seed=800 + idx, size=64) for idx in range(2)]
    test_sample = make_synthetic_agri_sample(seed=850, size=64)
    config = build_degradation_plan(scales=(4,), modes=("fog",))[0]
    degraded = degrade_sample(test_sample, config=config, seed=99)
    bicubic = bicubic_restore(degraded.low_res, degraded.gt.shape[:2])

    baseline = train_tiny_rescnn_baseline(
        train_samples=train_samples,
        degradation_plan=[config],
        model_path=tmp_path / "tiny_rescnn.pt",
        seed=10,
        steps=30,
        learning_rate=0.01,
    )
    restored = baseline.restore(degraded.low_res, degraded.gt.shape[:2])

    assert restored.shape == degraded.gt.shape
    assert restored.dtype == np.float32
    assert (tmp_path / "tiny_rescnn.pt").exists()
    assert psnr(degraded.gt, restored) > psnr(degraded.gt, bicubic)


def test_run_deep_baseline_experiment_writes_metrics_and_status(tmp_path: Path):
    train_samples = [make_synthetic_agri_sample(seed=900 + idx, size=64) for idx in range(2)]
    test_samples = [make_synthetic_agri_sample(seed=930 + idx, size=64).with_name(f"heldout_{idx}") for idx in range(2)]
    plan = build_degradation_plan(scales=(4,), modes=("fog",))

    summary = run_deep_baseline_experiment(
        output_dir=tmp_path / "deep",
        train_samples=train_samples,
        test_samples=test_samples,
        degradation_plan=plan,
        seed=12,
        steps=20,
    )

    metrics = (tmp_path / "deep" / "deep_baseline_metrics.csv").read_text(encoding="utf-8")
    status = (tmp_path / "deep" / "official_baseline_status.csv").read_text(encoding="utf-8")
    report = (tmp_path / "deep" / "deep_baseline_report.md").read_text(encoding="utf-8")

    assert summary.row_count == 4
    assert "tiny_rescnn" in metrics
    assert "task_miou" in metrics.splitlines()[0]
    assert "SwinIR" in status
    assert "Restormer" in status
    assert "Tiny ResCNN" in report


def test_run_deep_baseline_experiment_can_include_official_swinir(tmp_path: Path):
    repo = Path("external_baselines/SwinIR")
    if not repo.exists():
        return

    from cea_plus.deep_baselines import _build_swinir_lightweight_model

    train_samples = [make_synthetic_agri_sample(seed=960, size=64)]
    test_samples = [make_synthetic_agri_sample(seed=961, size=64).with_name("heldout")]
    plan = build_degradation_plan(scales=(2,), modes=("fog",))
    weight_path = tmp_path / "mock_swinir_x2.pth"
    model = _build_swinir_lightweight_model(repo_dir=repo, scale=2, device=torch.device("cpu"))
    torch.save({"params": model.state_dict()}, weight_path)

    summary = run_deep_baseline_experiment(
        output_dir=tmp_path / "deep_swinir",
        train_samples=train_samples,
        test_samples=test_samples,
        degradation_plan=plan,
        seed=14,
        steps=1,
        include_swinir=True,
        swinir_repo_dir=repo,
        swinir_weight_paths={2: weight_path},
        swinir_tile=8,
    )

    metrics = (tmp_path / "deep_swinir" / "deep_baseline_metrics.csv").read_text(encoding="utf-8")
    assert summary.row_count == 3
    assert "swinir_lightweight" in metrics


def test_run_restormer_baseline_experiment_writes_metrics(tmp_path: Path):
    repo = Path("external_baselines/Restormer")
    if not repo.exists():
        return

    from cea_plus.deep_baselines import _build_restormer_model

    samples = [make_synthetic_agri_sample(seed=980, size=32).with_name("heldout")]
    plan = build_degradation_plan(scales=(2,), modes=("rain",))
    weight_path = tmp_path / "mock_restormer.pth"
    model = _build_restormer_model(repo_dir=repo, device=torch.device("cpu"))
    torch.save({"params": model.state_dict()}, weight_path)

    summary = run_restormer_baseline_experiment(
        output_dir=tmp_path / "restormer",
        test_samples=samples,
        degradation_plan=plan,
        weight_path=weight_path,
        repo_dir=repo,
        tile=16,
        seed=19,
    )

    metrics = (tmp_path / "restormer" / "restormer_metrics.csv").read_text(encoding="utf-8")
    assert summary.row_count == 2
    assert "restormer_deraining" in metrics
    assert "task_miou" in metrics.splitlines()[0]


def test_run_hat_baseline_experiment_writes_metrics(tmp_path: Path):
    repo = Path("external_baselines/HAT")
    if not repo.exists():
        return

    from cea_plus.deep_baselines import _build_hat_imagenet_model

    samples = [make_synthetic_agri_sample(seed=990, size=32).with_name("heldout")]
    plan = build_degradation_plan(scales=(2,), modes=("fog",))
    weight_path = tmp_path / "mock_hat_x2.pth"
    model = _build_hat_imagenet_model(repo_dir=repo, scale=2, device=torch.device("cpu"))
    torch.save({"params_ema": model.state_dict()}, weight_path)

    summary = run_hat_baseline_experiment(
        output_dir=tmp_path / "hat",
        test_samples=samples,
        degradation_plan=plan,
        weight_paths={2: weight_path},
        repo_dir=repo,
        tile=16,
        seed=23,
    )

    metrics = (tmp_path / "hat" / "hat_metrics.csv").read_text(encoding="utf-8")
    assert summary.row_count == 2
    assert "hat_imagenet" in metrics
    assert "task_miou" in metrics.splitlines()[0]


def test_run_hat_baseline_experiment_flushes_completed_rows_on_failure(tmp_path: Path, monkeypatch):
    import cea_plus.deep_baselines as deep_baselines

    class FailingHatBaseline:
        weight_path = tmp_path / "mock_hat.pth"

        def __init__(self) -> None:
            self.calls = 0

        def restore(self, low_res, output_shape):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("simulated HAT interruption")
            return bicubic_restore(low_res, output_shape)

    baseline = FailingHatBaseline()
    monkeypatch.setattr(deep_baselines, "load_official_hat_baseline", lambda *args, **kwargs: baseline)
    samples = [
        make_synthetic_agri_sample(seed=1000, size=32).with_name("heldout_0"),
        make_synthetic_agri_sample(seed=1001, size=32).with_name("heldout_1"),
    ]
    plan = build_degradation_plan(scales=(2,), modes=("fog",))

    with pytest.raises(RuntimeError, match="simulated HAT interruption"):
        run_hat_baseline_experiment(
            output_dir=tmp_path / "hat_resume",
            test_samples=samples,
            degradation_plan=plan,
            resume=True,
        )

    metrics_path = tmp_path / "hat_resume" / "hat_metrics.csv"
    metrics = metrics_path.read_text(encoding="utf-8").splitlines()
    assert len(metrics) == 3
    assert "heldout_0" in metrics[1]
    assert "hat_imagenet" in metrics[2]


def test_run_hat_baseline_experiment_resumes_existing_metrics(tmp_path: Path, monkeypatch):
    import cea_plus.deep_baselines as deep_baselines

    class CountingHatBaseline:
        weight_path = tmp_path / "mock_hat.pth"

        def __init__(self) -> None:
            self.calls = 0

        def restore(self, low_res, output_shape):
            self.calls += 1
            return bicubic_restore(low_res, output_shape)

    baseline = CountingHatBaseline()
    monkeypatch.setattr(deep_baselines, "load_official_hat_baseline", lambda *args, **kwargs: baseline)
    samples = [
        make_synthetic_agri_sample(seed=1010, size=32).with_name("heldout_0"),
        make_synthetic_agri_sample(seed=1011, size=32).with_name("heldout_1"),
    ]
    plan = build_degradation_plan(scales=(2,), modes=("fog",))
    output_dir = tmp_path / "hat_resume"
    output_dir.mkdir()
    metrics_path = output_dir / "hat_metrics.csv"
    metrics_path.write_text(
        "\n".join(
            [
                "sample,degradation,scale,method,psnr,ssim,boundary_f,task_miou",
                "heldout_0,x2_fog,2,bicubic,1,1,1,1",
                "heldout_0,x2_fog,2,hat_imagenet,1,1,1,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_hat_baseline_experiment(
        output_dir=output_dir,
        test_samples=samples,
        degradation_plan=plan,
        resume=True,
    )

    metrics = metrics_path.read_text(encoding="utf-8").splitlines()
    assert summary.row_count == 4
    assert baseline.calls == 1
    assert sum(1 for line in metrics if line.startswith("heldout_0,")) == 2
    assert sum(1 for line in metrics if line.startswith("heldout_1,")) == 2


def test_official_liif_baseline_restores_expected_shape_when_weight_present():
    repo = Path("external_baselines/LIIF")
    weight_path = repo / "save" / "edsr-baseline-liif.pth"
    if not weight_path.exists():
        return

    baseline = load_official_liif_baseline(
        repo_dir=repo,
        weight_path=weight_path,
        device=torch.device("cpu"),
        eval_bsize=1024,
    )
    low_res = np.zeros((8, 8, 3), dtype=np.float32)
    restored = baseline.restore(low_res, output_shape=(16, 16))

    assert restored.shape == (16, 16, 3)
    assert restored.dtype == np.float32
    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0


def test_official_lte_baseline_restores_expected_shape_when_weight_present():
    repo = Path("external_baselines/LTE")
    weight_path = repo / "save" / "edsr-baseline-lte.pth"
    if not weight_path.exists():
        return

    baseline = load_official_lte_baseline(
        repo_dir=repo,
        weight_path=weight_path,
        device=torch.device("cpu"),
        eval_bsize=1024,
    )
    low_res = np.zeros((8, 8, 3), dtype=np.float32)
    restored = baseline.restore(low_res, output_shape=(16, 16))

    assert restored.shape == (16, 16, 3)
    assert restored.dtype == np.float32
    assert float(restored.min()) >= 0.0
    assert float(restored.max()) <= 1.0


def test_official_lte_baseline_handles_mps_request_when_weight_present():
    repo = Path("external_baselines/LTE")
    weight_path = repo / "save" / "edsr-baseline-lte.pth"
    if not weight_path.exists() or not torch.backends.mps.is_available():
        return

    baseline = load_official_lte_baseline(
        repo_dir=repo,
        weight_path=weight_path,
        device=torch.device("mps"),
        eval_bsize=1024,
    )
    low_res = np.zeros((8, 8, 3), dtype=np.float32)
    restored = baseline.restore(low_res, output_shape=(16, 16))

    assert baseline.device.type == "cpu"
    assert restored.shape == (16, 16, 3)


def test_deep_baseline_cli_runs_weedsgalore_when_present(tmp_path: Path):
    dataset_root = Path("data/external/weedsgalore-dataset")
    if not dataset_root.exists():
        return

    output_dir = tmp_path / "deep_cli"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cea_plus.run_deep_baselines",
            "--weedsgalore-root",
            str(dataset_root),
            "--train-limit",
            "2",
            "--test-limit",
            "1",
            "--out",
            str(output_dir),
            "--scales",
            "4",
            "--modes",
            "fog",
            "--steps",
            "10",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "rows=2" in result.stdout
    assert (output_dir / "deep_baseline_metrics.csv").exists()
    assert "tiny_rescnn" in (output_dir / "deep_baseline_metrics.csv").read_text(encoding="utf-8")


def test_deep_baseline_cli_can_include_official_swinir_with_weight_path(tmp_path: Path):
    repo = Path("external_baselines/SwinIR")
    if not repo.exists():
        return

    from cea_plus.deep_baselines import _build_swinir_lightweight_model

    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()
    for idx in range(2):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[..., 1] = 80 + idx * 20
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 4:12] = 255
        Image.fromarray(image).save(image_dir / f"sample_{idx}.png")
        Image.fromarray(mask).save(mask_dir / f"sample_{idx}.png")

    weight_path = tmp_path / "mock_swinir_x2.pth"
    model = _build_swinir_lightweight_model(repo_dir=repo, scale=2, device=torch.device("cpu"))
    torch.save({"params": model.state_dict()}, weight_path)
    output_dir = tmp_path / "deep_cli_swinir"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cea_plus.run_deep_baselines",
            "--image-dir",
            str(image_dir),
            "--mask-dir",
            str(mask_dir),
            "--train-limit",
            "1",
            "--test-limit",
            "1",
            "--out",
            str(output_dir),
            "--scales",
            "2",
            "--modes",
            "fog",
            "--steps",
            "1",
            "--include-swinir",
            "--swinir-repo",
            str(repo),
            "--swinir-weight-x2",
            str(weight_path),
            "--swinir-tile",
            "8",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env={"PYTHONPATH": "src"},
        text=True,
        capture_output=True,
        check=True,
    )

    assert "rows=3" in result.stdout
    assert "swinir_lightweight" in (output_dir / "deep_baseline_metrics.csv").read_text(encoding="utf-8")
