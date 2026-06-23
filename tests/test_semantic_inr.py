from pathlib import Path
import inspect

import torch

from cea_plus.degradation import build_degradation_plan, degrade_sample
from cea_plus.synthesis import make_synthetic_agri_sample


def test_semantic_inr_restore_api_excludes_oracle_inputs():
    from cea_plus.semantic_inr import restore_with_semantic_inr

    signature = inspect.signature(restore_with_semantic_inr)

    assert "mask" not in signature.parameters
    assert "gt_mask" not in signature.parameters
    assert "degradation_name" not in signature.parameters


def test_semantic_inr_forward_exposes_interpretable_fields():
    from cea_plus.semantic_inr import TinySemanticINR

    model = TinySemanticINR(hidden_channels=12)
    low_res = torch.rand(1, 3, 16, 16)

    output = model(low_res, output_shape=(32, 32))

    assert output.restored.shape == (1, 3, 32, 32)
    assert output.structure.shape == (1, 3, 32, 32)
    assert output.texture.shape == (1, 3, 32, 32)
    assert output.alpha.shape == (1, 1, 32, 32)
    assert output.base.shape == (1, 3, 32, 32)
    assert output.semantic_logits.shape == (1, 1, 32, 32)
    assert float(output.restored.detach().min()) >= 0.0
    assert float(output.restored.detach().max()) <= 1.0


def test_semantic_inr_encoder_has_spatial_receptive_field():
    from cea_plus.semantic_inr import TinySemanticINR

    model = TinySemanticINR(hidden_channels=12)
    conv_kernels = [module.kernel_size for module in model.encoder.modules() if isinstance(module, torch.nn.Conv2d)]

    assert any(kernel != (1, 1) for kernel in conv_kernels)


def test_semantic_inr_accepts_detail_preserving_base_strength():
    from cea_plus.semantic_inr import TinySemanticINR

    model = TinySemanticINR(
        hidden_channels=12,
        base_sharpen_strength=0.35,
        structure_residual_scale=0.05,
        texture_residual_scale=0.04,
        semantic_detail_boost=0.30,
    )

    assert model.base_sharpen_strength == 0.35
    assert model.structure_residual_scale == 0.05
    assert model.texture_residual_scale == 0.04
    assert model.semantic_detail_boost == 0.30


def test_semantic_inr_zero_initialized_heads_start_from_detail_base():
    from cea_plus.semantic_inr import TinySemanticINR, _smooth_tensor

    model = TinySemanticINR(
        hidden_channels=12,
        base_sharpen_strength=0.72,
        structure_residual_scale=0.04,
        texture_residual_scale=0.04,
        semantic_detail_boost=0.0,
    )
    low_res = torch.rand(1, 3, 16, 16)

    output = model(low_res, output_shape=(32, 32))
    up = torch.nn.functional.interpolate(low_res, size=(32, 32), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    expected = (up + 0.72 * (up - _smooth_tensor(up))).clamp(0.0, 1.0)

    assert torch.mean(torch.abs(output.restored - expected)) < 1e-5


def test_semantic_inr_training_step_reduces_reconstruction_loss():
    from cea_plus.semantic_inr import TinySemanticINR, train_semantic_inr_steps

    sample = make_synthetic_agri_sample(seed=101, size=32)
    config = build_degradation_plan(scales=(2,), modes=("fog",))[0]
    degraded = degrade_sample(sample, config=config, seed=303)

    model = TinySemanticINR(hidden_channels=12)
    history = train_semantic_inr_steps(
        model,
        batches=[(degraded.low_res, degraded.gt, degraded.mask)],
        steps=8,
        learning_rate=3e-3,
        task_loss_weight=0.0,
        seed=17,
        device=torch.device("cpu"),
    )

    assert history[-1]["reconstruction_loss"] < history[0]["reconstruction_loss"]
    assert "task_loss" in history[-1]


def test_semantic_inr_smoke_experiment_writes_metrics(tmp_path: Path):
    from cea_plus.semantic_inr_experiment import run_semantic_inr_smoke_experiment

    train_samples = [make_synthetic_agri_sample(seed=201 + idx, size=32).with_name(f"train_{idx}") for idx in range(2)]
    test_samples = [make_synthetic_agri_sample(seed=301, size=32).with_name("test_0")]

    summary = run_semantic_inr_smoke_experiment(
        output_dir=tmp_path / "inr_smoke",
        train_samples=train_samples,
        test_samples=test_samples,
        degradation_plan=build_degradation_plan(scales=(2,), modes=("fog",)),
        steps=8,
        seed=23,
        device=torch.device("cpu"),
    )

    metrics = summary.metrics_csv.read_text(encoding="utf-8")
    report = summary.report.read_text(encoding="utf-8")

    assert "semantic_inr_no_leak" in metrics
    assert "Semantic INR no-leak smoke" in report
