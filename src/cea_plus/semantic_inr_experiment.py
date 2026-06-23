from dataclasses import dataclass
from pathlib import Path
import csv
import json
from typing import Optional

import numpy as np
import torch

from .degradation import DegradationConfig, degrade_sample
from .metrics import boundary_f_score, proxy_segmentation_miou, psnr, ssim_score
from .restoration import bicubic_restore, uniform_sharp_restore
from .semantic_inr import TinySemanticINR, restore_with_semantic_inr, train_semantic_inr_steps
from .statistics import paired_significance


METRICS = ("psnr", "ssim", "boundary_f", "task_miou")


@dataclass(frozen=True)
class SemanticINRSmokeSummary:
    output_dir: Path
    metrics_csv: Path
    report: Path
    summary_json: Path
    model_path: Path
    train_history: Path


def _build_training_batches(train_samples: list, degradation_plan: list[DegradationConfig], seed: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for sample_idx, sample in enumerate(train_samples):
        for config_idx, config in enumerate(degradation_plan):
            degraded = degrade_sample(sample, config=config, seed=seed * 1000 + sample_idx * 100 + config_idx)
            batches.append((degraded.low_res, degraded.gt, degraded.mask))
    return batches


def _paired_delta(rows: list[dict[str, object]], method: str, metric: str) -> tuple[float, float, int]:
    keys = sorted({(row["sample"], row["degradation"], row["scale"]) for row in rows})
    by_key = {(row["sample"], row["degradation"], row["scale"], row["method"]): row for row in rows}
    base_values: list[float] = []
    method_values: list[float] = []
    for sample, degradation, scale in keys:
        base = by_key.get((sample, degradation, scale, "bicubic"))
        proposed = by_key.get((sample, degradation, scale, method))
        if base is not None and proposed is not None:
            base_values.append(float(base[metric]))
            method_values.append(float(proposed[metric]))
    if len(base_values) < 2:
        return float("nan"), float("nan"), len(base_values)
    sig = paired_significance(np.asarray(base_values), np.asarray(method_values), seed=43, bootstraps=1000)
    return sig.mean_delta, sig.ttest_p, len(base_values)


def _write_report(path: Path, rows: list[dict[str, object]], history: list[dict[str, float]]) -> None:
    methods = sorted({str(row["method"]) for row in rows})
    lines = [
        "# Semantic INR no-leak smoke",
        "",
        "## Protocol",
        "",
        "- 模型是可训练 PyTorch 坐标条件场。",
        "- 推理输入只包含 low_res 和 output_shape。",
        "- 推理不接收 GT mask。",
        "- 推理不接收 degradation_name。",
        "- 不使用 crop_prior 或按 mask 染绿。",
        "",
        "## Training",
        "",
        f"- Steps: {len(history)}",
        f"- Initial reconstruction loss: {history[0]['reconstruction_loss']:.6f}",
        f"- Final reconstruction loss: {history[-1]['reconstruction_loss']:.6f}",
        "",
        "## Method means",
        "",
        "| Method | PSNR | SSIM | Boundary F | Task mIoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        values = {metric: float(np.mean([float(row[metric]) for row in method_rows])) for metric in METRICS}
        lines.append(f"| `{method}` | {values['psnr']:.4f} | {values['ssim']:.4f} | {values['boundary_f']:.4f} | {values['task_miou']:.4f} |")

    lines.extend(["", "## Delta vs bicubic", "", "| Method | Metric | Mean delta | t-test p | n |", "|---|---|---:|---:|---:|"])
    for method in methods:
        if method == "bicubic":
            continue
        for metric in METRICS:
            delta, p_value, n_pairs = _paired_delta(rows, method, metric)
            lines.append(f"| `{method}` | {metric} | {delta:+.6f} | {p_value:.6g} | {n_pairs} |")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 这是小规模 no-leak 神经 INR smoke，不是最终论文主表。",
            "- 若出现正向 delta，只能作为真实方法路线可行信号。",
            "- 后续仍需扩大训练集、pseudo/端到端语义质量审计、公平 baseline 重跑。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_semantic_inr_smoke_experiment(
    output_dir: Path,
    train_samples: list,
    test_samples: list,
    degradation_plan: list[DegradationConfig],
    steps: int = 200,
    seed: int = 7,
    device: Optional[torch.device] = None,
    hidden_channels: int = 48,
    learning_rate: float = 1e-3,
    task_loss_weight: float = 0.0,
    semantic_loss_weight: float = 0.03,
    base_sharpen_strength: float = 0.0,
    structure_residual_scale: float = 0.18,
    texture_residual_scale: float = 0.16,
    semantic_detail_boost: float = 0.0,
    base_consistency_loss_weight: float = 0.0,
) -> SemanticINRSmokeSummary:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cpu")
    model = TinySemanticINR(
        hidden_channels=hidden_channels,
        base_sharpen_strength=base_sharpen_strength,
        structure_residual_scale=structure_residual_scale,
        texture_residual_scale=texture_residual_scale,
        semantic_detail_boost=semantic_detail_boost,
    )
    batches = _build_training_batches(train_samples, degradation_plan, seed=seed)
    history = train_semantic_inr_steps(
        model,
        batches=batches,
        steps=steps,
        learning_rate=learning_rate,
        task_loss_weight=task_loss_weight,
        semantic_loss_weight=semantic_loss_weight,
        base_consistency_loss_weight=base_consistency_loss_weight,
        seed=seed,
        device=device,
    )

    rows: list[dict[str, object]] = []
    for sample_idx, sample in enumerate(test_samples):
        for config_idx, config in enumerate(degradation_plan):
            degraded = degrade_sample(sample, config=config, seed=seed * 2000 + sample_idx * 100 + config_idx)
            restorations = {
                "bicubic": bicubic_restore(degraded.low_res, degraded.gt.shape[:2]),
                "uniform_sharp": uniform_sharp_restore(degraded.low_res, degraded.gt.shape[:2]),
                "semantic_inr_no_leak": restore_with_semantic_inr(model, degraded.low_res, degraded.gt.shape[:2], device=device),
            }
            for method, restored in restorations.items():
                rows.append(
                    {
                        "sample": sample.name,
                        "degradation": degraded.degradation_name,
                        "scale": degraded.scale,
                        "method": method,
                        "psnr": psnr(degraded.gt, restored),
                        "ssim": ssim_score(degraded.gt, restored),
                        "boundary_f": boundary_f_score(degraded.gt, restored),
                        "task_miou": proxy_segmentation_miou(degraded.mask, restored),
                    }
                )

    metrics_csv = output_dir / "metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample", "degradation", "scale", "method", *METRICS])
        writer.writeheader()
        writer.writerows(rows)

    train_history = output_dir / "train_history.csv"
    with train_history.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "total_loss",
                "reconstruction_loss",
                "structure_loss",
                "frequency_loss",
                "semantic_loss",
                "base_consistency_loss",
                "task_loss",
                "learned_task_loss",
            ],
        )
        writer.writeheader()
        for step, item in enumerate(history):
            writer.writerow({"step": step, **item})

    model_path = output_dir / "semantic_inr.pt"
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "hidden_channels": hidden_channels,
            "base_sharpen_strength": base_sharpen_strength,
            "structure_residual_scale": structure_residual_scale,
            "texture_residual_scale": texture_residual_scale,
            "semantic_detail_boost": semantic_detail_boost,
            "base_consistency_loss_weight": base_consistency_loss_weight,
        },
        model_path,
    )
    report = output_dir / "semantic_inr_report.md"
    _write_report(report, rows, history)

    summary_json = output_dir / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "steps": steps,
                "train_batches": len(batches),
                "test_rows": len(rows),
                "initial_reconstruction_loss": history[0]["reconstruction_loss"],
                "final_reconstruction_loss": history[-1]["reconstruction_loss"],
                "task_loss_weight": task_loss_weight,
                "semantic_loss_weight": semantic_loss_weight,
                "base_sharpen_strength": base_sharpen_strength,
                "structure_residual_scale": structure_residual_scale,
                "texture_residual_scale": texture_residual_scale,
                "semantic_detail_boost": semantic_detail_boost,
                "base_consistency_loss_weight": base_consistency_loss_weight,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return SemanticINRSmokeSummary(
        output_dir=output_dir,
        metrics_csv=metrics_csv,
        report=report,
        summary_json=summary_json,
        model_path=model_path,
        train_history=train_history,
    )
