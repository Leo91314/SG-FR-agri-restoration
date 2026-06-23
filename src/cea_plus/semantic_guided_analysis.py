from dataclasses import dataclass
from pathlib import Path
import csv
import json
from typing import Optional

import numpy as np
import torch

from .deep_downstream import train_torch_segmenter
from .degradation import DegradationConfig, degrade_sample
from .restoration import bicubic_restore


@dataclass(frozen=True)
class GuideQualitySummary:
    output_dir: Path
    metrics_csv: Path
    summary_json: Path
    report: Path
    guide_model_path: Path
    sample_pairs: int


def _mask_iou(pred: np.ndarray, target: np.ndarray) -> float:
    pred_bool = pred.astype(bool)
    target_bool = target.astype(bool)
    intersection = int(np.logical_and(pred_bool, target_bool).sum())
    union = int(np.logical_or(pred_bool, target_bool).sum())
    if union == 0:
        return 1.0
    return float(intersection / union)


def _safe_region_mean(values: np.ndarray, region: np.ndarray) -> float:
    region_bool = region.astype(bool)
    if not region_bool.any():
        return float("nan")
    return float(np.mean(values[region_bool]))


def _finite_mean(values: list[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(value)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "sample",
        "degradation",
        "scale",
        "pseudo_mask_iou",
        "mean_confidence",
        "pseudo_positive_rate",
        "gt_positive_rate",
        "foreground_probability_mean",
        "background_probability_mean",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary_by_degradation(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    names = sorted({str(row["degradation"]) for row in rows})
    result: dict[str, dict[str, float]] = {}
    for name in names:
        subset = [row for row in rows if str(row["degradation"]) == name]
        result[name] = {
            "sample_pairs": float(len(subset)),
            "mean_pseudo_mask_iou": _finite_mean([float(row["pseudo_mask_iou"]) for row in subset]),
            "mean_confidence": _finite_mean([float(row["mean_confidence"]) for row in subset]),
            "mean_pseudo_positive_rate": _finite_mean([float(row["pseudo_positive_rate"]) for row in subset]),
            "mean_gt_positive_rate": _finite_mean([float(row["gt_positive_rate"]) for row in subset]),
        }
    return result


def _write_report(path: Path, summary: dict[str, object], by_degradation: dict[str, dict[str, float]]) -> None:
    lines = [
        "# Guide quality analysis",
        "",
        "## Protocol",
        "",
        "- Guide training source: clean train images only.",
        "- Evaluation input: bicubic-restored degraded image.",
        "- GT masks are evaluation only; they are not used by restoration or guide prediction.",
        "- 该报告只诊断 pseudo-mask 质量，不作为恢复方法的训练或推理输入。",
        "",
        "## Overall",
        "",
        f"- Sample pairs: `{summary['sample_pairs']}`",
        f"- Mean pseudo-mask IoU: `{summary['mean_pseudo_mask_iou']:.6f}`",
        f"- Mean confidence: `{summary['mean_confidence']:.6f}`",
        f"- Mean pseudo positive rate: `{summary['mean_pseudo_positive_rate']:.6f}`",
        f"- Mean GT positive rate: `{summary['mean_gt_positive_rate']:.6f}`",
        "",
        "## By Degradation",
        "",
        "| Degradation | Pairs | Pseudo IoU | Confidence | Pseudo positive | GT positive |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in by_degradation.items():
        lines.append(
            f"| `{name}` | {int(stats['sample_pairs'])} | {stats['mean_pseudo_mask_iou']:.6f} | "
            f"{stats['mean_confidence']:.6f} | {stats['mean_pseudo_positive_rate']:.6f} | "
            f"{stats['mean_gt_positive_rate']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "- 低 pseudo-mask IoU 或明显前景率漂移，说明语义引导恢复的上限受 guide 质量限制。",
            "- 该分析可以解释结果强弱，但不能单独证明恢复方法优于 baseline。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_guide_quality_analysis(
    output_dir: Path,
    train_samples: list,
    test_samples: list,
    degradation_plan: list[DegradationConfig],
    guide_architecture: str = "tiny",
    guide_steps: int = 40,
    guide_crop_size: int = 128,
    eval_size: int = 256,
    seed: int = 7,
    device: Optional[torch.device] = None,
) -> GuideQualitySummary:
    if not train_samples:
        raise ValueError("at least one training sample is required")
    if not test_samples:
        raise ValueError("at least one test sample is required")
    if not degradation_plan:
        raise ValueError("at least one degradation config is required")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cpu")

    guide = train_torch_segmenter(
        train_samples=train_samples,
        architecture=guide_architecture,
        steps=guide_steps,
        crop_size=guide_crop_size,
        input_size=eval_size,
        seed=seed,
        device=device,
    )
    guide_model_path = output_dir / f"frozen_clean_guide_{guide_architecture}.pt"
    torch.save(
        {
            "architecture": guide_architecture,
            "state_dict": guide.model.state_dict(),
            "input_size": guide.input_size,
            "train_source": "clean_train_only",
            "role": "pseudo_mask_quality_analysis",
        },
        guide_model_path,
    )

    rows: list[dict[str, object]] = []
    for sample_idx, sample in enumerate(test_samples):
        for config_idx, config in enumerate(degradation_plan):
            degraded = degrade_sample(sample, config=config, seed=seed * 4000 + sample_idx * 100 + config_idx)
            restored_input = bicubic_restore(degraded.low_res, degraded.gt.shape[:2])
            probability = guide.predict_probability(restored_input).astype(np.float32).clip(0.0, 1.0)
            pseudo_mask = probability >= 0.5
            gt_mask = degraded.mask.astype(bool)
            confidence = np.abs(probability - 0.5) * 2.0
            rows.append(
                {
                    "sample": sample.name,
                    "degradation": config.name,
                    "scale": config.scale,
                    "pseudo_mask_iou": _mask_iou(pseudo_mask, gt_mask),
                    "mean_confidence": float(np.mean(confidence)),
                    "pseudo_positive_rate": float(np.mean(pseudo_mask)),
                    "gt_positive_rate": float(np.mean(gt_mask)),
                    "foreground_probability_mean": _safe_region_mean(probability, gt_mask),
                    "background_probability_mean": _safe_region_mean(probability, ~gt_mask),
                }
            )

    metrics_csv = output_dir / "guide_quality_metrics.csv"
    _write_metrics(metrics_csv, rows)

    by_degradation = _summary_by_degradation(rows)
    summary_payload: dict[str, object] = {
        "guide_architecture": guide_architecture,
        "guide_train_source": "clean_train_only",
        "evaluation_input": "bicubic_restored_degraded_image",
        "gt_mask_usage": "evaluation_only",
        "sample_pairs": len(rows),
        "mean_pseudo_mask_iou": _finite_mean([float(row["pseudo_mask_iou"]) for row in rows]),
        "mean_confidence": _finite_mean([float(row["mean_confidence"]) for row in rows]),
        "mean_pseudo_positive_rate": _finite_mean([float(row["pseudo_positive_rate"]) for row in rows]),
        "mean_gt_positive_rate": _finite_mean([float(row["gt_positive_rate"]) for row in rows]),
        "by_degradation": by_degradation,
    }
    summary_json = output_dir / "guide_quality_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True), encoding="utf-8")

    report = output_dir / "guide_quality_report.md"
    _write_report(report, summary_payload, by_degradation)
    return GuideQualitySummary(
        output_dir=output_dir,
        metrics_csv=metrics_csv,
        summary_json=summary_json,
        report=report,
        guide_model_path=guide_model_path,
        sample_pairs=len(rows),
    )
