from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np

from .degradation import DegradationConfig, degrade_sample
from .metrics import boundary_f_score, proxy_segmentation_miou, psnr, ssim_score
from .no_leak_restoration import estimate_pseudo_mask_from_low_res, restore_no_leak_methods
from .statistics import paired_significance


METRICS = ("psnr", "ssim", "boundary_f", "task_miou")


@dataclass(frozen=True)
class NoLeakExperimentSummary:
    output_dir: Path
    metrics_csv: Path
    report: Path
    summary_json: Path
    sample_count: int


def _mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    return intersection / union if union else 1.0


def _mean(rows: list[dict[str, object]], method: str, metric: str) -> float:
    values = [float(row[metric]) for row in rows if row["method"] == method]
    return float(np.mean(values)) if values else float("nan")


def _paired_values(rows: list[dict[str, object]], method: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
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
    return np.asarray(base_values), np.asarray(method_values)


def _write_report(path: Path, rows: list[dict[str, object]], sample_count: int) -> None:
    methods = sorted({str(row["method"]) for row in rows})
    lines = [
        "# No-leak smoke 审计",
        "",
        "## 范围",
        "",
        "- 恢复端不接收 GT mask。",
        "- 恢复端不接收 degradation_name。",
        "- 语义频率方法不使用 crop_prior 或绿通道染色。",
        "- GT mask 只用于最终指标评估。",
        "",
        f"- Samples: {sample_count}",
        f"- Rows: {len(rows)}",
        "",
        "## Method means",
        "",
        "| Method | PSNR | SSIM | Boundary F | Task mIoU | Pseudo mask IoU |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        lines.append(
            f"| `{method}` | "
            f"{_mean(rows, method, 'psnr'):.4f} | "
            f"{_mean(rows, method, 'ssim'):.4f} | "
            f"{_mean(rows, method, 'boundary_f'):.4f} | "
            f"{_mean(rows, method, 'task_miou'):.4f} | "
            f"{float(np.mean([float(row['pseudo_mask_iou']) for row in method_rows])):.4f} |"
        )

    lines.extend(["", "## Delta vs bicubic", "", "| Method | Metric | Mean delta | t-test p | n |", "|---|---|---:|---:|---:|"])
    for method in methods:
        if method == "bicubic":
            continue
        for metric in METRICS:
            base, proposed = _paired_values(rows, method, metric)
            if len(base) < 2:
                continue
            sig = paired_significance(base, proposed, seed=37, bootstraps=1000)
            lines.append(f"| `{method}` | {metric} | {sig.mean_delta:+.6f} | {sig.ttest_p:.6g} | {len(base)} |")

    lines.extend(
        [
            "",
            "## 判断",
            "",
            "- 这个报告只用于检查无泄漏协议下旧启发式收益是否还存在。",
            "- 若 task mIoU 明显回落，说明旧结果主要来自 oracle mask/颜色泄漏。",
            "- 该方法仍不是 INR，只是公平协议底座。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_no_leak_dataset_experiment(
    output_dir: Path,
    samples: list,
    degradation_plan: list[DegradationConfig],
    seed: int = 7,
) -> NoLeakExperimentSummary:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "metrics.csv"
    report = output_dir / "no_leak_report.md"
    summary_json = output_dir / "summary.json"

    rows: list[dict[str, object]] = []
    for sample_idx, sample in enumerate(samples):
        for config_idx, config in enumerate(degradation_plan):
            degraded = degrade_sample(sample, config=config, seed=seed * 1000 + sample_idx * 100 + config_idx)
            pseudo_mask = estimate_pseudo_mask_from_low_res(degraded.low_res, degraded.gt.shape[:2])
            restorations = restore_no_leak_methods(degraded.low_res, degraded.gt.shape[:2], pseudo_mask=pseudo_mask)
            pseudo_iou = _mask_iou(pseudo_mask, degraded.mask)
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
                        "pseudo_mask_iou": pseudo_iou,
                    }
                )

    fieldnames = ["sample", "degradation", "scale", "method", *METRICS, "pseudo_mask_iou"]
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "sample_count": len(samples),
        "row_count": len(rows),
        "methods": sorted({row["method"] for row in rows}),
        "mean_pseudo_mask_iou": float(np.mean([float(row["pseudo_mask_iou"]) for row in rows])) if rows else float("nan"),
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(report, rows, sample_count=len(samples))

    return NoLeakExperimentSummary(
        output_dir=output_dir,
        metrics_csv=metrics_csv,
        report=report,
        summary_json=summary_json,
        sample_count=len(samples),
    )
