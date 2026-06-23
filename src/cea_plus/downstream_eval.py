from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Optional

import numpy as np

from .degradation import DegradationConfig, degrade_sample
from .downstream import FrozenPixelSegmenter, evaluate_frozen_segmenter, save_segmenter, train_pixel_segmenter
from .restoration import restore_all_methods
from .statistics import SignificanceResult, paired_significance


@dataclass(frozen=True)
class DownstreamSummary:
    sample_count: int
    best_method: str
    significant_metrics: tuple[str, ...]
    model_path: Path


def _write_downstream_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["sample", "degradation", "scale", "method", "frozen_miou"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _method_table(rows: list[dict[str, object]]) -> dict[str, float]:
    methods = sorted({str(row["method"]) for row in rows})
    return {
        method: float(np.mean([float(row["frozen_miou"]) for row in rows if row["method"] == method]))
        for method in methods
    }


def _paired_values(rows: list[dict[str, object]], method: str) -> np.ndarray:
    method_rows = sorted(
        [row for row in rows if row["method"] == method],
        key=lambda item: (str(item["sample"]), str(item["degradation"])),
    )
    return np.array([float(row["frozen_miou"]) for row in method_rows], dtype=np.float64)


def _write_downstream_report(
    path: Path,
    rows: list[dict[str, object]],
    stats: SignificanceResult,
    model_path: Path,
) -> DownstreamSummary:
    table = _method_table(rows)
    primary_methods = [method for method in ("bicubic", "uniform_sharp", "semantic_frequency") if method in table]
    best_method = max(primary_methods or list(table), key=lambda method: table[method])
    significant = ("frozen_miou",) if stats.mean_delta > 0.0 and stats.ttest_p < 0.05 and stats.bootstrap_ci_low > 0.0 else ()

    lines = [
        "# 冻结下游分割显著性报告",
        "",
        "## 实验说明",
        "",
        "本报告训练一次轻量像素分割器并冻结参数，再评估不同恢复方法对下游二类植被分割 mIoU 的影响。",
        "",
        f"- 冻结模型：`{model_path}`",
        f"- 配对样本数：{len(_paired_values(rows, 'bicubic'))}",
        "",
        "## 平均指标",
        "",
        "| Method | Frozen mIoU |",
        "|---|---:|",
    ]
    for method, value in table.items():
        lines.append(f"| {method} | {value:.4f} |")

    lines.extend(
        [
            "",
            "## Semantic Frequency vs Bicubic",
            "",
            "| Metric | Mean Delta | t-test p | Wilcoxon p | Bootstrap 95% CI |",
            "|---|---:|---:|---:|---:|",
            f"| frozen_miou | {stats.mean_delta:.6f} | {stats.ttest_p:.6g} | {stats.wilcoxon_p:.6g} | "
            f"[{stats.bootstrap_ci_low:.6f}, {stats.bootstrap_ci_high:.6f}] |",
            "",
            "## 显著性结论",
            "",
        ]
    )
    if significant:
        lines.append("- semantic_frequency 在冻结下游分割 mIoU 上相对 bicubic 取得正向且统计显著提升。")
    else:
        lines.append("- 当前配置未得到冻结下游分割 mIoU 的正向显著提升。")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return DownstreamSummary(
        sample_count=len(_paired_values(rows, "bicubic")),
        best_method=best_method,
        significant_metrics=significant,
        model_path=model_path,
    )


def run_downstream_experiment(
    output_dir: Path,
    train_samples: list,
    test_samples: list,
    degradation_plan: list[DegradationConfig],
    max_pixels_per_sample: int = 2000,
    seed: int = 7,
    segmenter: Optional[FrozenPixelSegmenter] = None,
) -> DownstreamSummary:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "frozen_pixel_segmenter.joblib"
    if segmenter is None:
        segmenter = train_pixel_segmenter(train_samples, max_pixels_per_sample=max_pixels_per_sample, seed=seed)
    save_segmenter(segmenter, model_path)

    rows: list[dict[str, object]] = []
    for sample_idx, sample in enumerate(test_samples):
        for config_idx, config in enumerate(degradation_plan):
            degraded = degrade_sample(sample, config=config, seed=seed * 1000 + sample_idx * 100 + config_idx)
            restorations = restore_all_methods(
                degraded.low_res,
                degraded.mask,
                degraded.gt.shape[:2],
                degradation_name=degraded.degradation_name,
            )
            for method, restored in restorations.items():
                rows.append(
                    {
                        "sample": sample.name,
                        "degradation": degraded.degradation_name,
                        "scale": degraded.scale,
                        "method": method,
                        "frozen_miou": evaluate_frozen_segmenter(segmenter, restored, degraded.mask),
                    }
                )

    _write_downstream_metrics(output_dir / "downstream_metrics.csv", rows)
    stats = paired_significance(
        _paired_values(rows, "bicubic"),
        _paired_values(rows, "semantic_frequency"),
        seed=seed,
    )
    return _write_downstream_report(output_dir / "downstream_report.md", rows, stats, model_path)
