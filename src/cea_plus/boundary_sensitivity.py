from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np

from .degradation import DegradationConfig, degrade_sample
from .metrics import boundary_f_score, proxy_segmentation_miou
from .restoration import (
    bicubic_restore,
    semantic_boundary_guard_restore,
    semantic_edge_aware_restore,
    semantic_frequency_components,
    semantic_frequency_restore,
    uniform_sharp_restore,
)
from .statistics import paired_significance
from .synthesis import AgriSample


@dataclass(frozen=True)
class BoundarySensitivityDataset:
    name: str
    samples: list[AgriSample]
    description: str = ""


@dataclass(frozen=True)
class BoundarySensitivitySummary:
    output_dir: Path
    metrics_csv: Path
    report: Path
    summary_json: Path
    summary_rows: list[dict[str, object]]


FIELDNAMES = [
    "dataset",
    "sample",
    "degradation",
    "scale",
    "method",
    "edge_quantile",
    "boundary_f",
    "task_miou",
]


def _fmt_delta(value: float) -> str:
    return f"{value:+.6f}"


def _fmt_p(value: float) -> str:
    if value == 0.0:
        return "<1e-300"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.6f}"


def _pair_key(row: dict[str, object]) -> tuple[str, str, str]:
    return str(row["sample"]), str(row["degradation"]), str(row["scale"])


def _summarize_rows(rows: list[dict[str, object]], methods: tuple[str, ...]) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    datasets = sorted({str(row["dataset"]) for row in rows})
    quantiles = sorted({float(row["edge_quantile"]) for row in rows})
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for quantile in quantiles:
            quantile_rows = [row for row in dataset_rows if float(row["edge_quantile"]) == quantile]
            pairs = sorted({_pair_key(row) for row in quantile_rows})
            by_key_method = {(*_pair_key(row), str(row["method"])): row for row in quantile_rows}
            for method in methods:
                if method == "bicubic":
                    continue
                base_values: list[float] = []
                method_values: list[float] = []
                base_task: list[float] = []
                method_task: list[float] = []
                for sample, degradation, scale in pairs:
                    base = by_key_method.get((sample, degradation, scale, "bicubic"))
                    proposed = by_key_method.get((sample, degradation, scale, method))
                    if base is None or proposed is None:
                        continue
                    base_values.append(float(base["boundary_f"]))
                    method_values.append(float(proposed["boundary_f"]))
                    base_task.append(float(base["task_miou"]))
                    method_task.append(float(proposed["task_miou"]))
                if len(base_values) < 2:
                    continue
                sig = paired_significance(np.asarray(base_values), np.asarray(method_values), seed=29, bootstraps=2000)
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "edge_quantile": quantile,
                        "method": method,
                        "boundary_f_delta": sig.mean_delta,
                        "boundary_f_ttest_p": sig.ttest_p,
                        "boundary_f_ci_low": sig.bootstrap_ci_low,
                        "boundary_f_ci_high": sig.bootstrap_ci_high,
                        "task_miou_delta": float(np.mean(np.asarray(method_task) - np.asarray(base_task))),
                        "n_pairs": len(base_values),
                    }
                )
    return summary_rows


def _restore_requested_method(
    method: str,
    low_res: np.ndarray,
    mask: np.ndarray,
    output_shape: tuple[int, int],
    degradation_name: str,
) -> np.ndarray:
    if method == "bicubic":
        return bicubic_restore(low_res, output_shape)
    if method == "uniform_sharp":
        return uniform_sharp_restore(low_res, output_shape)
    if method == "semantic_frequency":
        return semantic_frequency_restore(low_res, mask, output_shape, degradation_name=degradation_name)
    if method == "semantic_edge_aware":
        return semantic_edge_aware_restore(low_res, mask, output_shape, degradation_name=degradation_name)
    if method == "semantic_boundary_guard":
        return semantic_boundary_guard_restore(low_res, mask, output_shape, degradation_name=degradation_name)
    if method == "semantic_no_mod":
        return semantic_frequency_components(
            low_res,
            mask,
            output_shape,
            modulation="none",
            degradation_name=degradation_name,
        ).restored
    if method == "semantic_fixed_alpha":
        return semantic_frequency_components(
            low_res,
            mask,
            output_shape,
            modulation="fixed",
            degradation_name=degradation_name,
        ).restored
    if method == "structure_only":
        return semantic_frequency_components(
            low_res,
            mask,
            output_shape,
            modulation="structure_only",
            degradation_name=degradation_name,
        ).restored
    raise ValueError(f"unknown restoration method: {method}")


def _write_report(path: Path, datasets: list[BoundarySensitivityDataset], summary_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Boundary F 阈值敏感性报告",
        "",
        "## 范围",
        "",
        "- 指标：Sobel edge map + quantile threshold + tolerance boundary F。",
        "- 目的：检查 Boundary F 结论是否依赖单一 quantile 取值。",
        "",
        "## 数据集",
        "",
        "| Dataset | Samples | Description |",
        "|---|---:|---|",
    ]
    for dataset in datasets:
        lines.append(f"| {dataset.name} | {len(dataset.samples)} | {dataset.description} |")

    lines.extend(
        [
            "",
            "## Delta vs bicubic",
            "",
            "| Dataset | Quantile | Method | Boundary F delta | t-test p | Bootstrap 95% CI | Task mIoU delta | n |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['dataset']} | {float(row['edge_quantile']):.2f} | `{row['method']}` | "
            f"{_fmt_delta(float(row['boundary_f_delta']))} | {_fmt_p(float(row['boundary_f_ttest_p']))} | "
            f"[{_fmt_delta(float(row['boundary_f_ci_low']))}, {_fmt_delta(float(row['boundary_f_ci_high']))}] | "
            f"{_fmt_delta(float(row['task_miou_delta']))} | {row['n_pairs']} |"
        )

    lines.extend(
        [
            "",
            "## 写法建议",
            "",
            "- 如果不同 quantile 下 Boundary F delta 符号一致，可写结论对阈值不敏感。",
            "- 如果符号随 quantile 改变，只能写 Boundary F 结论依赖阈值，不能作为稳定主结论。",
            "- 小样本敏感性分析只能作为审稿风险检查，不替代完整 split 主表。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_boundary_quantile_sensitivity(
    output_dir: Path,
    datasets: list[BoundarySensitivityDataset],
    degradation_plan: list[DegradationConfig],
    quantiles: tuple[float, ...] = (0.78, 0.82, 0.86),
    methods: tuple[str, ...] = ("bicubic", "semantic_frequency", "semantic_boundary_guard"),
    seed: int = 7,
) -> BoundarySensitivitySummary:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = output_dir / "metrics.csv"
    report = output_dir / "boundary_quantile_sensitivity_report.md"
    summary_json = output_dir / "summary.json"

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for sample_idx, sample in enumerate(dataset.samples):
            for config_idx, config in enumerate(degradation_plan):
                degraded = degrade_sample(sample, config=config, seed=seed * 1000 + sample_idx * 100 + config_idx)
                for method in methods:
                    restored = _restore_requested_method(
                        method,
                        degraded.low_res,
                        degraded.mask,
                        degraded.gt.shape[:2],
                        degradation_name=degraded.degradation_name,
                    )
                    task_miou = proxy_segmentation_miou(degraded.mask, restored)
                    for quantile in quantiles:
                        rows.append(
                            {
                                "dataset": dataset.name,
                                "sample": sample.name,
                                "degradation": degraded.degradation_name,
                                "scale": degraded.scale,
                                "method": method,
                                "edge_quantile": float(quantile),
                                "boundary_f": boundary_f_score(degraded.gt, restored, edge_quantile=float(quantile)),
                                "task_miou": task_miou,
                            }
                        )

    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = _summarize_rows(rows, methods=methods)
    summary_json.write_text(
        json.dumps({"summary_rows": summary_rows, "metrics_rows": len(rows)}, indent=2),
        encoding="utf-8",
    )
    _write_report(report, datasets=datasets, summary_rows=summary_rows)

    return BoundarySensitivitySummary(
        output_dir=output_dir,
        metrics_csv=metrics_csv,
        report=report,
        summary_json=summary_json,
        summary_rows=summary_rows,
    )
