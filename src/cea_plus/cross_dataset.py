from dataclasses import dataclass
from pathlib import Path
import csv
import json

import numpy as np

from .statistics import paired_significance


METRICS = ("psnr", "ssim", "boundary_f", "task_miou")
METRIC_LABELS = {
    "psnr": "PSNR",
    "ssim": "SSIM",
    "boundary_f": "Boundary F",
    "task_miou": "Task mIoU",
}
DEFAULT_METHODS = (
    "bicubic",
    "uniform_sharp",
    "semantic_frequency",
    "semantic_edge_aware",
    "semantic_boundary_guard",
    "structure_only",
)
DEFAULT_DELTA_METHODS = (
    "semantic_frequency",
    "semantic_edge_aware",
    "semantic_boundary_guard",
    "structure_only",
)


@dataclass(frozen=True)
class CrossDatasetSpec:
    name: str
    metrics_csv: Path
    description: str = ""


@dataclass(frozen=True)
class CrossDatasetReportSummary:
    output_dir: Path
    report: Path
    method_means: Path
    paired_deltas: Path
    summary_json: Path


def _degradation_mode(degradation: str) -> str:
    parts = degradation.split("_", 1)
    return parts[1] if len(parts) == 2 else degradation


def _read_rows(
    path: Path,
    methods: tuple[str, ...],
    include_degradations: tuple[str, ...] = (),
    include_modes: tuple[str, ...] = (),
) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    allowed_degradations = set(include_degradations)
    allowed_modes = {mode.lower() for mode in include_modes}
    return [
        row
        for row in rows
        if row["method"] in methods
        and (not allowed_degradations or row["degradation"] in allowed_degradations)
        and (not allowed_modes or _degradation_mode(row["degradation"]).lower() in allowed_modes)
    ]


def _pair_key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["sample"], row["degradation"], row["scale"]


def _fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def _fmt_delta(value: float, digits: int = 6) -> str:
    return f"{value:+.{digits}f}"


def _fmt_p(value: float) -> str:
    if value == 0.0:
        return "<1e-300"
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.6f}"


def _dataset_stats(
    spec: CrossDatasetSpec,
    methods: tuple[str, ...],
    delta_methods: tuple[str, ...],
    include_degradations: tuple[str, ...] = (),
    include_modes: tuple[str, ...] = (),
) -> dict[str, object]:
    rows = _read_rows(
        spec.metrics_csv,
        methods,
        include_degradations=include_degradations,
        include_modes=include_modes,
    )
    pairs = sorted({_pair_key(row) for row in rows})
    samples = sorted({row["sample"] for row in rows})
    by_method = {method: [row for row in rows if row["method"] == method] for method in methods}
    by_key_method = {(*_pair_key(row), row["method"]): row for row in rows}

    means: dict[str, dict[str, float]] = {}
    for method, method_rows in by_method.items():
        if method_rows:
            means[method] = {
                metric: float(np.mean([float(row[metric]) for row in method_rows]))
                for metric in METRICS
            }

    deltas: dict[str, dict[str, dict[str, float]]] = {}
    for method in delta_methods:
        if method not in means:
            continue
        deltas[method] = {}
        for metric in METRICS:
            base_values: list[float] = []
            method_values: list[float] = []
            for sample, degradation, scale in pairs:
                base = by_key_method.get((sample, degradation, scale, "bicubic"))
                proposed = by_key_method.get((sample, degradation, scale, method))
                if base is not None and proposed is not None:
                    base_values.append(float(base[metric]))
                    method_values.append(float(proposed[metric]))
            if len(base_values) < 2:
                continue
            sig = paired_significance(np.asarray(base_values), np.asarray(method_values), seed=17, bootstraps=3000)
            deltas[method][metric] = {
                "mean_delta": sig.mean_delta,
                "ttest_p": sig.ttest_p,
                "wilcoxon_p": sig.wilcoxon_p,
                "bootstrap_ci_low": sig.bootstrap_ci_low,
                "bootstrap_ci_high": sig.bootstrap_ci_high,
                "n_pairs": len(base_values),
            }

    return {
        "name": spec.name,
        "path": str(spec.metrics_csv),
        "description": spec.description,
        "samples": len(samples),
        "paired_inputs": len(pairs),
        "selected_rows": len(rows),
        "method_means": means,
        "paired_deltas_vs_bicubic": deltas,
    }


def _write_method_means(path: Path, datasets: list[dict[str, object]], methods: tuple[str, ...]) -> None:
    fieldnames = ["dataset_name", "method", "n", *METRICS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in datasets:
            means = dataset["method_means"]
            for method in methods:
                values = means.get(method) if isinstance(means, dict) else None
                if values is None:
                    continue
                writer.writerow(
                    {
                        "dataset_name": dataset["name"],
                        "method": method,
                        "n": dataset["paired_inputs"],
                        **{metric: values[metric] for metric in METRICS},
                    }
                )


def _write_paired_deltas(path: Path, datasets: list[dict[str, object]], delta_methods: tuple[str, ...]) -> None:
    fieldnames = [
        "dataset_name",
        "method",
        "metric",
        "mean_delta",
        "ttest_p",
        "wilcoxon_p",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "n_pairs",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for dataset in datasets:
            deltas = dataset["paired_deltas_vs_bicubic"]
            for method in delta_methods:
                if method not in deltas:
                    continue
                for metric in METRICS:
                    if metric not in deltas[method]:
                        continue
                    item = deltas[method][metric]
                    writer.writerow(
                        {
                            "dataset_name": dataset["name"],
                            "method": method,
                            "metric": metric,
                            **item,
                        }
                    )


def _write_report(path: Path, datasets: list[dict[str, object]], methods: tuple[str, ...], delta_methods: tuple[str, ...]) -> None:
    lines = [
        "# 跨数据集泛化报告",
        "",
        "## 范围",
        "",
        "- 统计：逐 sample/degradation/scale 配对，对 bicubic 做 paired t-test、Wilcoxon、bootstrap 95% CI。",
        "- 注意：报告支持农业 UAV 语义任务导向恢复，不支持通用恢复 SOTA 表述。",
        "",
        "## 数据集与行数",
        "",
        "| Dataset | Source CSV | Samples | Paired inputs | Selected rows |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in datasets:
        lines.append(
            f"| {dataset['name']} | `{dataset['path']}` | {dataset['samples']} | "
            f"{dataset['paired_inputs']} | {dataset['selected_rows']} |"
        )

    for dataset in datasets:
        lines.extend(["", f"## {dataset['name']}", ""])
        if dataset["description"]:
            lines.extend([str(dataset["description"]), ""])
        lines.extend(["| Method | PSNR | SSIM | Boundary F | Task mIoU |", "|---|---:|---:|---:|---:|"])
        means = dataset["method_means"]
        for method in methods:
            values = means.get(method) if isinstance(means, dict) else None
            if values is None:
                continue
            lines.append(
                f"| `{method}` | {_fmt(values['psnr'])} | {_fmt(values['ssim'])} | "
                f"{_fmt(values['boundary_f'])} | {_fmt(values['task_miou'])} |"
            )

        lines.extend(["", "### Delta vs bicubic", "", "| Method | Metric | Mean delta | t-test p | Bootstrap 95% CI |", "|---|---|---:|---:|---:|"])
        deltas = dataset["paired_deltas_vs_bicubic"]
        for method in delta_methods:
            if method not in deltas:
                continue
            for metric in METRICS:
                if metric not in deltas[method]:
                    continue
                item = deltas[method][metric]
                lines.append(
                    f"| `{method}` | {METRIC_LABELS[metric]} | {_fmt_delta(item['mean_delta'])} | "
                    f"{_fmt_p(item['ttest_p'])} | "
                    f"[{_fmt_delta(item['bootstrap_ci_low'])}, {_fmt_delta(item['bootstrap_ci_high'])}] |"
                )

    lines.extend(["", "## 论文写法建议", ""])
    lines.append("- 可写：方法在多个农业遥感数据集上稳定提升任务相关恢复效果，尤其 task mIoU。")
    lines.append("- 可写：边界指标显示任务导向恢复与边界/视觉保真存在 trade-off。")
    lines.append("- 不应写：通用图像恢复 SOTA、Boundary F 全面超越 SOTA、所有退化场景全指标胜出。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cross_dataset_report(
    dataset_specs: list[CrossDatasetSpec],
    output_dir: Path,
    methods: tuple[str, ...] = DEFAULT_METHODS,
    delta_methods: tuple[str, ...] = DEFAULT_DELTA_METHODS,
    include_degradations: tuple[str, ...] = (),
    include_modes: tuple[str, ...] = (),
) -> CrossDatasetReportSummary:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [
        _dataset_stats(
            spec,
            methods=methods,
            delta_methods=delta_methods,
            include_degradations=include_degradations,
            include_modes=include_modes,
        )
        for spec in dataset_specs
    ]

    method_means = output_dir / "method_means.csv"
    paired_deltas = output_dir / "paired_deltas_vs_bicubic.csv"
    summary_json = output_dir / "summary.json"
    report = output_dir / "cross_dataset_report.md"

    _write_method_means(method_means, datasets, methods)
    _write_paired_deltas(paired_deltas, datasets, delta_methods)
    summary_json.write_text(json.dumps({"datasets": datasets}, indent=2), encoding="utf-8")
    _write_report(report, datasets, methods, delta_methods)

    return CrossDatasetReportSummary(
        output_dir=output_dir,
        report=report,
        method_means=method_means,
        paired_deltas=paired_deltas,
        summary_json=summary_json,
    )
