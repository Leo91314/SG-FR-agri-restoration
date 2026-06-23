from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
from PIL import Image, ImageDraw

from .degradation import DegradationConfig, build_degradation_plan, degrade_sample
from .metrics import boundary_f_score, proxy_segmentation_miou, psnr, ssim_score
from .restoration import bicubic_restore, restore_all_methods, semantic_frequency_components
from .statistics import SignificanceResult, paired_significance
from .synthesis import make_synthetic_agri_sample


@dataclass(frozen=True)
class ExperimentSummary:
    sample_count: int
    best_method: str
    significant_metrics: tuple[str, ...]


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.uint8(np.clip(image, 0.0, 1.0) * 255.0)


def _save_case(path: Path, gt: np.ndarray, low_res: np.ndarray, bicubic: np.ndarray, uniform: np.ndarray, semantic: np.ndarray, alpha: np.ndarray, mask: np.ndarray) -> None:
    panels = [
        ("GT", gt),
        ("LQ", bicubic_restore(low_res, gt.shape[:2])),
        ("Bicubic", bicubic),
        ("Uniform", uniform),
        ("Semantic", semantic),
        ("Alpha", np.repeat(alpha[..., None].astype(np.float32), 3, axis=2)),
        ("Mask", np.repeat(mask[..., None].astype(np.float32), 3, axis=2)),
    ]
    h, w = gt.shape[:2]
    canvas = Image.new("RGB", (w * len(panels), h + 18), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, image) in enumerate(panels):
        canvas.paste(Image.fromarray(_to_uint8(image)), (idx * w, 18))
        draw.text((idx * w + 4, 3), label, fill=(0, 0, 0))
    canvas.save(path)


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["sample", "degradation", "scale", "method", "psnr", "ssim", "boundary_f", "task_miou"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_metrics(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = ["sample", "degradation", "scale", "method", "psnr", "ssim", "boundary_f", "task_miou"]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _completed_restoration_pairs(rows: list[dict[str, object]]) -> set[tuple[str, str, str]]:
    expected = {
        "bicubic",
        "uniform_sharp",
        "semantic_frequency",
        "semantic_edge_aware",
        "semantic_boundary_guard",
        "semantic_no_mod",
        "semantic_fixed_alpha",
        "structure_only",
    }
    methods_by_pair: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        key = (str(row["sample"]), str(row["degradation"]), str(row["scale"]))
        methods_by_pair.setdefault(key, set()).add(str(row["method"]))
    return {key for key, methods in methods_by_pair.items() if expected.issubset(methods)}


def _metric_table(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    methods = sorted({str(row["method"]) for row in rows})
    metrics = ["psnr", "ssim", "boundary_f", "task_miou"]
    table: dict[str, dict[str, float]] = {}
    for method in methods:
        table[method] = {}
        method_rows = [row for row in rows if row["method"] == method]
        for metric in metrics:
            table[method][metric] = float(np.mean([float(row[metric]) for row in method_rows]))
    return table


def _metric_table_by_degradation(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, float]]:
    groups = sorted({(str(row["degradation"]), str(row["method"])) for row in rows})
    metrics = ["psnr", "ssim", "boundary_f", "task_miou"]
    table: dict[tuple[str, str], dict[str, float]] = {}
    for degradation, method in groups:
        group_rows = [row for row in rows if row["degradation"] == degradation and row["method"] == method]
        table[(degradation, method)] = {}
        for metric in metrics:
            table[(degradation, method)][metric] = float(np.mean([float(row[metric]) for row in group_rows]))
    return table


def _paired_values(rows: list[dict[str, object]], method: str, metric: str) -> np.ndarray:
    method_rows = sorted([row for row in rows if row["method"] == method], key=lambda item: (str(item["sample"]), str(item["degradation"])))
    return np.array([float(row[metric]) for row in method_rows], dtype=np.float64)


def _write_report(path: Path, rows: list[dict[str, object]], stats_by_metric: dict[str, SignificanceResult]) -> ExperimentSummary:
    table = _metric_table(rows)
    primary_methods = [method for method in ("bicubic", "uniform_sharp", "semantic_frequency") if method in table]
    best_method = max(primary_methods or list(table), key=lambda method: table[method]["psnr"])
    significant = tuple(metric for metric, result in stats_by_metric.items() if result.mean_delta > 0.0 and result.ttest_p < 0.05 and result.bootstrap_ci_low > 0.0)

    lines = [
        "# CEA_plus 显著性报告",
        "",
        "## 实验说明",
        "",
        "本报告基于当前输入样本，验证语义频率调制代理方法相对普通恢复 baseline 的差异。若输入为真实数据集，本报告反映真实数据上的阶段性实验结果；若输入为合成样本，本报告用于工程闭环和机制预验证。",
        "",
        "## 平均指标",
        "",
        "| Method | PSNR | SSIM | Boundary F | Task mIoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in sorted(table):
        lines.append(
            f"| {method} | {table[method]['psnr']:.4f} | {table[method]['ssim']:.4f} | "
            f"{table[method]['boundary_f']:.4f} | {table[method]['task_miou']:.4f} |"
        )

    lines.extend(["", "## Semantic Frequency vs Bicubic", "", "| Metric | Mean Delta | t-test p | Wilcoxon p | Bootstrap 95% CI |", "|---|---:|---:|---:|---:|"])
    for metric, result in stats_by_metric.items():
        lines.append(
            f"| {metric} | {result.mean_delta:.6f} | {result.ttest_p:.6g} | {result.wilcoxon_p:.6g} | "
            f"[{result.bootstrap_ci_low:.6f}, {result.bootstrap_ci_high:.6f}] |"
        )
    lines.extend(["", "## 显著性结论", ""])
    if significant:
        joined = ", ".join(significant)
        lines.append(f"- semantic_frequency 在 {joined} 上相对 bicubic 取得正向且统计显著提升。")
    else:
        lines.append("- 当前 smoke 配置未得到满足阈值的统计显著提升。")
    lines.extend(["", "## 分退化统计", "", "| Degradation | Method | PSNR | SSIM | Boundary F | Task mIoU |", "|---|---|---:|---:|---:|---:|"])
    grouped = _metric_table_by_degradation(rows)
    for (degradation, method), values in grouped.items():
        lines.append(
            f"| {degradation} | {method} | {values['psnr']:.4f} | {values['ssim']:.4f} | "
            f"{values['boundary_f']:.4f} | {values['task_miou']:.4f} |"
        )
    lines.extend(["", "## 消融", ""])
    ablations = [method for method in ("semantic_no_mod", "semantic_fixed_alpha", "structure_only") if method in table]
    if ablations:
        lines.append("| Variant | PSNR | SSIM | Boundary F | Task mIoU |")
        lines.append("|---|---:|---:|---:|---:|")
        for method in ablations:
            lines.append(
                f"| {method} | {table[method]['psnr']:.4f} | {table[method]['ssim']:.4f} | "
                f"{table[method]['boundary_f']:.4f} | {table[method]['task_miou']:.4f} |"
            )
    else:
        lines.append("- 本次运行未包含消融方法。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ExperimentSummary(sample_count=len(_paired_values(rows, "bicubic", "psnr")), best_method=best_method, significant_metrics=significant)


def run_dataset_experiment(
    output_dir: Path,
    samples: list,
    degradation_plan: list[DegradationConfig],
    seed: int = 7,
    resume: bool = False,
) -> ExperimentSummary:
    output_dir = Path(output_dir)
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"

    rows: list[dict[str, object]] = _read_metrics(metrics_path) if resume else []
    completed_pairs = _completed_restoration_pairs(rows) if resume else set()
    case_count = 0
    for idx, sample in enumerate(samples):
        for config_idx, config in enumerate(degradation_plan):
            pair_key = (sample.name, config.name, str(config.scale))
            if pair_key in completed_pairs:
                continue
            degraded = degrade_sample(sample, config=config, seed=seed * 1000 + idx * 100 + config_idx)
            restorations = restore_all_methods(degraded.low_res, degraded.mask, degraded.gt.shape[:2], degradation_name=degraded.degradation_name)
            semantic_components = semantic_frequency_components(degraded.low_res, degraded.mask, degraded.gt.shape[:2], degradation_name=degraded.degradation_name)
            new_rows: list[dict[str, object]] = []
            for method, restored in restorations.items():
                new_rows.append(
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
            rows.extend(new_rows)
            if resume:
                _append_metrics(metrics_path, new_rows)
                completed_pairs.add(pair_key)
            if case_count < 4:
                _save_case(
                    cases_dir / f"case_{case_count:03d}_{sample.name}_{config.name}.png",
                    degraded.gt,
                    degraded.low_res,
                    restorations["bicubic"],
                    restorations["uniform_sharp"],
                    restorations["semantic_frequency"],
                    semantic_components.alpha,
                    degraded.mask,
                )
                if case_count == 0:
                    _save_case(
                        cases_dir / "case_000.png",
                        degraded.gt,
                        degraded.low_res,
                        restorations["bicubic"],
                        restorations["uniform_sharp"],
                        restorations["semantic_frequency"],
                        semantic_components.alpha,
                        degraded.mask,
                    )
                case_count += 1

    if not resume:
        _write_metrics(metrics_path, rows)
    stats_by_metric = {
        metric: paired_significance(_paired_values(rows, "bicubic", metric), _paired_values(rows, "semantic_frequency", metric), seed=seed + offset)
        for offset, metric in enumerate(["psnr", "ssim", "boundary_f", "task_miou"])
    }
    return _write_report(output_dir / "significance_report.md", rows, stats_by_metric)


def run_experiment(output_dir: Path, samples: int = 24, seed: int = 7, image_size: int = 128) -> ExperimentSummary:
    sample_list = [make_synthetic_agri_sample(seed=seed + idx, size=image_size).with_name(str(idx)) for idx in range(samples)]
    return run_dataset_experiment(
        output_dir=output_dir,
        samples=sample_list,
        degradation_plan=build_degradation_plan(scales=(4,), modes=("mixed",)),
        seed=seed,
    )
