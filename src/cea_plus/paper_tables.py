from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class PaperTableSummary:
    output_dir: Path
    main_table: Path
    ablation_table: Path
    generalization_table: Path
    efficiency_table: Path
    deep_baseline_table: Path
    summary_report: Path


RESTORATION_METRICS = ("psnr", "ssim", "boundary_f", "task_miou")
MAIN_METHODS = ("bicubic", "uniform_sharp", "semantic_frequency")
ABLATION_METHODS = (
    "semantic_frequency",
    "semantic_edge_aware",
    "semantic_boundary_guard",
    "semantic_no_mod",
    "semantic_fixed_alpha",
    "structure_only",
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mean_by_method(rows: list[dict[str, str]], metric: str) -> dict[str, float]:
    methods = sorted({row["method"] for row in rows})
    return {
        method: float(np.mean([float(row[metric]) for row in rows if row["method"] == method]))
        for method in methods
    }


def _metric_tables(restoration_rows: list[dict[str, str]], downstream_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    methods = sorted({row["method"] for row in restoration_rows} | {row["method"] for row in downstream_rows})
    table = {method: {} for method in methods}
    for metric in RESTORATION_METRICS:
        values = _mean_by_method(restoration_rows, metric)
        for method, value in values.items():
            table[method][metric] = value
    frozen = _mean_by_method(downstream_rows, "frozen_miou")
    for method, value in frozen.items():
        table[method]["frozen_miou"] = value
    return table


def _write_main_table(path: Path, table: dict[str, dict[str, float]]) -> None:
    lines = [
        "# 论文主表",
        "",
        "| Method | PSNR | SSIM | Boundary F | Task mIoU | Frozen mIoU |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in MAIN_METHODS:
        values = table.get(method, {})
        lines.append(
            f"| {method} | {values.get('psnr', float('nan')):.4f} | {values.get('ssim', float('nan')):.4f} | "
            f"{values.get('boundary_f', float('nan')):.4f} | {values.get('task_miou', float('nan')):.4f} | "
            f"{values.get('frozen_miou', float('nan')):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ablation_table(path: Path, table: dict[str, dict[str, float]]) -> None:
    lines = [
        "# 消融表",
        "",
        "| Variant | PSNR | SSIM | Boundary F | Task mIoU | Frozen mIoU |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ABLATION_METHODS:
        values = table.get(method, {})
        lines.append(
            f"| {method} | {values.get('psnr', float('nan')):.4f} | {values.get('ssim', float('nan')):.4f} | "
            f"{values.get('boundary_f', float('nan')):.4f} | {values.get('task_miou', float('nan')):.4f} | "
            f"{values.get('frozen_miou', float('nan')):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_generalization_table(path: Path, restoration_rows: list[dict[str, str]], downstream_rows: list[dict[str, str]]) -> None:
    downstream_lookup: dict[tuple[str, str], list[float]] = {}
    for row in downstream_rows:
        downstream_lookup.setdefault((row["degradation"], row["method"]), []).append(float(row["frozen_miou"]))

    lines = [
        "# 分退化泛化表",
        "",
        "| Degradation | PSNR Delta | SSIM Delta | Boundary F Delta | Task mIoU Delta | Frozen mIoU Delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    degradations = sorted({row["degradation"] for row in restoration_rows})
    for degradation in degradations:
        base_rows = [row for row in restoration_rows if row["degradation"] == degradation and row["method"] == "bicubic"]
        sem_rows = [row for row in restoration_rows if row["degradation"] == degradation and row["method"] == "semantic_frequency"]
        if not base_rows or not sem_rows:
            continue
        values = []
        for metric in RESTORATION_METRICS:
            base = float(np.mean([float(row[metric]) for row in base_rows]))
            sem = float(np.mean([float(row[metric]) for row in sem_rows]))
            values.append(sem - base)
        base_downstream = downstream_lookup.get((degradation, "bicubic"))
        sem_downstream = downstream_lookup.get((degradation, "semantic_frequency"))
        if base_downstream and sem_downstream:
            frozen_delta = float(np.mean(sem_downstream) - np.mean(base_downstream))
            frozen_text = f"{frozen_delta:.4f}"
        else:
            frozen_text = "NA"
        lines.append(
            f"| {degradation} | {values[0]:.4f} | {values[1]:.4f} | {values[2]:.4f} | {values[3]:.4f} | {frozen_text} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_efficiency_table(path: Path, efficiency_csv: Optional[Path]) -> None:
    lines = ["# 效率表", ""]
    if efficiency_csv is None or not Path(efficiency_csv).exists():
        lines.append("- 尚未提供效率 CSV。")
    else:
        rows = _read_rows(Path(efficiency_csv))
        lines.extend(["| Method | avg_ms |", "|---|---:|"])
        for row in rows:
            lines.append(f"| {row['method']} | {float(row['avg_ms']):.4f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_deep_baseline_table(path: Path, deep_baseline_metrics_csv: Optional[Path]) -> None:
    lines = ["# 深度 Baseline 表", ""]
    if deep_baseline_metrics_csv is None or not Path(deep_baseline_metrics_csv).exists():
        lines.append("- 尚未提供深度 baseline 指标 CSV。")
    else:
        rows = _read_rows(Path(deep_baseline_metrics_csv))
        methods = sorted({row["method"] for row in rows})
        has_task = bool(rows) and "task_miou" in rows[0]
        if has_task:
            lines.extend(["| Method | PSNR | SSIM | Boundary F | Task mIoU |", "|---|---:|---:|---:|---:|"])
        else:
            lines.extend(["| Method | PSNR | SSIM | Boundary F |", "|---|---:|---:|---:|"])
        for method in methods:
            method_rows = [row for row in rows if row["method"] == method]
            if has_task:
                lines.append(
                    f"| {method} | {np.mean([float(row['psnr']) for row in method_rows]):.4f} | "
                    f"{np.mean([float(row['ssim']) for row in method_rows]):.4f} | "
                    f"{np.mean([float(row['boundary_f']) for row in method_rows]):.4f} | "
                    f"{np.mean([float(row['task_miou']) for row in method_rows]):.4f} |"
                )
            else:
                lines.append(
                    f"| {method} | {np.mean([float(row['psnr']) for row in method_rows]):.4f} | "
                    f"{np.mean([float(row['ssim']) for row in method_rows]):.4f} | "
                    f"{np.mean([float(row['boundary_f']) for row in method_rows]):.4f} |"
                )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_summary(path: Path, table: dict[str, dict[str, float]]) -> None:
    semantic = table["semantic_frequency"]
    bicubic = table["bicubic"]
    structure = table.get("structure_only", {})
    lines = [
        "# 论文表格摘要",
        "",
        f"- semantic_frequency 相对 bicubic：PSNR +{semantic['psnr'] - bicubic['psnr']:.4f}，SSIM +{semantic['ssim'] - bicubic['ssim']:.4f}，Task mIoU +{semantic['task_miou'] - bicubic['task_miou']:.4f}。",
        f"- 冻结下游分割：semantic_frequency {semantic['frozen_miou']:.4f}，bicubic {bicubic['frozen_miou']:.4f}。",
    ]
    if structure:
        lines.append(f"- 关键消融：semantic_frequency frozen mIoU 比 structure_only 高 {semantic['frozen_miou'] - structure['frozen_miou']:.4f}。")
    lines.append("- 风险：Boundary F 仍需作为受控小幅损失报告。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_paper_tables(
    restoration_metrics_csv: Path,
    downstream_metrics_csv: Path,
    output_dir: Path,
    efficiency_csv: Optional[Path] = None,
    deep_baseline_metrics_csv: Optional[Path] = None,
) -> PaperTableSummary:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    restoration_rows = _read_rows(Path(restoration_metrics_csv))
    downstream_rows = _read_rows(Path(downstream_metrics_csv))
    table = _metric_tables(restoration_rows, downstream_rows)

    main_table = output_dir / "main_table.md"
    ablation_table = output_dir / "ablation_table.md"
    generalization_table = output_dir / "generalization_table.md"
    efficiency_table = output_dir / "efficiency_table.md"
    deep_baseline_table = output_dir / "deep_baseline_table.md"
    summary_report = output_dir / "paper_tables_summary.md"
    _write_main_table(main_table, table)
    _write_ablation_table(ablation_table, table)
    _write_generalization_table(generalization_table, restoration_rows, downstream_rows)
    _write_efficiency_table(efficiency_table, efficiency_csv)
    _write_deep_baseline_table(deep_baseline_table, deep_baseline_metrics_csv)
    _write_summary(summary_report, table)
    return PaperTableSummary(
        output_dir=output_dir,
        main_table=main_table,
        ablation_table=ablation_table,
        generalization_table=generalization_table,
        efficiency_table=efficiency_table,
        deep_baseline_table=deep_baseline_table,
        summary_report=summary_report,
    )
