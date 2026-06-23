from pathlib import Path
import json
import subprocess
import sys


def _write_metrics(path: Path, dataset: str, semantic_task: float, semantic_boundary: float) -> None:
    rows = ["sample,degradation,scale,method,psnr,ssim,boundary_f,task_miou"]
    for idx in range(2):
        sample = f"{dataset}_{idx}"
        rows.extend(
            [
                f"{sample},x2_fog,2,bicubic,10,0.40,0.70,0.20",
                f"{sample},x2_fog,2,semantic_frequency,13,0.50,{semantic_boundary},{semantic_task}",
                f"{sample},x2_fog,2,semantic_boundary_guard,11,0.45,0.72,0.22",
                f"{sample},x2_fog,2,structure_only,14,0.55,0.71,0.20",
            ]
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_cross_dataset_report_cli_writes_summary_tables(tmp_path: Path):
    weeds = tmp_path / "weeds.csv"
    rural = tmp_path / "rural.csv"
    urban = tmp_path / "urban.csv"
    _write_metrics(weeds, "weeds", semantic_task=0.60, semantic_boundary=0.69)
    _write_metrics(rural, "rural", semantic_task=0.30, semantic_boundary=0.68)
    _write_metrics(urban, "urban", semantic_task=0.40, semantic_boundary=0.67)

    out = tmp_path / "report"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cea_plus.run_cross_dataset_report",
            "--out",
            str(out),
            "--dataset",
            f"Weeds|{weeds}|sample full description",
            "--dataset",
            f"LoveDA Rural|{rural}|rural n=2",
            "--dataset",
            f"LoveDA Urban|{urban}|urban n=2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = (out / "cross_dataset_report.md").read_text(encoding="utf-8")
    deltas = (out / "paired_deltas_vs_bicubic.csv").read_text(encoding="utf-8")
    summary = (out / "summary.json").read_text(encoding="utf-8")

    assert "report=" in result.stdout
    assert "LoveDA Rural" in report
    assert "Task mIoU" in report
    assert "+0.100000" in report
    assert "semantic_boundary_guard" in report
    assert "dataset_name,method,metric,mean_delta" in deltas
    assert '"datasets"' in summary


def test_cross_dataset_report_cli_filters_degradations(tmp_path: Path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "\n".join(
            [
                "sample,degradation,scale,method,psnr,ssim,boundary_f,task_miou",
                "a,x2_fog,2,bicubic,10,0.4,0.70,0.20",
                "a,x2_fog,2,semantic_frequency,13,0.5,0.68,0.30",
                "b,x2_fog,2,bicubic,10,0.4,0.70,0.20",
                "b,x2_fog,2,semantic_frequency,13,0.5,0.68,0.30",
                "a,x2_mixed,2,bicubic,10,0.4,0.70,0.20",
                "a,x2_mixed,2,semantic_frequency,13,0.5,0.80,0.40",
                "b,x2_mixed,2,bicubic,10,0.4,0.70,0.20",
                "b,x2_mixed,2,semantic_frequency,13,0.5,0.80,0.40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "fog_report"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cea_plus.run_cross_dataset_report",
            "--out",
            str(out),
            "--include-degradations",
            "x2_fog",
            "--dataset",
            f"Toy|{metrics}|toy fog only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = (out / "cross_dataset_report.md").read_text(encoding="utf-8")
    deltas = (out / "paired_deltas_vs_bicubic.csv").read_text(encoding="utf-8")

    assert "| Toy |" in report
    assert "| Toy | `Toy" not in report
    assert "Selected rows |" in report
    assert "| Toy | `" in report
    assert "+0.100000" in report
    assert "0.8000000000000002" not in deltas


def test_cross_dataset_report_cli_filters_degradation_modes(tmp_path: Path):
    metrics = tmp_path / "metrics.csv"
    metrics.write_text(
        "\n".join(
            [
                "sample,degradation,scale,method,psnr,ssim,boundary_f,task_miou",
                "a,x2_fog,2,bicubic,10,0.4,0.70,0.20",
                "a,x2_fog,2,semantic_frequency,13,0.5,0.68,0.30",
                "b,x2_fog,2,bicubic,10,0.4,0.70,0.20",
                "b,x2_fog,2,semantic_frequency,13,0.5,0.68,0.30",
                "a,x4_fog,4,bicubic,10,0.4,0.70,0.20",
                "a,x4_fog,4,semantic_frequency,13,0.5,0.67,0.31",
                "b,x4_fog,4,bicubic,10,0.4,0.70,0.20",
                "b,x4_fog,4,semantic_frequency,13,0.5,0.67,0.31",
                "a,x2_mixed,2,bicubic,10,0.4,0.70,0.20",
                "a,x2_mixed,2,semantic_frequency,13,0.5,0.80,0.40",
                "b,x2_mixed,2,bicubic,10,0.4,0.70,0.20",
                "b,x2_mixed,2,semantic_frequency,13,0.5,0.80,0.40",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = tmp_path / "fog_mode_report"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cea_plus.run_cross_dataset_report",
            "--out",
            str(out),
            "--include-modes",
            "fog",
            "--dataset",
            f"Toy|{metrics}|toy fog mode",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    deltas = (out / "paired_deltas_vs_bicubic.csv").read_text(encoding="utf-8")

    dataset = summary["datasets"][0]
    assert dataset["paired_inputs"] == 4
    assert dataset["selected_rows"] == 8
    assert "0.8000000000000002" not in deltas
