from pathlib import Path

from cea_plus.paper_tables import write_paper_tables


def test_write_paper_tables_combines_restoration_downstream_and_efficiency(tmp_path: Path):
    restoration_csv = tmp_path / "metrics.csv"
    restoration_csv.write_text(
        "\n".join(
            [
                "sample,degradation,scale,method,psnr,ssim,boundary_f,task_miou",
                "a,x4_fog,4,bicubic,10,0.4,0.5,0.3",
                "a,x4_fog,4,uniform_sharp,11,0.5,0.6,0.31",
                "a,x4_fog,4,semantic_frequency,14,0.6,0.58,0.7",
                "a,x4_fog,4,semantic_edge_aware,13.5,0.62,0.61,0.65",
                "a,x4_fog,4,semantic_no_mod,13,0.58,0.57,0.32",
                "a,x4_fog,4,semantic_fixed_alpha,13.2,0.59,0.575,0.34",
                "a,x4_fog,4,structure_only,13.1,0.585,0.572,0.33",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    downstream_csv = tmp_path / "downstream_metrics.csv"
    downstream_csv.write_text(
        "\n".join(
            [
                "sample,degradation,scale,method,frozen_miou",
                "a,x4_fog,4,bicubic,0.4",
                "a,x4_fog,4,uniform_sharp,0.41",
                "a,x4_fog,4,semantic_frequency,0.8",
                "a,x4_fog,4,semantic_edge_aware,0.75",
                "a,x4_fog,4,semantic_no_mod,0.45",
                "a,x4_fog,4,semantic_fixed_alpha,0.46",
                "a,x4_fog,4,structure_only,0.44",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    efficiency_csv = tmp_path / "efficiency.csv"
    efficiency_csv.write_text("method,avg_ms\nbicubic,1.2\nsemantic_frequency,1.5\n", encoding="utf-8")
    deep_csv = tmp_path / "deep_baseline_metrics.csv"
    deep_csv.write_text(
        "\n".join(
            [
                "sample,degradation,scale,method,psnr,ssim,boundary_f,task_miou",
                "a,x4_fog,4,bicubic,10,0.4,0.5,0.3",
                "a,x4_fog,4,tiny_rescnn,13,0.55,0.52,0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = write_paper_tables(
        restoration_metrics_csv=restoration_csv,
        downstream_metrics_csv=downstream_csv,
        efficiency_csv=efficiency_csv,
        deep_baseline_metrics_csv=deep_csv,
        output_dir=tmp_path / "paper",
    )

    main = summary.main_table.read_text(encoding="utf-8")
    ablation = summary.ablation_table.read_text(encoding="utf-8")
    generalization = summary.generalization_table.read_text(encoding="utf-8")
    efficiency = summary.efficiency_table.read_text(encoding="utf-8")
    deep = summary.deep_baseline_table.read_text(encoding="utf-8")

    assert "Frozen mIoU" in main
    assert "semantic_frequency" in main
    assert "0.8000" in main
    assert "structure_only" in ablation
    assert "semantic_edge_aware" in ablation
    assert "x4_fog" in generalization
    assert "avg_ms" in efficiency
    assert "tiny_rescnn" in deep
    assert "Task mIoU" in deep
    assert "| tiny_rescnn | 13.0000 | 0.5500 | 0.5200 | 0.5000 |" in deep
