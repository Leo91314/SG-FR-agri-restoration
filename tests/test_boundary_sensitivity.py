from pathlib import Path

from cea_plus.boundary_sensitivity import BoundarySensitivityDataset, run_boundary_quantile_sensitivity
from cea_plus.degradation import build_degradation_plan
from cea_plus.synthesis import make_synthetic_agri_sample


def test_boundary_quantile_sensitivity_writes_metrics_and_report(tmp_path: Path):
    samples = [
        make_synthetic_agri_sample(seed=11, size=64),
        make_synthetic_agri_sample(seed=12, size=64),
    ]

    summary = run_boundary_quantile_sensitivity(
        output_dir=tmp_path / "sensitivity",
        datasets=[BoundarySensitivityDataset(name="Synthetic", samples=samples)],
        degradation_plan=build_degradation_plan(scales=(2,), modes=("fog",)),
        quantiles=(0.78, 0.82),
        methods=("bicubic", "semantic_frequency"),
        seed=23,
    )

    metrics = summary.metrics_csv.read_text(encoding="utf-8")
    report = summary.report.read_text(encoding="utf-8")
    summary_json = summary.summary_json.read_text(encoding="utf-8")

    assert "dataset,sample,degradation,scale,method,edge_quantile,boundary_f,task_miou" in metrics
    assert "0.78" in metrics
    assert "0.82" in metrics
    assert "Boundary F 阈值敏感性报告" in report
    assert "semantic_frequency" in report
    assert '"summary_rows"' in summary_json


def test_boundary_quantile_sensitivity_skips_unused_restorations(tmp_path: Path, monkeypatch):
    from cea_plus import restoration

    def fail_if_called(*args, **kwargs):
        raise AssertionError("unused restoration was computed")

    monkeypatch.setattr(restoration, "semantic_edge_aware_restore", fail_if_called)

    run_boundary_quantile_sensitivity(
        output_dir=tmp_path / "sensitivity",
        datasets=[BoundarySensitivityDataset(name="Synthetic", samples=[make_synthetic_agri_sample(seed=13, size=64)])],
        degradation_plan=build_degradation_plan(scales=(2,), modes=("fog",)),
        quantiles=(0.82,),
        methods=("bicubic", "semantic_frequency"),
        seed=23,
    )
