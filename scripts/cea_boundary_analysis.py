"""Phase B: boundary trade-off + Sobel-quantile sensitivity (Layer-3 analysis data).

This is the *analysis/ablation branch*, not the no-leak main method: it uses the mask-aware
semantic_frequency restorer to study WHY restoration that helps the downstream task can still lower
Boundary-F under veil/fog. We compute, per (dataset, degradation), the Boundary-F delta and Task-mIoU
delta (for the Fig.4 trade-off scatter), and sweep the Sobel edge quantile q in {0.78, 0.82, 0.86}
(for the Fig.5 sensitivity plot) to show the negative Boundary-F under fog is not a threshold artifact.

Datasets: WeedsGalore (UAV maize), LoveDA Rural, LoveDA Urban. Degradations: x4 blur, fog, mixed.
Outputs: results/cea/boundary_analysis/{metrics.csv, summary.json, *_report.md}.
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

from pathlib import Path

from cea_plus.boundary_sensitivity import BoundarySensitivityDataset, run_boundary_quantile_sensitivity
from cea_plus.degradation import DegradationConfig

from cea_exp import load_dataset

DEGRADATIONS = [
    DegradationConfig(name="x4_blur", scale=4, rain_density=0, fog_strength=0.0, blur_sigma=1.5, noise_sigma=0.01),
    DegradationConfig(name="x4_fog", scale=4, rain_density=0, fog_strength=0.45, blur_sigma=0.8, noise_sigma=0.01),
    DegradationConfig(name="x4_mixed", scale=4, rain_density=30, fog_strength=0.38, blur_sigma=0.8, noise_sigma=0.015, jpeg_quality=50),
]

DATASETS = [
    ("WeedsGalore", "weedsgalore", "UAV maize crop/weed rows"),
    ("LoveDA-Rural", "loveda_rural", "Rural remote-sensing land cover (OOD scope)"),
    ("LoveDA-Urban", "loveda_urban", "Urban remote-sensing land cover (OOD scope)"),
]


def main() -> None:
    crop, n = 256, 12
    datasets = []
    for label, key, desc in DATASETS:
        _, test = load_dataset(key, crop, 20, n)
        datasets.append(BoundarySensitivityDataset(name=label, samples=list(test)[:n], description=desc))
        print(f"[load] {label}: {len(test)} samples")

    summary = run_boundary_quantile_sensitivity(
        output_dir=Path("results/cea/boundary_analysis"),
        datasets=datasets,
        degradation_plan=DEGRADATIONS,
        quantiles=(0.78, 0.82, 0.86),
        methods=("bicubic", "semantic_frequency", "semantic_boundary_guard"),
        seed=7,
    )
    print(f"[done] {len(summary.summary_rows)} summary rows -> {summary.summary_json}")


if __name__ == "__main__":
    main()
