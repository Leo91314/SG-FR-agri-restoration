"""Phase C: CEA figure system.

Generates publication-style figures into paper/figures/:
  fig1_core_insight.png       - perception-oriented overview (from hand-drawn PPTX)
  fig4_boundary_tradeoff.png  - Boundary-F vs Task-mIoU improvement scatter (color=degradation, shape=dataset)
  fig5_sobel_sensitivity.png  - Boundary-F delta across Sobel edge quantiles q in {0.78,0.82,0.86}
  fig1_problem_motivation.png - clean / degraded / restored + segmentation montage (cea_figures_samples.py)
  fig3_qualitative_grid.png   - rows=scenes, cols={Degraded, strong baselines, Ours, GT} with zoom patches

Fig1 montage/Fig3 read real samples; Fig3 reuses the cached strong-baseline restorers (Phase A weights).
Run subcommands: `python scripts/cea_figures.py fig2 fig1_core` (PPTX sync) or `... fig4 fig5` (fast).
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

FIGDIR = Path("paper/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)
REPO = Path(__file__).resolve().parents[1]
FIG2_PPTX = REPO / "paper/figure_redraws/reference_style_single/cea_plus_reference_style_overview_clean.pptx"
FIG1_CORE_PPTX = REPO / "paper/figure_redraws/figure1_core_insight/figure1_core_insight.pptx"


def _sync_pptx_png(pptx: Path, pdf_stem: str, out_name: str, dpi: int = 300) -> None:
    import shutil
    import subprocess
    import tempfile

    if not pptx.exists():
        raise FileNotFoundError(f"missing PPTX: {pptx}")
    with tempfile.TemporaryDirectory(prefix="cea_pptx_") as tmpdir:
        tmp = Path(tmpdir)
        subprocess.run(
            [
                "soffice",
                f"-env:UserInstallation=file://{tmp / 'lo_profile'}",
                "--headless",
                "--convert-to",
                "pdf",
                "--outdir",
                str(tmp),
                str(pptx),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        pdf = tmp / f"{pdf_stem}.pdf"
        if not pdf.exists():
            raise FileNotFoundError(f"PPTX export failed: {pdf}")
        prefix = tmp / out_name.removesuffix(".png")
        subprocess.run(["pdftoppm", "-png", "-r", str(dpi), str(pdf), str(prefix)], check=True)
        exported = tmp / f"{prefix.name}-1.png"
        if not exported.exists():
            raise FileNotFoundError(f"PDF rasterization failed under {tmp}")
        shutil.copy2(exported, FIGDIR / out_name)
    print("wrote", FIGDIR / out_name, "from", pptx)


# ----------------------------------------------------------------- Fig 1 (core insight)
def fig1_core_insight() -> None:
    """Sync perception-oriented overview from hand-drawn PPTX."""
    _sync_pptx_png(FIG1_CORE_PPTX, "figure1_core_insight", "fig1_core_insight.png")


# ----------------------------------------------------------------- Fig 2
def fig2_framework() -> None:
    """Sync framework figure from the hand-drawn PPTX (reference style overview)."""
    _sync_pptx_png(FIG2_PPTX, "cea_plus_reference_style_overview_clean", "fig2_framework.png")


BOUND = Path("results/cea/boundary_analysis")

DEG_COLOR = {"x4_blur": "#1b9e77", "x4_fog": "#d95f02", "x4_mixed": "#7570b3"}
DEG_LABEL = {"x4_blur": "structure (x4 blur)", "x4_fog": "veil (fog)", "x4_mixed": "mixed"}
DS_MARKER = {"WeedsGalore": "o", "LoveDA-Rural": "s", "LoveDA-Urban": "^"}


def _read_metrics():
    rows = []
    with (BOUND / "metrics.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


# ----------------------------------------------------------------- Fig 4
def fig4_boundary_tradeoff(method: str = "semantic_frequency", quantile: float = 0.82) -> None:
    rows = [r for r in _read_metrics() if abs(float(r["edge_quantile"]) - quantile) < 1e-6]
    # group by (dataset, degradation, sample) -> {method: (bf, miou)}
    agg = defaultdict(dict)
    for r in rows:
        key = (r["dataset"], r["degradation"], r["sample"])
        agg[key][r["method"]] = (float(r["boundary_f"]), float(r["task_miou"]))
    deltas = defaultdict(lambda: {"bf": [], "miou": []})
    for (ds, deg, _samp), md in agg.items():
        if "bicubic" in md and method in md:
            deltas[(ds, deg)]["bf"].append(md[method][0] - md["bicubic"][0])
            deltas[(ds, deg)]["miou"].append(md[method][1] - md["bicubic"][1])

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.axhline(0, color="#bbb", lw=0.8, zorder=0)
    ax.axvline(0, color="#bbb", lw=0.8, zorder=0)
    for (ds, deg), d in sorted(deltas.items()):
        x, y = float(np.mean(d["bf"])), float(np.mean(d["miou"]))
        ax.scatter(x, y, s=120, marker=DS_MARKER.get(ds, "o"), color=DEG_COLOR.get(deg, "#444"),
                   edgecolor="black", linewidth=0.6, alpha=0.9, zorder=3)
    ax.set_xlabel("Boundary-F improvement vs bicubic")
    ax.set_ylabel("Task mIoU improvement vs bicubic")
    ax.set_title("Boundary fidelity vs semantic utility trade-off")
    ax.margins(x=0.18, y=0.12)
    # quadrant annotation in the empty mid-region
    ax.text(0.40, 0.62, "task gain despite\nboundary loss", transform=ax.transAxes,
            va="center", ha="center", fontsize=9, color="#777", style="italic")
    deg_handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=DEG_COLOR[k], markeredgecolor="k",
                          markersize=9, label=DEG_LABEL[k]) for k in DEG_COLOR]
    ds_handles = [Line2D([0], [0], marker=DS_MARKER[k], color="w", markerfacecolor="#888", markeredgecolor="k",
                         markersize=9, label=k) for k in DS_MARKER]
    leg1 = ax.legend(handles=deg_handles, title="degradation", loc="center left", fontsize=8, title_fontsize=8.5)
    ax.add_artist(leg1)
    ax.legend(handles=ds_handles, title="dataset", loc="upper right", fontsize=8, title_fontsize=8.5)
    fig.tight_layout()
    out = FIGDIR / "fig4_boundary_tradeoff.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ----------------------------------------------------------------- Fig 5
def fig5_sobel_sensitivity(method: str = "semantic_frequency") -> None:
    s = json.loads((BOUND / "summary.json").read_text(encoding="utf-8"))
    rows = [r for r in s["summary_rows"] if r["method"] == method]
    datasets = sorted({r["dataset"] for r in rows})
    quantiles = sorted({float(r["edge_quantile"]) for r in rows})
    colors = {"WeedsGalore": "#1b9e77", "LoveDA-Rural": "#d95f02", "LoveDA-Urban": "#7570b3"}

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.axhline(0, color="#bbb", lw=0.8)
    for ds in datasets:
        ys, los, his = [], [], []
        for q in quantiles:
            m = next(r for r in rows if r["dataset"] == ds and abs(float(r["edge_quantile"]) - q) < 1e-6)
            ys.append(m["boundary_f_delta"]); los.append(m["boundary_f_ci_low"]); his.append(m["boundary_f_ci_high"])
        ax.errorbar(quantiles, ys,
                    yerr=[np.array(ys) - np.array(los), np.array(his) - np.array(ys)],
                    marker="o", capsize=3, color=colors.get(ds, "#444"), label=ds, lw=1.6)
    ax.set_xlabel("Sobel edge quantile $q$")
    ax.set_ylabel(r"Boundary-F $\Delta$ vs bicubic")
    ax.set_title("Boundary-F trade-off is stable across edge thresholds")
    ax.set_xticks(quantiles)
    ax.legend(fontsize=8.5, title="dataset", title_fontsize=9)
    fig.tight_layout()
    out = FIGDIR / "fig5_sobel_sensitivity.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    cmds = sys.argv[1:] or ["fig4", "fig5"]
    if "fig1_core" in cmds:
        fig1_core_insight()
    if "fig2" in cmds:
        fig2_framework()
    if "fig4" in cmds:
        fig4_boundary_tradeoff()
    if "fig5" in cmds:
        fig5_sobel_sensitivity()
    if "fig1" in cmds:
        from cea_figures_samples import fig1_problem_motivation
        fig1_problem_motivation()
    if "fig3" in cmds:
        from cea_figures_samples import fig3_qualitative_grid
        fig3_qualitative_grid()
