"""SAT-style main-text figures: leakage-free framework (Fig.1) and when-restoration-helps montage (Fig.4)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

FIGDIR = Path("paper/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "figure.dpi": 300,
    }
)


def _box(ax, xy, wh, text, fc="#f7f9fc", ec="#2c3e50", fontsize=8, weight="normal"):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight)


def _arrow(ax, p0, p1, color="#2c3e50"):
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=1.1,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def fig1_framework_sat():
    """Three-column leakage-free narrative: training-only | blind restoration | frozen evaluation."""
    fig, ax = plt.subplots(figsize=(12.5, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    titles = [
        "(A) Training only",
        "(B) Blind restoration (inference)",
        "(C) Task-oriented evaluation",
    ]
    for i, title in enumerate(titles):
        ax.text(0.17 + i * 0.33, 0.94, title, ha="center", fontsize=11, weight="bold")

    # Panel A
    _box(ax, (0.03, 0.62), (0.28, 0.18), "Degraded UAV patch\n(low-res / blur / fog)")
    _box(ax, (0.03, 0.38), (0.28, 0.18), "Clean reference patch\n(full-res GT)")
    _box(ax, (0.03, 0.14), (0.28, 0.18), "Semantic guide\n(GT mask or pseudo-mask)")
    _arrow(ax, (0.17, 0.62), (0.17, 0.56))
    _box(
        ax,
        (0.06, 0.48),
        (0.22, 0.08),
        "SG-FR training\n(reconstruction + semantic aux.)",
        fc="#e8f4fd",
        weight="bold",
    )
    ax.text(
        0.17,
        0.05,
        "Semantic head active\nNo mIoU loss",
        ha="center",
        fontsize=8,
        color="#c0392b",
        weight="bold",
    )

    # Panel B
    _box(ax, (0.36, 0.62), (0.26, 0.18), "Degraded UAV image\n(blind input)")
    _arrow(ax, (0.49, 0.62), (0.49, 0.56))
    _box(ax, (0.39, 0.46), (0.20, 0.10), "Shared encoder", fc="#eef7ee")
    _arrow(ax, (0.49, 0.46), (0.49, 0.40))
    _box(ax, (0.36, 0.28), (0.08, 0.10), "Structure\n$S$")
    _box(ax, (0.45, 0.28), (0.08, 0.10), "Texture\n$T$")
    _box(ax, (0.54, 0.28), (0.08, 0.10), "Modulation\n$\\alpha$")
    _arrow(ax, (0.49, 0.28), (0.49, 0.20))
    _box(ax, (0.39, 0.10), (0.20, 0.10), "Restored image $\\hat I=S+\\alpha\\odot T$", fc="#fff4e6")
    ax.text(
        0.49,
        0.03,
        "No GT mask / no degradation label\nSemantic head inactive (inert)",
        ha="center",
        fontsize=8,
        color="#c0392b",
        weight="bold",
    )

    # Panel C
    _box(ax, (0.69, 0.58), (0.26, 0.16), "Restored image\n(no GT at inference)")
    _arrow(ax, (0.82, 0.58), (0.82, 0.50))
    _box(
        ax,
        (0.69, 0.34),
        (0.26, 0.14),
        "Frozen segmenter\nSegFormer-B0 / DeepLabV3+",
        fc="#f5eef8",
    )
    _arrow(ax, (0.82, 0.34), (0.82, 0.26))
    _box(ax, (0.72, 0.12), (0.20, 0.12), "Task metric\nmIoU vs bicubic", fc="#fdecea", weight="bold")
    ax.text(
        0.82,
        0.04,
        "Segmenter never trained on restored images\nRestorer never optimized on mIoU",
        ha="center",
        fontsize=7.5,
        color="#c0392b",
        style="italic",
    )

    out = FIGDIR / "fig1_framework.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


def fig4_when_helps():
    """Merge existing analysis plots into one SAT narrative figure."""
    paths = [
        ("A  Task damage vs task gain", FIGDIR / "quality_gain_regimes.png"),
        ("B  Boundary-F vs task gain", FIGDIR / "fig4_boundary_tradeoff.png"),
        ("C  Structure vs veil response", FIGDIR / "veil_vs_structure.png"),
    ]
    imgs = []
    for _, p in paths:
        if not p.exists():
            raise FileNotFoundError(p)
        imgs.append(Image.open(p).convert("RGB"))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    for ax, (label, im) in zip(axes, zip([p[0] for p in paths], imgs)):
        ax.imshow(im)
        ax.set_title(label, fontsize=10, loc="left", weight="bold")
        ax.axis("off")
    fig.suptitle(
        "When restoration helps agricultural perception (and when it does not)",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    out = FIGDIR / "fig4_when_helps.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    import sys

    cmds = sys.argv[1:] or ["fig1", "fig4"]
    if "fig1" in cmds:
        fig1_framework_sat()
    if "fig4" in cmds:
        fig4_when_helps()
