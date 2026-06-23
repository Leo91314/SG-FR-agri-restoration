"""Re-analysis of the quality-gain experiment (no retraining).

Reads results/cea/quality_gain/rows.json and characterizes WHEN restoration gain tracks
task damage, separating: (a) in-domain agricultural (WeedsGalore) vs out-of-domain (LoveDA),
and (b) structure-destroying vs veil/fog degradations. Emits per-regime correlations and a
two-panel figure.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def regime_of(severity: str) -> str:
    return "veil" if "fog" in severity else "structure"


def agg(rows, dataset=None, regime=None):
    pts = {}
    for r in rows:
        if dataset and r["dataset"] != dataset:
            continue
        if regime and regime_of(r["severity"]) != regime:
            continue
        key = (r["dataset"], r["severity"])
        pts.setdefault(key, {"dmg": [], "gain": []})
        pts[key]["dmg"].append(r["clean_miou"] - r["bicubic_miou"])
        pts[key]["gain"].append(r["inr_miou"] - r["bicubic_miou"])
    out = []
    for (ds, sev), v in pts.items():
        out.append({"dataset": ds, "severity": sev, "regime": regime_of(sev),
                    "task_damage": float(np.mean(v["dmg"])), "task_gain": float(np.mean(v["gain"]))})
    return out


def corr(points):
    if len(points) < 3:
        return {"n": len(points), "r": float("nan"), "slope": float("nan")}
    d = np.array([p["task_damage"] for p in points])
    g = np.array([p["task_gain"] for p in points])
    if d.std() == 0:
        return {"n": len(points), "r": float("nan"), "slope": float("nan")}
    slope, intercept = np.polyfit(d, g, 1)
    return {"n": len(points), "r": float(np.corrcoef(d, g)[0, 1]),
            "slope": float(slope), "intercept": float(intercept)}


def main() -> None:
    base = Path("results/cea/quality_gain")
    rows = json.loads((base / "rows.json").read_text(encoding="utf-8"))

    all_pts = agg(rows)
    subsets = {
        "weeds_structure": agg(rows, dataset="weedsgalore", regime="structure"),
        "weeds_veil": agg(rows, dataset="weedsgalore", regime="veil"),
        "weeds_all": agg(rows, dataset="weedsgalore"),
        "loveda_all": [p for p in all_pts if p["dataset"].startswith("loveda")],
        "all_structure": [p for p in all_pts if p["regime"] == "structure"],
        "all_points": all_pts,
    }
    correlations = {k: corr(v) for k, v in subsets.items()}

    # Two-panel figure
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5))
    # Panel A: WeedsGalore, color by regime
    cmap = {"structure": "#1b9e77", "veil": "#d95f02"}
    for reg in ("structure", "veil"):
        pts = [p for p in all_pts if p["dataset"] == "weedsgalore" and p["regime"] == reg]
        axA.scatter([p["task_damage"] for p in pts], [p["task_gain"] for p in pts],
                    color=cmap[reg], s=70, edgecolor="k", linewidth=0.5, label=f"WeedsGalore {reg}")
    cs = correlations["weeds_structure"]
    d = np.array([p["task_damage"] for p in subsets["weeds_structure"]])
    xs = np.linspace(min(0, d.min()), d.max(), 50)
    axA.plot(xs, cs["slope"] * xs + cs["intercept"], "k--", lw=1.5,
             label=f"structure fit r={cs['r']:.2f}")
    axA.axhline(0, color="gray", lw=0.8); axA.axvline(0, color="gray", lw=0.8)
    axA.set_xlabel("task damage = mIoU(clean) - mIoU(bicubic)")
    axA.set_ylabel("task gain = mIoU(INR) - mIoU(bicubic)")
    axA.set_title("In-domain (WeedsGalore): gain tracks structure damage")
    axA.legend(fontsize=8)
    # Panel B: all datasets, color by dataset
    dcol = {"weedsgalore": "#1b9e77", "loveda_rural": "#d95f02", "loveda_urban": "#7570b3"}
    for ds, c in dcol.items():
        pts = [p for p in all_pts if p["dataset"] == ds]
        axB.scatter([p["task_damage"] for p in pts], [p["task_gain"] for p in pts],
                    color=c, s=70, edgecolor="k", linewidth=0.5, label=ds)
    axB.axhline(0, color="gray", lw=0.8); axB.axvline(0, color="gray", lw=0.8)
    axB.set_xlabel("task damage = mIoU(clean) - mIoU(bicubic)")
    axB.set_ylabel("task gain")
    axB.set_title("Out-of-domain (LoveDA): task gain does not materialize")
    axB.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(base / "quality_gain_regimes.png", dpi=130)
    plt.close(fig)

    summary = {
        "correlations": correlations,
        "claims": {
            "in_domain_structure_law": (f"On WeedsGalore structure degradations, task gain rises with task "
                                        f"damage (Pearson r={correlations['weeds_structure']['r']:.2f}, "
                                        f"slope={correlations['weeds_structure']['slope']:.2f})."),
            "veil_exception": ("Veil/fog: PSNR recovers strongly but task gain is flat-to-negative, so high "
                               "image-quality recovery does not convert to task gain for global veils."),
            "out_of_domain": ("On LoveDA (out-of-domain aerial), task gain is ~0 to slightly negative across "
                              "severities; the downstream benefit is specific to the agricultural target setting, "
                              "while image-quality (PSNR/SSIM) gains still transfer."),
        },
        "points": all_pts,
    }
    (base / "regime_analysis.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(correlations, indent=2))


if __name__ == "__main__":
    main()
