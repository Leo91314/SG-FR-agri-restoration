"""Reviewer point 2 + 11: honest re-aggregated statistics and quality--gain correlation.

The original main table paired Delta(mIoU) at the patch level (n=156-468), which yields
implausibly tiny p-values because patches from one image / one seed are not independent.
This script recomputes the SG-FR-vs-bicubic Delta with the pairing unit raised to:
  - patch  : every (seed, sample, degradation, scale) row (reference; the old unit)
  - image  : average over patches of each image, bootstrap over images
  - seed   : average over all patches of each seed, paired over seeds (n=3)
and reports percentile bootstrap 95% CIs at each level.

It also runs Spearman/Pearson correlation of Delta-quality (PSNR/SSIM/BoundaryF) vs Delta-mIoU,
stratified by degradation regime.

Outputs results/cea/revision_stats/{summary.json, stats.md}. No model re-run required.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CSV = ROOT / "results/cea/phase2_main/metrics.csv"
OUT = ROOT / "results/cea/revision_stats"
PROPOSED = "semantic_inr"
BASELINE = "bicubic"


def load_rows():
    with CSV.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("deep_frozen_miou", "psnr", "ssim", "boundary_f"):
            r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
        r["seed"] = int(r["seed"])
    return rows


def bootstrap_ci(delta: np.ndarray, n_boot: int = 5000, seed: int = 17):
    delta = np.asarray(delta, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, delta.size, size=(n_boot, delta.size))
    means = delta[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(delta.mean()), float(lo), float(hi)


def paired_by(rows, arch, plan, key_fields, value="deep_frozen_miou"):
    """Return paired (baseline, proposed) arrays keyed by key_fields within arch+plan."""
    sub = [r for r in rows if r["arch"] == arch and (plan is None or r["plan"] == plan)]
    base = {tuple(r[k] for k in key_fields): r[value] for r in sub if r["method"] == BASELINE}
    prop = {tuple(r[k] for k in key_fields): r[value] for r in sub if r["method"] == PROPOSED}
    common = sorted(set(base) & set(prop), key=lambda x: tuple(map(str, x)))
    b = np.array([base[k] for k in common])
    p = np.array([prop[k] for k in common])
    return b, p, common


def aggregate_delta(rows, arch, plan, unit):
    """unit in {'patch','image','seed'}: returns the paired Delta array at that unit."""
    b, p, keys = paired_by(rows, arch, plan, ["seed", "sample", "degradation", "scale"])
    delta = p - b
    if unit == "patch":
        return delta
    # group the patch-level deltas by the aggregation field
    if unit == "image":
        field_idx = 1  # sample
    elif unit == "seed":
        field_idx = 0  # seed
    else:
        raise ValueError(unit)
    groups = defaultdict(list)
    for d, k in zip(delta, keys):
        groups[k[field_idx]].append(d)
    return np.array([np.mean(v) for v in groups.values()])


def main():
    rows = load_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    archs = sorted({r["arch"] for r in rows})
    plans = sorted({r["plan"] for r in rows})

    summary = {"comparison": f"{PROPOSED} - {BASELINE}", "metric": "deep_frozen_miou", "levels": {}}
    md = ["# Revision statistics (reviewer points 2 and 11)\n",
          f"Comparison: {PROPOSED} vs {BASELINE} on deep_frozen_miou.\n",
          "## Re-aggregated paired Delta(mIoU) with 95% bootstrap CI\n",
          "| arch | regime | patch n / mean [CI] | image n / mean [CI] | seed n / mean [CI] |",
          "|---|---|---|---|---|"]

    for arch in archs:
        for plan in plans + [None]:
            label = plan or "ALL"
            cell = {}
            for unit in ("patch", "image", "seed"):
                d = aggregate_delta(rows, arch, plan, unit)
                if d.size < 2:
                    cell[unit] = {"n": int(d.size), "mean": float(d.mean()) if d.size else None,
                                  "ci_low": None, "ci_high": None}
                    continue
                m, lo, hi = bootstrap_ci(d)
                cell[unit] = {"n": int(d.size), "mean": m, "ci_low": lo, "ci_high": hi}
                if unit == "seed":
                    # also a parametric paired-style CI on the seed means (n=3)
                    tcrit = stats.t.ppf(0.975, d.size - 1)
                    se = d.std(ddof=1) / np.sqrt(d.size)
                    cell[unit]["t_ci_low"] = float(d.mean() - tcrit * se)
                    cell[unit]["t_ci_high"] = float(d.mean() + tcrit * se)
            summary["levels"][f"{arch}|{label}"] = cell

            def fmt(u):
                c = cell[u]
                if c["ci_low"] is None:
                    return f"n={c['n']} {c['mean']:+.3f}" if c["mean"] is not None else "n/a"
                return f"n={c['n']} {c['mean']:+.3f} [{c['ci_low']:+.3f},{c['ci_high']:+.3f}]"
            md.append(f"| {arch.replace('_imagenet','')} | {label} | {fmt('patch')} | {fmt('image')} | {fmt('seed')} |")

    # ---- correlation: Delta-quality vs Delta-mIoU, stratified by regime ----
    md += ["\n## Quality--gain correlation (Delta-metric vs Delta-mIoU), by regime\n",
           "| arch | regime | metric | Spearman rho (p) | Pearson r (p) | n |",
           "|---|---|---|---|---|---|"]
    corr = {}
    for arch in archs:
        for plan in plans + [None]:
            label = plan or "ALL"
            bm, pm, keys = paired_by(rows, arch, plan, ["seed", "sample", "degradation", "scale"], "deep_frozen_miou")
            dmiou = pm - bm
            for qmetric in ("psnr", "ssim", "boundary_f"):
                bq, pq, keys2 = paired_by(rows, arch, plan, ["seed", "sample", "degradation", "scale"], qmetric)
                # align keys (same ordering since same filter)
                dq = pq - bq
                mask = np.isfinite(dq) & np.isfinite(dmiou)
                if mask.sum() < 5:
                    continue
                rho, prho = stats.spearmanr(dq[mask], dmiou[mask])
                r, pr = stats.pearsonr(dq[mask], dmiou[mask])
                corr[f"{arch}|{label}|{qmetric}"] = {"spearman_rho": float(rho), "spearman_p": float(prho),
                                                      "pearson_r": float(r), "pearson_p": float(pr),
                                                      "n": int(mask.sum())}
                md.append(f"| {arch.replace('_imagenet','')} | {label} | d{qmetric} | "
                          f"{rho:+.3f} ({prho:.1e}) | {r:+.3f} ({pr:.1e}) | {int(mask.sum())} |")
    summary["correlation"] = corr

    # ---- no-reference BRISQUE vs task gain (restricted x4 subset; no boundary_f, no regime split) ----
    sb_path = ROOT / "results/cea/strong_baselines/rows.json"
    if sb_path.exists():
        sb = json.loads(sb_path.read_text(encoding="utf-8"))
        base = {(r["seed"], r["sample"], r["condition"]): r for r in sb if r["method"] == BASELINE}
        dbris, dmiou_b, dpsnr_b = [], [], []
        for r in sb:
            if r["method"] == BASELINE:
                continue
            b = base.get((r["seed"], r["sample"], r["condition"]))
            if b is None:
                continue
            dbris.append(r["brisque"] - b["brisque"])       # negative = quality improved
            dpsnr_b.append(r["psnr"] - b["psnr"])
            dmiou_b.append(r["miou"] - b["miou"])
        dbris = np.array(dbris); dmiou_b = np.array(dmiou_b); dpsnr_b = np.array(dpsnr_b)
        bris_block = {}
        for name, arr in (("d_brisque", dbris), ("d_psnr", dpsnr_b)):
            rho, prho = stats.spearmanr(arr, dmiou_b)
            r_, pr = stats.pearsonr(arr, dmiou_b)
            bris_block[name] = {"spearman_rho": float(rho), "spearman_p": float(prho),
                                "pearson_r": float(r_), "pearson_p": float(pr), "n": int(arr.size)}
        summary["brisque_correlation"] = bris_block
        md += ["\n## No-reference quality vs task gain (strong_baselines x4 subset; all methods vs bicubic pooled)\n",
               "Note: x4 single-degradation + compound only; no boundary_f; no structure/composite/veil split.",
               "Lower BRISQUE = better; a *negative* d_brisque that helps the task would correlate *negatively* with d_mIoU.\n",
               "| metric vs d_mIoU | Spearman rho (p) | Pearson r (p) | n |",
               "|---|---|---|---|"]
        for name, c in bris_block.items():
            md.append(f"| {name} | {c['spearman_rho']:+.3f} ({c['spearman_p']:.1e}) | "
                      f"{c['pearson_r']:+.3f} ({c['pearson_p']:.1e}) | {c['n']} |")

    # ---- CWFID / CoFly image-level CIs (second_dataset rows.json format) ----
    md += ["\n## Agricultural replication image-level CI (SegFormer-B0, semantic_inr vs bicubic)\n",
           "| dataset | regime | image n / mean [CI] |", "|---|---|---|"]
    repl = {}
    for ds_name, path in (("CWFID", ROOT / "results/cea/second_dataset/rows.json"),
                          ("CoFly", ROOT / "results/cea/dataset_cofly/rows.json")):
        if not path.exists():
            continue
        rr = json.loads(path.read_text(encoding="utf-8"))
        col = "miou_segformer_b0_imagenet"
        base = {(r["seed"], r["sample"], r["condition"]): r[col] for r in rr if r["method"] == BASELINE}
        prop = {(r["seed"], r["sample"], r["condition"]): r[col] for r in rr if r["method"] == PROPOSED}
        regimes = sorted({r["regime"] for r in rr}) + [None]
        reg_of = {(r["seed"], r["sample"], r["condition"]): r["regime"] for r in rr}
        for reg in regimes:
            keys = [k for k in (set(base) & set(prop)) if reg is None or reg_of[k] == reg]
            groups = defaultdict(list)
            for k in keys:
                groups[k[1]].append(prop[k] - base[k])  # by sample
            d = np.array([np.mean(v) for v in groups.values()])
            if d.size < 2:
                continue
            m, lo, hi = bootstrap_ci(d)
            repl[f"{ds_name}|{reg or 'all'}"] = {"n": int(d.size), "mean": m, "ci_low": lo, "ci_high": hi}
            md.append(f"| {ds_name} | {reg or 'all'} | n={d.size} {m:+.3f} [{lo:+.3f},{hi:+.3f}] |")
    summary["agri_replication"] = repl

    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (OUT / "stats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
