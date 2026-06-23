"""Second agricultural dataset: replicate the positive task result on CWFID.

CWFID (Haug & Ostermann 2014) is a real carrot/weed field dataset with pixel-level plant annotations.
We form the same binary plants-vs-soil target as WeedsGalore and run the same no-leak protocol
(frozen clean-trained SegFormer-B0 and DeepLabV3+; restorer blind at inference; paired bootstrap CIs)
on the structure-destroying and composite degradation plans. Goal: show the downstream benefit is not
a WeedsGalore artifact.
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import json
import sys
from pathlib import Path

import numpy as np
import torch

from cea_plus.deep_downstream import _resize_image, _resize_mask, train_torch_segmenter
from cea_plus.degradation import degrade_sample
from cea_plus.downstream import evaluate_frozen_segmenter
from cea_plus.metrics import psnr, ssim_score
from cea_plus.restoration import bicubic_restore
from cea_plus.semantic_inr import restore_with_semantic_inr
from cea_plus.statistics import paired_significance

from cea_exp import (load_dataset, structure_plan, composite_plan, train_inr,
                     train_tiny_rescnn, restore_tiny_rescnn)

BACKBONES = ["segformer_b0_imagenet", "deeplabv3plus_imagenet"]
METHODS = ["bicubic", "tiny_rescnn", "inr_no_semantic", "semantic_inr"]


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cwfid"
    out_name = "second_dataset" if dataset == "cwfid" else f"dataset_{dataset}"
    out = Path(f"results/cea/{out_name}")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    crop, train_limit, inr_steps, seg_steps = 256, 40, 2000, 150
    seeds = [91, 92]
    plan = structure_plan() + composite_plan()

    rows = []
    for seed in seeds:
        train_pool, test = load_dataset(dataset, crop, train_limit, 16)
        segmenters = {b: train_torch_segmenter(train_samples=train_pool, architecture=b, steps=seg_steps,
                                               crop_size=128, input_size=crop, seed=seed, device=dev)
                      for b in BACKBONES}
        inrs = {v: train_inr(v, train_pool, plan, inr_steps, seed, dev)
                for v in ("semantic_inr", "inr_no_semantic")}
        rescnn = train_tiny_rescnn(train_pool, plan, inr_steps, seed, dev)
        for s_idx, sample in enumerate(test):
            for config in plan:
                d = degrade_sample(sample, config=config, seed=seed * 2000 + s_idx * 100)
                shape = d.gt.shape[:2]
                mask_eval = _resize_mask(d.mask, (crop, crop))
                restorations = {
                    "bicubic": bicubic_restore(d.low_res, shape),
                    "tiny_rescnn": restore_tiny_rescnn(rescnn, d.low_res, shape, dev),
                    "inr_no_semantic": restore_with_semantic_inr(inrs["inr_no_semantic"], d.low_res, shape, device=dev),
                    "semantic_inr": restore_with_semantic_inr(inrs["semantic_inr"], d.low_res, shape, device=dev),
                }
                regime = "structure" if config.name.startswith(("x8_blur", "x8_noise")) else "composite"
                for m, img in restorations.items():
                    img_eval = _resize_image(img, (crop, crop))
                    row = {"seed": seed, "sample": sample.name, "condition": config.name, "regime": regime,
                           "method": m, "psnr": psnr(d.gt, img), "ssim": ssim_score(d.gt, img)}
                    for b in BACKBONES:
                        row[f"miou_{b}"] = evaluate_frozen_segmenter(segmenters[b], img_eval, mask_eval)
                    rows.append(row)

    keys = sorted({(r["seed"], r["sample"], r["condition"]) for r in rows})
    by = {(r["seed"], r["sample"], r["condition"], r["method"]): r for r in rows}

    def subset_keys(regime):
        return [k for k in keys if by[(*k, "bicubic")]["regime"] == regime]

    summary = {"dataset": dataset, "seeds": seeds, "backbones": BACKBONES, "means": {}, "delta_vs_bicubic": {}}
    metric_cols = [f"miou_{b}" for b in BACKBONES] + ["psnr", "ssim"]
    for m in METHODS:
        summary["means"][m] = {met: float(np.mean([by[(*k, m)][met] for k in keys])) for met in metric_cols}
    for m in METHODS:
        if m == "bicubic":
            continue
        entry = {}
        for b in BACKBONES:
            met = f"miou_{b}"
            for regime in ("structure", "composite", "all"):
                ks = keys if regime == "all" else subset_keys(regime)
                base = np.array([by[(*k, "bicubic")][met] for k in ks])
                prop = np.array([by[(*k, m)][met] for k in ks])
                sig = paired_significance(base, prop, seed=93, bootstraps=1500)
                entry[f"{b}:{regime}"] = {"mean_delta": sig.mean_delta, "ci_low": sig.bootstrap_ci_low,
                                          "ci_high": sig.bootstrap_ci_high, "ttest_p": sig.ttest_p}
        summary["delta_vs_bicubic"][m] = entry

    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out / "rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"means": summary["means"],
                      "semantic_inr_delta": summary["delta_vs_bicubic"]["semantic_inr"]}, indent=2))


if __name__ == "__main__":
    main()
