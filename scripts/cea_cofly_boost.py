"""Diagnostic: can stronger semantic-guided detail close the CoFly gap vs a generic restorer?

Our default Semantic-INR damps texture (texture_residual_scale=0.2) and injects semantic-guided
detail with semantic_detail_boost=0.30. On CoFly's fine-grained sparse weed target this under-restores
detail and loses to Tiny ResCNN. We sweep the two principled knobs (semantic_detail_boost,
texture_residual_scale) and compare frozen SegFormer-B0 mIoU against bicubic and Tiny ResCNN on the
same structure+composite plan. This is a sensitivity analysis, not per-dataset tuning: any winning
setting must later be validated on WeedsGalore before adoption.
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
from cea_plus.semantic_inr import TinySemanticINR, train_semantic_inr_steps, restore_with_semantic_inr
from cea_plus.statistics import paired_significance
from cea_plus.restoration import bicubic_restore

from cea_exp import load_dataset, structure_plan, composite_plan, build_batches, train_tiny_rescnn, restore_tiny_rescnn

CONFIGS = {
    # default: semantic-detail pathway is DEAD because base_sharpen=0 => (base-up)=0
    "default_sharp0": dict(semantic_detail_boost=0.30, texture_residual_scale=0.2, base_sharpen_strength=0.0),
    # activate the pathway: base_sharpen>0 makes semantic-guided detail injection live
    "sharp0.5_boost1.0": dict(semantic_detail_boost=1.00, texture_residual_scale=0.2, base_sharpen_strength=0.5),
    "sharp1.0_boost1.0": dict(semantic_detail_boost=1.00, texture_residual_scale=0.2, base_sharpen_strength=1.0),
    "sharp1.0_boost2.0_tex0.4": dict(semantic_detail_boost=2.00, texture_residual_scale=0.4, base_sharpen_strength=1.0),
}


def train_custom_inr(train_pool, plan, steps, seed, device, semantic_detail_boost, texture_residual_scale,
                     base_sharpen_strength=0.0):
    torch.manual_seed(seed)
    model = TinySemanticINR(hidden_channels=48, base_sharpen_strength=base_sharpen_strength,
                            structure_residual_scale=0.6, texture_residual_scale=texture_residual_scale,
                            semantic_detail_boost=semantic_detail_boost)
    train_semantic_inr_steps(model, batches=build_batches(train_pool, plan, seed), steps=steps,
                             learning_rate=1e-3, task_loss_weight=0.0, semantic_loss_weight=0.03,
                             frequency_loss_weight=0.20, seed=seed, device=device)
    return model


def main() -> None:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "cofly"
    out = Path(f"results/cea/boost_{dataset}")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    crop, train_limit, inr_steps, seg_steps = 256, 40, 2000, 150
    seeds = [91, 92]
    plan = structure_plan() + composite_plan()

    rows = []
    for seed in seeds:
        train_pool, test = load_dataset(dataset, crop, train_limit, 16)
        seg = train_torch_segmenter(train_samples=train_pool, architecture="segformer_b0_imagenet",
                                    steps=seg_steps, crop_size=128, input_size=crop, seed=seed, device=dev)
        rescnn = train_tiny_rescnn(train_pool, plan, inr_steps, seed, dev)
        inrs = {name: train_custom_inr(train_pool, plan, inr_steps, seed, dev, **cfg)
                for name, cfg in CONFIGS.items()}
        for s_idx, sample in enumerate(test):
            for config in plan:
                d = degrade_sample(sample, config=config, seed=seed * 2000 + s_idx * 100)
                shape = d.gt.shape[:2]
                mask_eval = _resize_mask(d.mask, (crop, crop))
                outs = {"bicubic": bicubic_restore(d.low_res, shape),
                        "tiny_rescnn": restore_tiny_rescnn(rescnn, d.low_res, shape, dev)}
                for name, model in inrs.items():
                    outs[name] = restore_with_semantic_inr(model, d.low_res, shape, device=dev)
                for m, img in outs.items():
                    rows.append({"seed": seed, "sample": sample.name, "condition": config.name, "method": m,
                                 "miou": evaluate_frozen_segmenter(seg, _resize_image(img, (crop, crop)), mask_eval)})

    methods = ["bicubic", "tiny_rescnn"] + list(CONFIGS)
    keys = sorted({(r["seed"], r["sample"], r["condition"]) for r in rows})
    by = {(r["seed"], r["sample"], r["condition"], r["method"]): r for r in rows}
    means = {m: float(np.mean([by[(*k, m)]["miou"] for k in keys])) for m in methods}
    tiny = np.array([by[(*k, "tiny_rescnn")]["miou"] for k in keys])
    vs_tiny = {}
    for m in CONFIGS:
        ours = np.array([by[(*k, m)]["miou"] for k in keys])
        sig = paired_significance(ours, tiny, seed=7, bootstraps=2000)  # tiny - ours
        vs_tiny[m] = {"ours_minus_tiny": -sig.mean_delta, "p": sig.ttest_p}
    summary = {"dataset": dataset, "means": means, "vs_tiny": vs_tiny}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
