"""Phase 6: semantic-source robustness.

Does the task gain survive when the semantic supervision used to train the INR is NOT
ground truth, but a pseudo-mask from a clean-trained SegFormer guide? We compare:
  - gt_mask       : semantic head supervised by GT crop mask (paper default, training-only)
  - pseudo_mask   : semantic head supervised by a frozen SegFormer guide's pseudo-mask
  - no_semantic   : semantic supervision + semantic-driven detail disabled

Evaluation uses an INDEPENDENT clean-trained frozen SegFormer segmenter (different seed),
so no metric/guide leakage. GT masks are used only for INR training (gt_mask arm) and for
the final mIoU scoring, never given to the restorer at inference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cea_plus.degradation import degrade_sample
from cea_plus.deep_downstream import _resize_image, _resize_mask, train_torch_segmenter
from cea_plus.downstream import evaluate_frozen_segmenter
from cea_plus.restoration import bicubic_restore
from cea_plus.semantic_inr import TinySemanticINR, restore_with_semantic_inr, train_semantic_inr_steps
from cea_plus.statistics import paired_significance

from cea_exp import PLANS, center_crop, load_dataset


def build_batches_with_mask_source(train_pool, plan, seed, guide=None):
    """If guide is None, use GT mask; else use guide pseudo-mask on bicubic-upsampled low_res."""
    batches = []
    for s_idx, sample in enumerate(train_pool):
        for c_idx, config in enumerate(plan):
            d = degrade_sample(sample, config=config, seed=seed * 1000 + s_idx * 100 + c_idx)
            if guide is None:
                mask = d.mask
            else:
                up = bicubic_restore(d.low_res, d.gt.shape[:2])
                mask = guide.predict_mask(up).astype(np.float32)
            batches.append((d.low_res, d.gt, mask))
    return batches


def train_inr_from_batches(batches, steps, seed, device, semantic=True):
    torch.manual_seed(seed)
    model = TinySemanticINR(
        hidden_channels=48, base_sharpen_strength=0.0,
        structure_residual_scale=0.6, texture_residual_scale=0.2,
        semantic_detail_boost=0.30 if semantic else 0.0,
    )
    train_semantic_inr_steps(
        model, batches=batches, steps=steps, learning_rate=1e-3, task_loss_weight=0.0,
        semantic_loss_weight=0.03 if semantic else 0.0, frequency_loss_weight=0.20,
        seed=seed, device=device,
    )
    return model


def main() -> None:
    out = Path("results/cea/phase6_semantic_source")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    crop, train_limit, inr_steps, seg_steps = 256, 40, 1500, 150
    seeds = [71, 72]
    plan = PLANS["structure"]()

    rows = []
    for seed in seeds:
        train_pool, test = load_dataset("weedsgalore", crop, train_limit, None)
        # independent frozen downstream evaluator (different seed than the guide)
        evaluator = train_torch_segmenter(train_samples=train_pool, architecture="segformer_b0_imagenet",
                                          steps=seg_steps, crop_size=128, input_size=crop,
                                          seed=seed + 5000, device=device)
        # clean-trained guide used to produce pseudo-masks (no GT at inference)
        guide = train_torch_segmenter(train_samples=train_pool, architecture="segformer_b0_imagenet",
                                      steps=seg_steps, crop_size=128, input_size=crop,
                                      seed=seed + 1000, device=device)

        arms = {
            "gt_mask": train_inr_from_batches(build_batches_with_mask_source(train_pool, plan, seed),
                                              inr_steps, seed, device, semantic=True),
            "pseudo_mask": train_inr_from_batches(build_batches_with_mask_source(train_pool, plan, seed, guide=guide),
                                                  inr_steps, seed, device, semantic=True),
            "no_semantic": train_inr_from_batches(build_batches_with_mask_source(train_pool, plan, seed),
                                                  inr_steps, seed, device, semantic=False),
        }

        for s_idx, sample in enumerate(test):
            for c_idx, config in enumerate(plan):
                d = degrade_sample(sample, config=config, seed=seed * 2000 + s_idx * 100 + c_idx)
                shape = d.gt.shape[:2]
                mask_eval = _resize_mask(d.mask, (crop, crop))
                key = (seed, sample.name, d.degradation_name, d.scale)
                bic = _resize_image(bicubic_restore(d.low_res, shape), (crop, crop))
                rows.append({"key": key, "arm": "bicubic",
                             "miou": evaluate_frozen_segmenter(evaluator, bic, mask_eval)})
                for arm, model in arms.items():
                    img = _resize_image(restore_with_semantic_inr(model, d.low_res, shape, device=device), (crop, crop))
                    rows.append({"key": key, "arm": arm,
                                 "miou": evaluate_frozen_segmenter(evaluator, img, mask_eval)})

    by = {(r["key"], r["arm"]): r["miou"] for r in rows}
    keys = sorted({r["key"] for r in rows})
    summary = {"seeds": seeds, "plan": "structure", "n_samples": len(keys), "delta_vs_bicubic": {}}
    for arm in ("gt_mask", "pseudo_mask", "no_semantic"):
        base = np.array([by[(k, "bicubic")] for k in keys])
        prop = np.array([by[(k, arm)] for k in keys])
        sig = paired_significance(base, prop, seed=73, bootstraps=2000)
        summary["delta_vs_bicubic"][arm] = {
            "mean_bicubic": float(base.mean()), "mean_arm": float(prop.mean()),
            "mean_delta": sig.mean_delta, "ttest_p": sig.ttest_p, "wilcoxon_p": sig.wilcoxon_p,
            "ci_low": sig.bootstrap_ci_low, "ci_high": sig.bootstrap_ci_high,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
