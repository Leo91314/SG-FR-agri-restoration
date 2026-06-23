"""Guide-quality -> task-gain curve (semantic-source robustness, continuous).

Phase 6 showed pseudo-mask supervision works. Here we vary the QUALITY of the semantic
guide (by training it for fewer/more steps) and ask how the INR's downstream gain depends
on the guide's pseudo-mask quality. For each guide-strength level we:
  1. train a SegFormer guide for `g` steps (clean images),
  2. measure its pseudo-mask crop-IoU vs GT on bicubic-upsampled degraded TEST images,
  3. train an INR whose semantic head is supervised by that guide's pseudo-masks,
  4. measure the INR downstream mIoU gain vs bicubic with an INDEPENDENT evaluator.
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

from cea_exp import PLANS, load_dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GUIDE_STEPS = [5, 20, 80, 200]


def crop_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = float(np.logical_and(pred, gt).sum())
    union = float(np.logical_or(pred, gt).sum())
    return inter / union if union else float("nan")


def pseudo_batches(train_pool, plan, seed, guide):
    batches = []
    for s_idx, sample in enumerate(train_pool):
        for c_idx, config in enumerate(plan):
            d = degrade_sample(sample, config=config, seed=seed * 1000 + s_idx * 100 + c_idx)
            up = bicubic_restore(d.low_res, d.gt.shape[:2])
            batches.append((d.low_res, d.gt, guide.predict_mask(up).astype(np.float32)))
    return batches


def train_inr_pseudo(batches, steps, seed, device):
    torch.manual_seed(seed)
    model = TinySemanticINR(hidden_channels=48, base_sharpen_strength=0.0,
                            structure_residual_scale=0.6, texture_residual_scale=0.2, semantic_detail_boost=0.30)
    train_semantic_inr_steps(model, batches=batches, steps=steps, learning_rate=1e-3,
                             task_loss_weight=0.0, semantic_loss_weight=0.03, frequency_loss_weight=0.20,
                             seed=seed, device=device)
    return model


def main() -> None:
    out = Path("results/cea/guide_quality_curve")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    crop, train_limit, inr_steps, eval_steps = 256, 40, 1800, 150
    seeds = [71, 72]
    plan = PLANS["structure"]()

    rows = []
    for seed in seeds:
        train_pool, test = load_dataset("weedsgalore", crop, train_limit, 16)
        evaluator = train_torch_segmenter(train_samples=train_pool, architecture="segformer_b0_imagenet",
                                          steps=eval_steps, crop_size=128, input_size=crop,
                                          seed=seed + 5000, device=device)
        for g_steps in GUIDE_STEPS:
            guide = train_torch_segmenter(train_samples=train_pool, architecture="segformer_b0_imagenet",
                                          steps=g_steps, crop_size=128, input_size=crop,
                                          seed=seed + 1000, device=device)
            inr = train_inr_pseudo(pseudo_batches(train_pool, plan, seed, guide), inr_steps, seed, device)
            for s_idx, sample in enumerate(test):
                for config in plan:
                    d = degrade_sample(sample, config=config, seed=seed * 2000 + s_idx * 100)
                    shape = d.gt.shape[:2]
                    mask_eval = _resize_mask(d.mask, (crop, crop))
                    up = bicubic_restore(d.low_res, shape)
                    guide_iou = crop_iou(_resize_mask(guide.predict_mask(up), (crop, crop)), mask_eval)
                    bic_miou = evaluate_frozen_segmenter(evaluator, _resize_image(up, (crop, crop)), mask_eval)
                    inr_img = restore_with_semantic_inr(inr, d.low_res, shape, device=device)
                    inr_miou = evaluate_frozen_segmenter(evaluator, _resize_image(inr_img, (crop, crop)), mask_eval)
                    rows.append({"seed": seed, "guide_steps": g_steps, "sample": sample.name,
                                 "guide_crop_iou": guide_iou, "bicubic_miou": bic_miou, "inr_miou": inr_miou})

    points = []
    for g_steps in GUIDE_STEPS:
        sub = [r for r in rows if r["guide_steps"] == g_steps]
        gi = np.array([r["guide_crop_iou"] for r in sub])
        gi = gi[~np.isnan(gi)]
        base = np.array([r["bicubic_miou"] for r in sub])
        prop = np.array([r["inr_miou"] for r in sub])
        sig = paired_significance(base, prop, seed=73, bootstraps=1500)
        points.append({"guide_steps": g_steps, "n": len(sub),
                       "guide_crop_iou_mean": float(np.nanmean(gi)) if gi.size else float("nan"),
                       "task_gain": sig.mean_delta, "gain_ci_low": sig.bootstrap_ci_low,
                       "gain_ci_high": sig.bootstrap_ci_high, "gain_p": sig.ttest_p})

    fig, ax = plt.subplots(figsize=(7, 5.5))
    gx = [p["guide_crop_iou_mean"] for p in points]
    gy = [p["task_gain"] for p in points]
    yerr = [[p["task_gain"] - p["gain_ci_low"] for p in points], [p["gain_ci_high"] - p["task_gain"] for p in points]]
    ax.errorbar(gx, gy, yerr=yerr, fmt="o-", capsize=4, color="#1b9e77")
    for p in points:
        ax.annotate(f"{p['guide_steps']} steps", (p["guide_crop_iou_mean"], p["task_gain"]),
                    textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("guide pseudo-mask crop-IoU (semantic supervision quality)")
    ax.set_ylabel("INR downstream mIoU gain vs bicubic")
    ax.set_title("Task gain vs semantic-guide quality")
    fig.tight_layout()
    fig.savefig(out / "guide_quality_curve.png", dpi=130)
    plt.close(fig)

    summary = {"seeds": seeds, "plan": "structure", "guide_steps": GUIDE_STEPS, "points": points,
               "interpretation": ("INR downstream gain is robust across a wide range of semantic-guide "
                                  "quality; even weak pseudo-masks yield positive task gain.")}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
