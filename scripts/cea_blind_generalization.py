"""Phase 7: blind generalization to unseen degradation severities.

The INR is blind at inference (no degradation label is ever passed). We train it on the
structure regime (x8 blur sigma=2.0; x8 noise sigma=0.06) and then test on HELD-OUT
severities never seen in training:
  - interp : x8, blur 1.6 + noise 0.03   (interpolated severity / unseen combination)
  - extrap : x8, blur 2.5 + noise 0.08   (harsher than any training sample)
We report image quality (PSNR/SSIM) and downstream deep mIoU vs bicubic on these unseen
degradations. Real captured-UAV blind degradation is unavailable here and is documented as
a limitation; this synthetic held-out test is the in-scope generalization evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cea_plus.degradation import DegradationConfig, degrade_sample
from cea_plus.deep_downstream import _resize_image, _resize_mask, train_torch_segmenter
from cea_plus.downstream import evaluate_frozen_segmenter
from cea_plus.metrics import psnr, ssim_score
from cea_plus.restoration import bicubic_restore
from cea_plus.semantic_inr import restore_with_semantic_inr
from cea_plus.statistics import paired_significance

from cea_exp import PLANS, load_dataset, train_inr

HELD_OUT = {
    "interp": DegradationConfig(name="x8_interp", scale=8, rain_density=0, fog_strength=0.0, blur_sigma=1.6, noise_sigma=0.03),
    "extrap": DegradationConfig(name="x8_extrap", scale=8, rain_density=0, fog_strength=0.0, blur_sigma=2.5, noise_sigma=0.08),
}


def main() -> None:
    out = Path("results/cea/phase7_blind_generalization")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    crop, train_limit, inr_steps, seg_steps = 256, 48, 2000, 150
    seeds = [71, 72]
    train_plan = PLANS["structure"]()

    rows = []
    for seed in seeds:
        train_pool, test = load_dataset("weedsgalore", crop, train_limit, None)
        evaluator = train_torch_segmenter(train_samples=train_pool, architecture="segformer_b0_imagenet",
                                          steps=seg_steps, crop_size=128, input_size=crop,
                                          seed=seed + 5000, device=device)
        inr = train_inr("semantic_inr", train_pool, train_plan, inr_steps, seed, device)

        for held_name, config in HELD_OUT.items():
            for s_idx, sample in enumerate(test):
                d = degrade_sample(sample, config=config, seed=seed * 3000 + s_idx * 100)
                shape = d.gt.shape[:2]
                mask_eval = _resize_mask(d.mask, (crop, crop))
                key = (seed, held_name, sample.name)
                bic = bicubic_restore(d.low_res, shape)
                inr_img = restore_with_semantic_inr(inr, d.low_res, shape, device=device)
                rows.append({"key": key, "held": held_name, "method": "bicubic",
                             "psnr": psnr(d.gt, bic), "ssim": ssim_score(d.gt, bic),
                             "miou": evaluate_frozen_segmenter(evaluator, _resize_image(bic, (crop, crop)), mask_eval)})
                rows.append({"key": key, "held": held_name, "method": "semantic_inr",
                             "psnr": psnr(d.gt, inr_img), "ssim": ssim_score(d.gt, inr_img),
                             "miou": evaluate_frozen_segmenter(evaluator, _resize_image(inr_img, (crop, crop)), mask_eval)})

    summary = {"seeds": seeds, "train_regime": "structure", "held_out": list(HELD_OUT), "delta_vs_bicubic": {}}
    for held in HELD_OUT:
        sub = [r for r in rows if r["held"] == held]
        keys = sorted({r["key"] for r in sub})
        by = {(r["key"], r["method"]): r for r in sub}
        entry = {}
        for met in ("miou", "psnr", "ssim"):
            base = np.array([by[(k, "bicubic")][met] for k in keys])
            prop = np.array([by[(k, "semantic_inr")][met] for k in keys])
            sig = paired_significance(base, prop, seed=73, bootstraps=2000)
            entry[met] = {"mean_bicubic": float(base.mean()), "mean_inr": float(prop.mean()),
                          "mean_delta": sig.mean_delta, "ttest_p": sig.ttest_p,
                          "ci_low": sig.bootstrap_ci_low, "ci_high": sig.bootstrap_ci_high}
        summary["delta_vs_bicubic"][held] = entry
    summary["limitation"] = ("Real captured-UAV blind degradations are unavailable; generalization is "
                             "demonstrated on synthetic held-out severities (interp/extrap) unseen in training.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
