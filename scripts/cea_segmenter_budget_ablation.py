"""Segmenter training-budget ablation (WeedsGalore, official split).

Retrain frozen SegFormer-B0 at 150 / 500 / 1000 clean-training steps, then run the
standard SG-FR pipeline (structure+composite, 1500 INR steps) and report Delta mIoU
vs bicubic with image-level bootstrap CI (2 seeds).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cea_plus.dataset import load_weedsgalore_dataset
from cea_plus.degradation import degrade_sample
from cea_plus.deep_downstream import _resize_image, _resize_mask, train_torch_segmenter
from cea_plus.downstream import evaluate_frozen_segmenter
from cea_plus.restoration import bicubic_restore
from cea_plus.semantic_inr import restore_with_semantic_inr
from cea_plus.statistics import paired_significance

from cea_exp import center_crop, composite_plan, structure_plan, train_inr

SEG_STEPS_LIST = [150, 500, 1000]
SEEDS = [71, 72]
CROP = 256
INR_STEPS = 1500
ROOT = Path("data/external/weedsgalore-dataset")


def image_level_ci(base, prop, keys, seed=17, n_boot=3000):
    groups = defaultdict(list)
    for b, p, k in zip(base, prop, keys):
        groups[k[1]].append(p - b)
    delta = np.array([np.mean(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, delta.size, size=(n_boot, delta.size))
    means = delta[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(delta.mean()), float(lo), float(hi), int(delta.size)


def main() -> None:
    out = Path("results/cea/segmenter_budget_ablation")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    plan = structure_plan() + composite_plan()

    train_raw = load_weedsgalore_dataset(ROOT, split="train")
    test_raw = load_weedsgalore_dataset(ROOT, split="test")
    train_pool = [center_crop(s, CROP) for s in train_raw]
    test = [center_crop(s, CROP) for s in test_raw]

    summaries = []
    for seg_steps in SEG_STEPS_LIST:
        rows = []
        clean_miou = []
        for seed in SEEDS:
            seg = train_torch_segmenter(
                train_samples=train_pool, architecture="segformer_b0_imagenet",
                steps=seg_steps, crop_size=128, input_size=CROP, seed=seed, device=device,
            )
            for sample in test:
                mask_eval = _resize_mask(sample.mask, (CROP, CROP))
                clean_miou.append(evaluate_frozen_segmenter(seg, sample.image, mask_eval))
            inr = train_inr("semantic_inr", train_pool, plan, INR_STEPS, seed, device)
            for s_idx, sample in enumerate(test):
                for config in plan:
                    d = degrade_sample(sample, config=config, seed=seed * 2000 + s_idx * 100)
                    shape = d.gt.shape[:2]
                    mask_eval = _resize_mask(d.mask, (CROP, CROP))
                    bic = _resize_image(bicubic_restore(d.low_res, shape), (CROP, CROP))
                    res = _resize_image(restore_with_semantic_inr(inr, d.low_res, shape, device=device), (CROP, CROP))
                    key = (seed, sample.name, config.name)
                    rows.append({"key": key, "bicubic": evaluate_frozen_segmenter(seg, bic, mask_eval),
                                 "sgfr": evaluate_frozen_segmenter(seg, res, mask_eval)})

        keys = [tuple(r["key"]) for r in rows]
        base = np.array([r["bicubic"] for r in rows])
        prop = np.array([r["sgfr"] for r in rows])
        sig = paired_significance(base, prop, seed=73, bootstraps=2000)
        mean, lo, hi, n_img = image_level_ci(base, prop, keys)
        summaries.append({
            "seg_steps": seg_steps,
            "inr_steps": INR_STEPS,
            "plan": "structure+composite",
            "evaluator": "segformer_b0_imagenet",
            "seeds": SEEDS,
            "n_test_images": len(test),
            "n_patch_pairs": len(rows),
            "clean_test_miou_mean": float(np.mean(clean_miou)),
            "mean_bicubic": float(base.mean()),
            "mean_sgfr": float(prop.mean()),
            "mean_delta": sig.mean_delta,
            "patch_ci_low": sig.bootstrap_ci_low,
            "patch_ci_high": sig.bootstrap_ci_high,
            "image_level_delta": mean,
            "image_level_ci_low": lo,
            "image_level_ci_high": hi,
            "n_image": n_img,
        })
        print(f"seg_steps={seg_steps} delta={mean:.4f} CI=[{lo:.4f},{hi:.4f}]", flush=True)

    (out / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
