"""WeedsGalore date-disjoint sanity check (supplementary, does not replace main results).

Official WeedsGalore splits share all four capture dates across train/test. Here we hold out
later dates for test to reduce temporal overlap:
  train dates: 2023-05-25, 2023-05-30
  test dates:  2023-06-06, 2023-06-15

Reports SG-FR vs bicubic Delta mIoU with image-level bootstrap CI under structure+composite regimes.
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

TRAIN_DATES = {"2023-05-25", "2023-05-30"}
TEST_DATES = {"2023-06-06", "2023-06-15"}
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
    out = Path("results/cea/date_disjoint_sanity")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    crop, inr_steps, seg_steps = 256, 1500, 150
    seeds = [71, 72]
    plan = structure_plan() + composite_plan()

    train_raw = load_weedsgalore_dataset(ROOT, split="train", date_allowlist=TRAIN_DATES)
    test_raw = load_weedsgalore_dataset(ROOT, split="test", date_allowlist=TEST_DATES)
    train_pool = [center_crop(s, crop) for s in train_raw]
    test = [center_crop(s, crop) for s in test_raw]
    print(f"date-disjoint: train={len(train_pool)} test={len(test)} images", flush=True)

    rows = []
    for seed in seeds:
        seg = train_torch_segmenter(
            train_samples=train_pool, architecture="segformer_b0_imagenet",
            steps=seg_steps, crop_size=128, input_size=crop, seed=seed, device=device,
        )
        inr = train_inr("semantic_inr", train_pool, plan, inr_steps, seed, device)
        for s_idx, sample in enumerate(test):
            for config in plan:
                d = degrade_sample(sample, config=config, seed=seed * 2000 + s_idx * 100)
                shape = d.gt.shape[:2]
                mask_eval = _resize_mask(d.mask, (crop, crop))
                bic = _resize_image(bicubic_restore(d.low_res, shape), (crop, crop))
                res = _resize_image(restore_with_semantic_inr(inr, d.low_res, shape, device=device), (crop, crop))
                key = (seed, sample.name, config.name)
                rows.append({"key": key, "bicubic": evaluate_frozen_segmenter(seg, bic, mask_eval),
                             "sgfr": evaluate_frozen_segmenter(seg, res, mask_eval)})

    keys = [tuple(r["key"]) for r in rows]
    base = np.array([r["bicubic"] for r in rows])
    prop = np.array([r["sgfr"] for r in rows])
    sig = paired_significance(base, prop, seed=73, bootstraps=2000)
    mean, lo, hi, n_img = image_level_ci(base, prop, keys)

    summary = {
        "train_dates": sorted(TRAIN_DATES),
        "test_dates": sorted(TEST_DATES),
        "n_train_images": len(train_pool),
        "n_test_images": len(test),
        "n_patch_pairs": len(rows),
        "seeds": seeds,
        "plan": "structure+composite",
        "evaluator": "segformer_b0_imagenet",
        "mean_bicubic": float(base.mean()),
        "mean_sgfr": float(prop.mean()),
        "mean_delta": sig.mean_delta,
        "patch_ci_low": sig.bootstrap_ci_low,
        "patch_ci_high": sig.bootstrap_ci_high,
        "image_level_delta": mean,
        "image_level_ci_low": lo,
        "image_level_ci_high": hi,
        "n_image": n_img,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
