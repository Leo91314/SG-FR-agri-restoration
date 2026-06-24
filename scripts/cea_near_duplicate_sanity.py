"""Near-duplicate train--test overlap sensitivity (WeedsGalore + CoFly).

For each official test image, compute max SSIM vs all train images (256x256 center crop).
Re-run SG-FR vs bicubic under structure+composite with:
  - all test images
  - exclude test images whose max train SSIM >= 0.90
  - exclude test images whose max train SSIM >= 0.85

Reports image-level bootstrap 95% CI for Delta mIoU (SegFormer-B0, 2 seeds).
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cea_plus.dataset import load_cofly_dataset, load_weedsgalore_dataset
from cea_plus.degradation import degrade_sample
from cea_plus.deep_downstream import _resize_image, _resize_mask, train_torch_segmenter
from cea_plus.downstream import evaluate_frozen_segmenter
from cea_plus.metrics import ssim_score
from cea_plus.restoration import bicubic_restore
from cea_plus.semantic_inr import restore_with_semantic_inr
from cea_plus.statistics import paired_significance

from cea_exp import center_crop, composite_plan, structure_plan, train_inr

THRESHOLDS = [None, 0.90, 0.85]  # None = all test images
SEEDS = [71, 72]
CROP = 256
INR_STEPS = 1500
SEG_STEPS = 150


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


def load_pairs(dataset: str):
    if dataset == "weedsgalore":
        root = Path("data/external/weedsgalore-dataset")
        train = [center_crop(s, CROP) for s in load_weedsgalore_dataset(root, split="train")]
        test = [center_crop(s, CROP) for s in load_weedsgalore_dataset(root, split="test")]
        return train, test
    if dataset == "cofly":
        root = Path("data/external/cofly/CoFly-WeedDB")
        train = load_cofly_dataset(root, split="train", split_index=1, crop_size=CROP,
                                   crop_strategy="mask_center", crop_seed=41)
        test = load_cofly_dataset(root, split="test", split_index=1, crop_size=CROP,
                                  crop_strategy="mask_center", crop_seed=42)
        return train, test
    raise ValueError(dataset)


def max_train_ssim(test_sample, train_samples) -> float:
    best = -1.0
    for tr in train_samples:
        best = max(best, ssim_score(tr.image, test_sample.image))
    return best


def filter_test(test, train, threshold: float | None):
    if threshold is None:
        return test
    kept = []
    for s in test:
        if max_train_ssim(s, train) < threshold:
            kept.append(s)
    return kept


def run_filter(dataset: str, threshold: float | None, device: torch.device) -> dict:
    train, test_all = load_pairs(dataset)
    test = filter_test(test_all, train, threshold)
    plan = structure_plan() + composite_plan()
    rows = []
    for seed in SEEDS:
        seg = train_torch_segmenter(
            train_samples=train, architecture="segformer_b0_imagenet",
            steps=SEG_STEPS, crop_size=128, input_size=CROP, seed=seed, device=device,
        )
        inr = train_inr("semantic_inr", train, plan, INR_STEPS, seed, device)
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
    return {
        "dataset": dataset,
        "filter": "all" if threshold is None else f"ssim_lt_{threshold}",
        "threshold": threshold,
        "n_train": len(train),
        "n_test_total": len(test_all),
        "n_test_used": len(test),
        "n_patch_pairs": len(rows),
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


def main() -> None:
    out = Path("results/cea/near_duplicate_sanity")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    summaries = []
    for dataset in ("weedsgalore", "cofly"):
        train, test_all = load_pairs(dataset)
        ssims = [max_train_ssim(t, train) for t in test_all]
        print(f"{dataset}: n_test={len(test_all)} max_ssim mean={np.mean(ssims):.3f} "
              f"max={np.max(ssims):.3f}", flush=True)
        for thr in THRESHOLDS:
            print(f"  running filter={thr} ...", flush=True)
            summaries.append(run_filter(dataset, thr, device))

    (out / "summary.json").write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
