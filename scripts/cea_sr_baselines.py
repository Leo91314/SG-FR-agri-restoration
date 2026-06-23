"""T1: strong pretrained SR baselines under the same no-leak protocol.

Adds Swin2SR (classical x4) and Swin2SR (real-world BSRGAN x4) -- a SOTA transformer SR family --
as restorers, compared at matched scale x4 against bicubic, Tiny ResCNN, and our Semantic-INR.
Pretrained SR expects a clean bicubically-downsampled LR; feeding our blur/noise/compound LR tests
robustness, which is exactly the agricultural-deployment question. We report downstream frozen
SegFormer-B0 mIoU plus full-reference (PSNR/SSIM) and no-reference (BRISQUE) quality.

Note: pretrained Swin2SR is fixed integer-scale (x4), so this study is run on x4 conditions only;
x8 cascading is out of scope and discussed as a limitation.
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import json
from pathlib import Path

import numpy as np
import torch
import piq
from transformers import Swin2SRForImageSuperResolution

from cea_plus.degradation import DegradationConfig, degrade_sample
from cea_plus.deep_downstream import _resize_image, _resize_mask, train_torch_segmenter
from cea_plus.downstream import evaluate_frozen_segmenter
from cea_plus.metrics import psnr, ssim_score
from cea_plus.restoration import bicubic_restore
from cea_plus.semantic_inr import restore_with_semantic_inr
from cea_plus.statistics import paired_significance

from cea_exp import load_dataset, train_inr, train_tiny_rescnn, restore_tiny_rescnn

# x4 conditions only (pretrained SR is fixed-scale x4).
X4_PLAN = [
    DegradationConfig(name="x4_blur", scale=4, rain_density=0, fog_strength=0.0, blur_sigma=1.5, noise_sigma=0.01),
    DegradationConfig(name="x4_noise", scale=4, rain_density=0, fog_strength=0.0, blur_sigma=0.8, noise_sigma=0.05),
    DegradationConfig(name="x4_compound", scale=4, rain_density=20, fog_strength=0.25, blur_sigma=1.5, noise_sigma=0.04, jpeg_quality=45),
]

SR_MODELS = {
    "swin2sr_classical": "caidas/swin2SR-classical-sr-x4-64",
    "swin2sr_realworld": "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
}


def load_sr(model_id, device):
    m = Swin2SRForImageSuperResolution.from_pretrained(model_id).to(device).eval()
    return m


@torch.no_grad()
def sr_restore(model, low_res, output_shape, device):
    x = torch.from_numpy(np.clip(low_res, 0, 1).transpose(2, 0, 1)[None]).float().to(device)
    out = model(x).reconstruction.clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
    if out.shape[:2] != tuple(output_shape):
        out = _resize_image(out, tuple(output_shape))
    return out.astype(np.float32)


def brisque_of(img, device):
    x = torch.from_numpy(np.clip(img, 0, 1).transpose(2, 0, 1)[None]).float().to(device)
    with torch.no_grad():
        return float(piq.brisque(x, data_range=1.0, reduction="none").item())


def main() -> None:
    out = Path("results/cea/sr_baselines")
    out.mkdir(parents=True, exist_ok=True)
    train_dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sr_dev = torch.device("cpu")  # Swin2SR + BRISQUE on CPU for op-coverage safety
    crop, train_limit, inr_steps, seg_steps = 256, 40, 2000, 150
    seeds = [71, 72]

    sr_models = {name: load_sr(mid, sr_dev) for name, mid in SR_MODELS.items()}

    rows = []
    for seed in seeds:
        train_pool, test = load_dataset("weedsgalore", crop, train_limit, 16)
        seg = train_torch_segmenter(train_samples=train_pool, architecture="segformer_b0_imagenet",
                                    steps=seg_steps, crop_size=128, input_size=crop, seed=seed, device=train_dev)
        inr = train_inr("semantic_inr", train_pool, X4_PLAN, inr_steps, seed, train_dev)
        rescnn = train_tiny_rescnn(train_pool, X4_PLAN, inr_steps, seed, train_dev)
        for s_idx, sample in enumerate(test):
            for config in X4_PLAN:
                d = degrade_sample(sample, config=config, seed=seed * 2000 + s_idx * 100)
                shape = d.gt.shape[:2]
                mask_eval = _resize_mask(d.mask, (crop, crop))
                restorations = {
                    "bicubic": bicubic_restore(d.low_res, shape),
                    "tiny_rescnn": restore_tiny_rescnn(rescnn, d.low_res, shape, train_dev),
                    "semantic_inr": restore_with_semantic_inr(inr, d.low_res, shape, device=train_dev),
                }
                for name, model in sr_models.items():
                    restorations[name] = sr_restore(model, d.low_res, shape, sr_dev)
                for m, img in restorations.items():
                    rows.append({
                        "seed": seed, "sample": sample.name, "condition": config.name, "method": m,
                        "miou": evaluate_frozen_segmenter(seg, _resize_image(img, (crop, crop)), mask_eval),
                        "psnr": psnr(d.gt, img), "ssim": ssim_score(d.gt, img),
                        "brisque": brisque_of(img, sr_dev),
                    })

    methods = ["bicubic", "tiny_rescnn", "swin2sr_classical", "swin2sr_realworld", "semantic_inr"]
    keys = sorted({(r["seed"], r["sample"], r["condition"]) for r in rows})
    by = {(r["seed"], r["sample"], r["condition"], r["method"]): r for r in rows}

    summary = {"seeds": seeds, "scale": "x4", "conditions": [c.name for c in X4_PLAN],
               "sr_models": SR_MODELS, "means": {}, "delta_vs_bicubic": {}}
    for m in methods:
        vals = {met: np.array([by[(*k, m)][met] for k in keys if (*k, m) in by]) for met in ("miou", "psnr", "ssim", "brisque")}
        summary["means"][m] = {met: float(v.mean()) for met, v in vals.items()}
    for m in methods:
        if m == "bicubic":
            continue
        entry = {}
        for met in ("miou", "psnr", "ssim", "brisque"):
            base = np.array([by[(*k, "bicubic")][met] for k in keys])
            prop = np.array([by[(*k, m)][met] for k in keys])
            # for brisque lower is better: improvement = base - prop; else prop - base
            if met == "brisque":
                sig = paired_significance(prop, base, seed=73, bootstraps=1500)
            else:
                sig = paired_significance(base, prop, seed=73, bootstraps=1500)
            entry[met] = {"mean_delta": sig.mean_delta, "ci_low": sig.bootstrap_ci_low,
                          "ci_high": sig.bootstrap_ci_high, "ttest_p": sig.ttest_p}
        summary["delta_vs_bicubic"][m] = entry

    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out / "rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"means": summary["means"],
                      "miou_delta": {m: summary["delta_vs_bicubic"][m]["miou"] for m in methods if m != "bicubic"}},
                     indent=2))


if __name__ == "__main__":
    main()
