"""T4: robustness to a realistic, held-out degradation distribution + real-capture gap probe.

We have no real captured UAV-degraded frames on disk, so the honest proxy is to test generalization
to a *realistic, randomized* degradation chain (BSRGAN-flavored: random blur, Poisson+Gaussian sensor
noise, double JPEG, mild haze) that the restorer never saw during training (it is trained on the
fixed structure plan). We score downstream frozen-segmenter mIoU plus full-reference (PSNR/SSIM) and
no-reference (BRISQUE) quality. The no-reference channel is the exact path that would consume real
frames once available, so this also validates that pipeline end-to-end.
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import piq
from PIL import Image
from scipy import ndimage

from cea_plus.degradation import DegradationConfig, DegradedSample
from cea_plus.deep_downstream import _resize_image, _resize_mask, train_torch_segmenter
from cea_plus.downstream import evaluate_frozen_segmenter
from cea_plus.metrics import psnr, ssim_score
from cea_plus.restoration import bicubic_restore
from cea_plus.semantic_inr import restore_with_semantic_inr
from cea_plus.statistics import paired_significance

from cea_exp import load_dataset, train_inr, train_tiny_rescnn, restore_tiny_rescnn

# training distribution (fixed structure plan, x4) -- the model only ever sees these
TRAIN_PLAN = [
    DegradationConfig(name="x4_blur", scale=4, rain_density=0, fog_strength=0.0, blur_sigma=1.5, noise_sigma=0.01),
    DegradationConfig(name="x4_noise", scale=4, rain_density=0, fog_strength=0.0, blur_sigma=0.8, noise_sigma=0.05),
]


def _jpeg(img: np.ndarray, q: int) -> np.ndarray:
    buf = BytesIO()
    Image.fromarray(np.uint8(np.clip(img, 0, 1) * 255)).save(buf, format="JPEG", quality=int(q))
    buf.seek(0)
    return (np.asarray(Image.open(buf).convert("RGB")).astype(np.float32) / 255).clip(0, 1)


def realistic_degrade(sample, seed: int, scale: int = 4) -> DegradedSample:
    """Per-image randomized realistic chain, disjoint from the fixed training configs."""
    rng = np.random.default_rng(seed)
    img = sample.image
    h, w = img.shape[:2]
    # anisotropic-ish defocus+motion proxy: two gaussian passes with random sigmas
    s1 = float(rng.uniform(0.8, 2.2))
    s2 = float(rng.uniform(0.4, 1.4))
    blurred = ndimage.gaussian_filter(img, sigma=(s1, s2, 0.0))
    low = Image.fromarray(np.uint8(np.clip(blurred, 0, 1) * 255)).resize((w // scale, h // scale), Image.BICUBIC)
    low = np.asarray(low).astype(np.float32) / 255
    # mild atmospheric haze (random)
    fog = float(rng.uniform(0.0, 0.20))
    low = low * (1 - fog) + fog
    # double JPEG with random qualities (realistic transcoding)
    low = _jpeg(low, int(rng.integers(55, 80)))
    low = _jpeg(low, int(rng.integers(35, 60)))
    # Poisson shot noise + Gaussian read noise
    peak = float(rng.uniform(20, 90))
    low = rng.poisson(np.clip(low, 0, 1) * peak) / peak
    low = low + rng.normal(0, float(rng.uniform(0.01, 0.04)), low.shape)
    return DegradedSample(low_res=np.clip(low, 0, 1).astype(np.float32), gt=img, mask=sample.mask,
                          degradation_name="realistic_heldout", scale=scale)


def brisque_of(img, dev):
    x = torch.from_numpy(np.clip(img, 0, 1).transpose(2, 0, 1)[None]).float().to(dev)
    with torch.no_grad():
        return float(piq.brisque(x, data_range=1.0, reduction="none").item())


def main() -> None:
    out = Path("results/cea/realistic_degradation")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    cpu = torch.device("cpu")
    crop, train_limit, inr_steps, seg_steps = 256, 40, 2000, 150
    seeds = [81, 82]

    rows = []
    for seed in seeds:
        train_pool, test = load_dataset("weedsgalore", crop, train_limit, 16)
        seg = train_torch_segmenter(train_samples=train_pool, architecture="segformer_b0_imagenet",
                                    steps=seg_steps, crop_size=128, input_size=crop, seed=seed, device=dev)
        inr = train_inr("semantic_inr", train_pool, TRAIN_PLAN, inr_steps, seed, dev)
        rescnn = train_tiny_rescnn(train_pool, TRAIN_PLAN, inr_steps, seed, dev)
        for s_idx, sample in enumerate(test):
            d = realistic_degrade(sample, seed=seed * 3000 + s_idx)
            shape = d.gt.shape[:2]
            mask_eval = _resize_mask(d.mask, (crop, crop))
            restorations = {
                "bicubic": bicubic_restore(d.low_res, shape),
                "tiny_rescnn": restore_tiny_rescnn(rescnn, d.low_res, shape, dev),
                "semantic_inr": restore_with_semantic_inr(inr, d.low_res, shape, device=dev),
            }
            for m, img in restorations.items():
                rows.append({"seed": seed, "sample": sample.name, "method": m,
                             "miou": evaluate_frozen_segmenter(seg, _resize_image(img, (crop, crop)), mask_eval),
                             "psnr": psnr(d.gt, img), "ssim": ssim_score(d.gt, img),
                             "brisque": brisque_of(img, cpu)})

    methods = ["bicubic", "tiny_rescnn", "semantic_inr"]
    keys = sorted({(r["seed"], r["sample"]) for r in rows})
    by = {(r["seed"], r["sample"], r["method"]): r for r in rows}
    summary = {"seeds": seeds, "degradation": "realistic_heldout_x4", "means": {}, "delta_vs_bicubic": {}}
    for m in methods:
        vals = {met: np.array([by[(*k, m)][met] for k in keys]) for met in ("miou", "psnr", "ssim", "brisque")}
        summary["means"][m] = {met: float(v.mean()) for met, v in vals.items()}
    for m in methods:
        if m == "bicubic":
            continue
        entry = {}
        for met in ("miou", "psnr", "ssim", "brisque"):
            base = np.array([by[(*k, "bicubic")][met] for k in keys])
            prop = np.array([by[(*k, m)][met] for k in keys])
            sig = (paired_significance(prop, base, seed=83, bootstraps=1500) if met == "brisque"
                   else paired_significance(base, prop, seed=83, bootstraps=1500))
            entry[met] = {"mean_delta": sig.mean_delta, "ci_low": sig.bootstrap_ci_low,
                          "ci_high": sig.bootstrap_ci_high, "ttest_p": sig.ttest_p}
        summary["delta_vs_bicubic"][m] = entry

    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out / "rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
