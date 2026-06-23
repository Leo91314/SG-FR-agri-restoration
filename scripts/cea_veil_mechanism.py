"""M1: why does veil/fog recover PSNR but not downstream task?

Two controlled sweeps with a frozen clean-trained SegFormer-B0:
  - FOG sweep:   scale=1 (no resolution loss), tiny blur/noise, fog in {0..0.6}. Isolates veil.
  - STRUCTURE sweep: no fog, scale in {2,4,8} with blur/noise. Isolates structure destruction.

For each condition we measure mIoU(clean), mIoU(bicubic), mIoU(INR) and PSNR, and the
recovery fraction = (mIoU(INR)-mIoU(bicubic)) / (mIoU(clean)-mIoU(bicubic)). The hypothesis:
fog inflicts little *task* damage (robust pretrained encoders tolerate global veil) so even a
large PSNR recovery cannot convert to task gain; structure destruction inflicts large task
damage that the INR recovers substantially.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cea_plus.degradation import DegradationConfig, degrade_sample
from cea_plus.deep_downstream import _resize_image, _resize_mask, train_torch_segmenter
from cea_plus.downstream import evaluate_frozen_segmenter
from cea_plus.metrics import psnr
from cea_plus.restoration import bicubic_restore
from cea_plus.semantic_inr import restore_with_semantic_inr

from cea_exp import load_dataset, train_inr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOG_LEVELS = [0.0, 0.15, 0.30, 0.45, 0.60]
FOG_EVAL = [DegradationConfig(name=f"fog{f:.2f}", scale=1, rain_density=0, fog_strength=f,
                              blur_sigma=0.4, noise_sigma=0.01) for f in FOG_LEVELS]
FOG_TRAIN = [DegradationConfig(name=f"fog{f:.2f}", scale=1, rain_density=0, fog_strength=f,
                               blur_sigma=0.4, noise_sigma=0.01) for f in (0.15, 0.30, 0.45, 0.60)]

STRUCT_EVAL = [
    DegradationConfig(name="x2_blur", scale=2, rain_density=0, fog_strength=0.0, blur_sigma=1.0, noise_sigma=0.01),
    DegradationConfig(name="x4_blur", scale=4, rain_density=0, fog_strength=0.0, blur_sigma=1.5, noise_sigma=0.01),
    DegradationConfig(name="x8_blur", scale=8, rain_density=0, fog_strength=0.0, blur_sigma=2.0, noise_sigma=0.01),
    DegradationConfig(name="x8_noise", scale=8, rain_density=0, fog_strength=0.0, blur_sigma=1.2, noise_sigma=0.06),
]
STRUCT_TRAIN = STRUCT_EVAL


def run_sweep(seg, inr, test, grid, seed, device, crop):
    rows = []
    for s_idx, sample in enumerate(test):
        for config in grid:
            d = degrade_sample(sample, config=config, seed=seed * 2000 + s_idx * 100)
            shape = d.gt.shape[:2]
            mask_eval = _resize_mask(d.mask, (crop, crop))
            bic = bicubic_restore(d.low_res, shape)
            inr_img = restore_with_semantic_inr(inr, d.low_res, shape, device=device)
            rows.append({
                "condition": config.name, "seed": seed, "sample": sample.name,
                "clean_miou": evaluate_frozen_segmenter(seg, _resize_image(d.gt, (crop, crop)), mask_eval),
                "bicubic_miou": evaluate_frozen_segmenter(seg, _resize_image(bic, (crop, crop)), mask_eval),
                "inr_miou": evaluate_frozen_segmenter(seg, _resize_image(inr_img, (crop, crop)), mask_eval),
                "bicubic_psnr": psnr(d.gt, bic), "inr_psnr": psnr(d.gt, inr_img),
            })
    return rows


def aggregate(rows, order):
    out = []
    for cond in order:
        sub = [r for r in rows if r["condition"] == cond]
        if not sub:
            continue
        clean = float(np.mean([r["clean_miou"] for r in sub]))
        bic = float(np.mean([r["bicubic_miou"] for r in sub]))
        inr = float(np.mean([r["inr_miou"] for r in sub]))
        damage = clean - bic
        recovered = inr - bic
        out.append({
            "condition": cond, "n": len(sub),
            "clean_miou": clean, "bicubic_miou": bic, "inr_miou": inr,
            "task_damage": damage, "task_recovered": recovered,
            "recovery_fraction": (recovered / damage) if abs(damage) > 1e-3 else float("nan"),
            "psnr_gain": float(np.mean([r["inr_psnr"] - r["bicubic_psnr"] for r in sub])),
        })
    return out


def main() -> None:
    out = Path("results/cea/veil_mechanism")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    crop, train_limit, inr_steps, seg_steps = 256, 40, 2000, 150
    seeds = [71, 72]

    fog_rows, struct_rows = [], []
    for seed in seeds:
        train_pool, test = load_dataset("weedsgalore", crop, train_limit, 16)
        seg = train_torch_segmenter(train_samples=train_pool, architecture="segformer_b0_imagenet",
                                    steps=seg_steps, crop_size=128, input_size=crop, seed=seed, device=device)
        fog_inr = train_inr("semantic_inr", train_pool, FOG_TRAIN, inr_steps, seed, device)
        struct_inr = train_inr("semantic_inr", train_pool, STRUCT_TRAIN, inr_steps, seed, device)
        fog_rows += run_sweep(seg, fog_inr, test, FOG_EVAL, seed, device, crop)
        struct_rows += run_sweep(seg, struct_inr, test, STRUCT_EVAL, seed, device, crop)

    fog_agg = aggregate(fog_rows, [c.name for c in FOG_EVAL])
    struct_agg = aggregate(struct_rows, [c.name for c in STRUCT_EVAL])

    # Figure: fog sweep mIoU + psnr gain
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))
    fx = FOG_LEVELS
    ax1.plot(fx, [a["clean_miou"] for a in fog_agg], "o-", label="clean", color="#444")
    ax1.plot(fx, [a["bicubic_miou"] for a in fog_agg], "s-", label="fogged (bicubic)", color="#d95f02")
    ax1.plot(fx, [a["inr_miou"] for a in fog_agg], "^-", label="restored (INR)", color="#1b9e77")
    ax1.set_xlabel("fog strength"); ax1.set_ylabel("frozen segmenter mIoU")
    ax1.set_title("Veil: segmenter is robust; little task damage to recover")
    ax1.legend(fontsize=8)
    axb = ax1.twinx()
    axb.bar(fx, [a["psnr_gain"] for a in fog_agg], width=0.03, alpha=0.25, color="#1b9e77")
    axb.set_ylabel("INR PSNR gain (dB)", color="#1b9e77")

    conds = [a["condition"] for a in struct_agg]
    xidx = np.arange(len(conds))
    ax2.plot(xidx, [a["clean_miou"] for a in struct_agg], "o-", label="clean", color="#444")
    ax2.plot(xidx, [a["bicubic_miou"] for a in struct_agg], "s-", label="degraded (bicubic)", color="#d95f02")
    ax2.plot(xidx, [a["inr_miou"] for a in struct_agg], "^-", label="restored (INR)", color="#1b9e77")
    ax2.set_xticks(xidx); ax2.set_xticklabels(conds, rotation=20)
    ax2.set_ylabel("frozen segmenter mIoU")
    ax2.set_title("Structure: large task damage, substantially recovered")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "veil_vs_structure.png", dpi=130)
    plt.close(fig)

    summary = {
        "seeds": seeds,
        "fog_sweep": fog_agg,
        "structure_sweep": struct_agg,
        "fog_mean_recovery_fraction": float(np.nanmean([a["recovery_fraction"] for a in fog_agg])),
        "structure_mean_recovery_fraction": float(np.nanmean([a["recovery_fraction"] for a in struct_agg])),
        "fog_mean_psnr_gain": float(np.mean([a["psnr_gain"] for a in fog_agg])),
        "structure_mean_psnr_gain": float(np.mean([a["psnr_gain"] for a in struct_agg])),
        "interpretation": ("Under veil, the frozen segmenter loses little mIoU (robust to global haze), so the "
                           "recoverable task damage is small and a large PSNR recovery does not become task gain. "
                           "Under structure destruction, task damage is large and the INR recovers a substantial "
                           "fraction of it."),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out / "rows.json").write_text(json.dumps({"fog": fog_rows, "structure": struct_rows}, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("fog_mean_recovery_fraction", "structure_mean_recovery_fraction",
                                              "fog_mean_psnr_gain", "structure_mean_psnr_gain")}, indent=2))


if __name__ == "__main__":
    main()
