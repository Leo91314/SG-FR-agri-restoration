"""Route A verification: does a properly-trained, no-leak Semantic INR beat bicubic
on a frozen deep downstream segmenter under heavier degradation?

Strict protocol (audited):
- INR inference receives only low_res + output_shape (no GT mask, no degradation name).
- The downstream segmenter is trained on CLEAN train images only, then frozen.
- The INR is trained with reconstruction + structure + frequency + semantic supervision.
  The green-excess task loss is OFF (task_loss_weight=0) to avoid metric coupling.
- The ONLY headline claim we accept is: INR vs bicubic, deep frozen mIoU, paired-significant positive.

Run:
  PYTHONPATH=src python3 scripts/route_a_verify.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cea_plus.dataset import load_weedsgalore_dataset
from cea_plus.degradation import DegradationConfig, degrade_sample
from cea_plus.deep_downstream import (
    _resize_image,
    _resize_mask,
    train_torch_segmenter,
)
from cea_plus.downstream import evaluate_frozen_segmenter
from cea_plus.metrics import psnr, ssim_score
from cea_plus.restoration import bicubic_restore, uniform_sharp_restore
from cea_plus.semantic_inr import (
    TinySemanticINR,
    restore_with_semantic_inr,
    train_semantic_inr_steps,
)
from cea_plus.statistics import paired_significance
from cea_plus.synthesis import AgriSample


def center_crop(sample: AgriSample, size: int) -> AgriSample:
    h, w = sample.image.shape[:2]
    top = max((h - size) // 2, 0)
    left = max((w - size) // 2, 0)
    image = sample.image[top : top + size, left : left + size]
    mask = sample.mask[top : top + size, left : left + size]
    if image.shape[:2] != (size, size):
        image = _resize_image(image, (size, size))
        mask = _resize_mask(mask, (size, size))
    return AgriSample(image=image, mask=mask, name=sample.name)


def heavy_plan() -> list[DegradationConfig]:
    # Global-veil degradations (fog/light mixed). Deep nets are often robust to these.
    return [
        DegradationConfig(name="x4_heavyfog", scale=4, rain_density=0, fog_strength=0.45, blur_sigma=0.8, noise_sigma=0.01),
        DegradationConfig(name="x4_heavymixed", scale=4, rain_density=30, fog_strength=0.38, blur_sigma=0.8, noise_sigma=0.015, jpeg_quality=50),
    ]


def structure_plan() -> list[DegradationConfig]:
    # Structure-destroying degradations: very low resolution + heavy blur + heavy noise.
    # These remove task-relevant detail the segmenter cannot see through, giving
    # restoration genuine room to help the downstream task.
    return [
        DegradationConfig(name="x8_blur", scale=8, rain_density=0, fog_strength=0.0, blur_sigma=2.0, noise_sigma=0.01),
        DegradationConfig(name="x8_noise", scale=8, rain_density=0, fog_strength=0.0, blur_sigma=1.2, noise_sigma=0.06),
    ]


PLANS = {"veil": heavy_plan, "structure": structure_plan}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weedsgalore-root", default="data/external/weedsgalore-dataset")
    parser.add_argument("--out", default="results/route_a_verify_v1")
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--train-limit", type=int, default=24)
    parser.add_argument("--inr-steps", type=int, default=700)
    parser.add_argument("--seg-steps", type=int, default=120)
    parser.add_argument("--hidden", type=int, default=48)
    parser.add_argument("--structure-residual", type=float, default=0.6)
    parser.add_argument("--texture-residual", type=float, default=0.2)
    parser.add_argument("--semantic-detail", type=float, default=0.3)
    parser.add_argument("--seg-arch", default="segformer_b0_imagenet")
    parser.add_argument("--plan-kind", default="veil", choices=list(PLANS))
    parser.add_argument("--seed", type=int, default=73)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    root = Path(args.weedsgalore_root)
    train_pool = [center_crop(s, args.crop) for s in load_weedsgalore_dataset(root, split="train", limit=args.train_limit)]
    test = [center_crop(s, args.crop) for s in load_weedsgalore_dataset(root, split="test")]
    plan = PLANS[args.plan_kind]()

    # 1) Frozen deep segmenter trained on CLEAN train images only.
    segmenter = train_torch_segmenter(
        train_samples=train_pool,
        architecture=args.seg_arch,
        steps=args.seg_steps,
        crop_size=128,
        input_size=args.crop,
        seed=args.seed,
        device=device,
    )

    # 2) Train the no-leak Semantic INR (reconstruction + structure + frequency + semantic).
    inr_batches = []
    for s_idx, sample in enumerate(train_pool):
        for c_idx, config in enumerate(plan):
            degraded = degrade_sample(sample, config=config, seed=args.seed * 1000 + s_idx * 100 + c_idx)
            inr_batches.append((degraded.low_res, degraded.gt, degraded.mask))

    model = TinySemanticINR(
        hidden_channels=args.hidden,
        base_sharpen_strength=0.0,
        structure_residual_scale=args.structure_residual,
        texture_residual_scale=args.texture_residual,
        semantic_detail_boost=args.semantic_detail,
    )
    history = train_semantic_inr_steps(
        model,
        batches=inr_batches,
        steps=args.inr_steps,
        learning_rate=1e-3,
        task_loss_weight=0.0,  # green-excess metric coupling OFF
        semantic_loss_weight=0.03,
        seed=args.seed,
        device=device,
    )

    # 3) Evaluate bicubic / uniform_sharp / INR on the frozen deep segmenter + image quality.
    rows: list[dict[str, object]] = []
    for s_idx, sample in enumerate(test):
        for c_idx, config in enumerate(plan):
            degraded = degrade_sample(sample, config=config, seed=args.seed * 2000 + s_idx * 100 + c_idx)
            out_shape = degraded.gt.shape[:2]
            restorations = {
                "bicubic": bicubic_restore(degraded.low_res, out_shape),
                "uniform_sharp": uniform_sharp_restore(degraded.low_res, out_shape),
                "semantic_inr_no_leak": restore_with_semantic_inr(model, degraded.low_res, out_shape, device=device),
            }
            mask_eval = _resize_mask(degraded.mask, (args.crop, args.crop))
            for method, restored in restorations.items():
                img_eval = _resize_image(restored, (args.crop, args.crop))
                rows.append(
                    {
                        "sample": sample.name,
                        "degradation": degraded.degradation_name,
                        "scale": degraded.scale,
                        "method": method,
                        "deep_frozen_miou": evaluate_frozen_segmenter(segmenter, img_eval, mask_eval),
                        "psnr": psnr(degraded.gt, restored),
                        "ssim": ssim_score(degraded.gt, restored),
                    }
                )

    # 4) Paired stats: each method vs bicubic.
    def paired(method: str, metric: str):
        keys = sorted({(r["sample"], r["degradation"], r["scale"]) for r in rows})
        by = {(r["sample"], r["degradation"], r["scale"], r["method"]): r for r in rows}
        base, prop = [], []
        for k in keys:
            b = by.get((*k, "bicubic"))
            p = by.get((*k, method))
            if b and p:
                base.append(float(b[metric]))
                prop.append(float(p[metric]))
        sig = paired_significance(np.asarray(base), np.asarray(prop), seed=args.seed, bootstraps=2000)
        return sig

    methods = ["uniform_sharp", "semantic_inr_no_leak"]
    metrics = ["deep_frozen_miou", "psnr", "ssim"]
    means = {
        m: {met: float(np.mean([float(r[met]) for r in rows if r["method"] == m])) for met in metrics}
        for m in ["bicubic", *methods]
    }
    deltas = {}
    for m in methods:
        deltas[m] = {}
        for met in metrics:
            sig = paired(m, met)
            deltas[m][met] = {
                "mean_delta": sig.mean_delta,
                "ttest_p": sig.ttest_p,
                "ci_low": sig.bootstrap_ci_low,
                "ci_high": sig.bootstrap_ci_high,
            }

    # Per-degradation deep mIoU delta vs bicubic for the INR.
    by_deg = {}
    for deg in sorted({str(r["degradation"]) for r in rows}):
        pairs = {}
        for r in rows:
            if str(r["degradation"]) == deg:
                pairs.setdefault((r["sample"], r["scale"]), {})[r["method"]] = float(r["deep_frozen_miou"])
        vals = [v["semantic_inr_no_leak"] - v["bicubic"] for v in pairs.values() if "bicubic" in v and "semantic_inr_no_leak" in v]
        by_deg[deg] = {"n": len(vals), "inr_minus_bicubic_deep_miou": float(np.mean(vals)) if vals else float("nan")}

    payload = {
        "device": str(device),
        "test_pairs": len([r for r in rows if r["method"] == "bicubic"]),
        "inr_steps": args.inr_steps,
        "structure_residual": args.structure_residual,
        "texture_residual": args.texture_residual,
        "semantic_detail": args.semantic_detail,
        "initial_reconstruction_loss": history[0]["reconstruction_loss"],
        "final_reconstruction_loss": history[-1]["reconstruction_loss"],
        "means": means,
        "delta_vs_bicubic": deltas,
        "by_degradation_inr_vs_bicubic_deep_miou": by_deg,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    import csv

    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample", "degradation", "scale", "method", "deep_frozen_miou", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
