"""Unified CEA experiment driver (no-leak, task-oriented Semantic INR).

Covers plan phases 1-5: mechanism contrast, main downstream result, fair learned
baselines, ablations, and cross-dataset generalization.

Strict no-leak protocol (audited):
- Restoration inference receives only low_res + output_shape (no GT mask, no degradation name).
- Downstream segmenters are trained on CLEAN train images only, then frozen.
- INR training uses reconstruction+structure+frequency+semantic supervision; the
  green-excess task loss is OFF (task_loss_weight=0).
- Headline claim is ALWAYS: method vs bicubic, deep frozen mIoU, paired-significant positive.

Run examples:
  PYTHONPATH=src python3.12 scripts/cea_exp.py --tag phase2_main \
    --dataset weedsgalore --plan structure,composite \
    --methods bicubic,uniform_sharp,semantic_inr --archs segformer_b0_imagenet,deeplabv3plus_imagenet \
    --seeds 71,72,73
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cea_plus.dataset import load_cofly_dataset, load_cwfid_dataset, load_loveda_dataset, load_weedsgalore_dataset
from cea_plus.degradation import DegradationConfig, degrade_sample
from cea_plus.deep_baselines import TinyResCNN
from cea_plus.deep_downstream import _resize_image, _resize_mask, train_torch_segmenter
from cea_plus.downstream import evaluate_frozen_segmenter
from cea_plus.metrics import boundary_f_score, psnr, ssim_score
from cea_plus.restoration import bicubic_restore, uniform_sharp_restore
from cea_plus.semantic_inr import TinySemanticINR, restore_with_semantic_inr, train_semantic_inr_steps
from cea_plus.statistics import paired_significance
from cea_plus.synthesis import AgriSample


# ----- degradation plans (mapped to real agri-UAV causes) -------------------
def veil_plan() -> list[DegradationConfig]:
    # haze/illumination veil; global, deep nets often robust.
    return [
        DegradationConfig(name="x4_heavyfog", scale=4, rain_density=0, fog_strength=0.45, blur_sigma=0.8, noise_sigma=0.01),
        DegradationConfig(name="x4_heavymixed", scale=4, rain_density=30, fog_strength=0.38, blur_sigma=0.8, noise_sigma=0.015, jpeg_quality=50),
    ]


def structure_plan() -> list[DegradationConfig]:
    # high altitude -> very low resolution; defocus/motion -> blur; sensor -> noise.
    return [
        DegradationConfig(name="x8_blur", scale=8, rain_density=0, fog_strength=0.0, blur_sigma=2.0, noise_sigma=0.01),
        DegradationConfig(name="x8_noise", scale=8, rain_density=0, fog_strength=0.0, blur_sigma=1.2, noise_sigma=0.06),
    ]


def composite_plan() -> list[DegradationConfig]:
    # realistic compound: low-res + blur + noise + mild haze + compression.
    return [
        DegradationConfig(name="x4_compound", scale=4, rain_density=20, fog_strength=0.25, blur_sigma=1.5, noise_sigma=0.04, jpeg_quality=45),
        DegradationConfig(name="x8_compound", scale=8, rain_density=0, fog_strength=0.20, blur_sigma=1.8, noise_sigma=0.05, jpeg_quality=55),
    ]


PLANS = {"veil": veil_plan, "structure": structure_plan, "composite": composite_plan}


# ----- INR / learned-restorer method registry -------------------------------
INR_VARIANTS = {
    # full model
    "semantic_inr": dict(semantic_detail_boost=0.30, structure_residual_scale=0.6, texture_residual_scale=0.2,
                         semantic_loss_weight=0.03, frequency_loss_weight=0.20),
    # ablations
    "inr_no_semantic": dict(semantic_detail_boost=0.0, structure_residual_scale=0.6, texture_residual_scale=0.2,
                            semantic_loss_weight=0.0, frequency_loss_weight=0.20),
    "inr_no_texture": dict(semantic_detail_boost=0.30, structure_residual_scale=0.6, texture_residual_scale=0.0,
                           semantic_loss_weight=0.03, frequency_loss_weight=0.20),
    "inr_no_freq_loss": dict(semantic_detail_boost=0.30, structure_residual_scale=0.6, texture_residual_scale=0.2,
                             semantic_loss_weight=0.03, frequency_loss_weight=0.0),
}


def center_crop(sample: AgriSample, size: int) -> AgriSample:
    h, w = sample.image.shape[:2]
    top = max((h - size) // 2, 0)
    left = max((w - size) // 2, 0)
    image = sample.image[top:top + size, left:left + size]
    mask = sample.mask[top:top + size, left:left + size]
    if image.shape[:2] != (size, size):
        image = _resize_image(image, (size, size))
        mask = _resize_mask(mask, (size, size))
    return AgriSample(image=image, mask=mask, name=sample.name)


def load_dataset(name: str, crop: int, train_limit: int, test_limit: int | None):
    if name == "weedsgalore":
        root = Path("data/external/weedsgalore-dataset")
        train = [center_crop(s, crop) for s in load_weedsgalore_dataset(root, split="train", limit=train_limit)]
        test = [center_crop(s, crop) for s in load_weedsgalore_dataset(root, split="test", limit=test_limit)]
        return train, test
    if name == "cofly":
        root = Path("data/external/cofly/CoFly-WeedDB")
        train = load_cofly_dataset(root, split="train", split_index=1, limit=train_limit, crop_size=crop,
                                   crop_strategy="mask_center", crop_seed=41)
        test = load_cofly_dataset(root, split="test", split_index=1, limit=test_limit, crop_size=crop,
                                  crop_strategy="mask_center", crop_seed=42)
        return train, test
    if name == "cwfid":
        root = Path("data/external/cwfid")
        train = load_cwfid_dataset(root, split="train", limit=train_limit, crop_size=crop,
                                   crop_strategy="mask_center", crop_seed=41)
        test = load_cwfid_dataset(root, split="test", limit=test_limit, crop_size=crop,
                                  crop_strategy="mask_center", crop_seed=42)
        return train, test
    if name in ("loveda_rural", "loveda_urban"):
        domain = "Rural" if name.endswith("rural") else "Urban"
        samples = load_loveda_dataset(Path("data/external/loveda"), split="Val", domain=domain,
                                      target_labels=(7,), limit=(train_limit + (test_limit or 20) + 8),
                                      crop_size=crop, crop_strategy="random", crop_seed=41, require_mask=True)
        train = samples[:train_limit]
        test = samples[train_limit:train_limit + (test_limit or 20)]
        return train, test
    raise ValueError(f"unknown dataset {name}")


def build_batches(train_pool, plan, seed):
    batches = []
    for s_idx, sample in enumerate(train_pool):
        for c_idx, config in enumerate(plan):
            d = degrade_sample(sample, config=config, seed=seed * 1000 + s_idx * 100 + c_idx)
            batches.append((d.low_res, d.gt, d.mask))
    return batches


def train_inr(variant, train_pool, plan, steps, seed, device):
    cfg = INR_VARIANTS[variant]
    torch.manual_seed(seed)  # deterministic encoder init regardless of call order
    model = TinySemanticINR(
        hidden_channels=48,
        base_sharpen_strength=0.0,
        structure_residual_scale=cfg["structure_residual_scale"],
        texture_residual_scale=cfg["texture_residual_scale"],
        semantic_detail_boost=cfg["semantic_detail_boost"],
    )
    train_semantic_inr_steps(
        model, batches=build_batches(train_pool, plan, seed), steps=steps, learning_rate=1e-3,
        task_loss_weight=0.0, semantic_loss_weight=cfg["semantic_loss_weight"],
        frequency_loss_weight=cfg["frequency_loss_weight"], seed=seed, device=device,
    )
    return model


def train_tiny_rescnn(train_pool, plan, steps, seed, device):
    torch.manual_seed(seed)
    model = TinyResCNN(width=24).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batches = build_batches(train_pool, plan, seed)
    model.train()
    for step in range(steps):
        low, gt, _ = batches[step % len(batches)]
        up = bicubic_restore(low, gt.shape[:2])
        x = torch.from_numpy(up.transpose(2, 0, 1)[None]).float().to(device)
        target = torch.from_numpy(gt.transpose(2, 0, 1)[None]).float().to(device)
        out = model(x)
        loss = F.l1_loss(out, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


def restore_tiny_rescnn(model, low_res, shape, device):
    up = bicubic_restore(low_res, shape)
    with torch.no_grad():
        x = torch.from_numpy(up.transpose(2, 0, 1)[None]).float().to(device)
        out = model(x).clamp(0.0, 1.0).cpu().numpy()[0].transpose(1, 2, 0)
    return out.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--dataset", default="weedsgalore")
    p.add_argument("--plan", default="structure")  # comma list
    p.add_argument("--methods", default="bicubic,uniform_sharp,semantic_inr")  # comma list
    p.add_argument("--archs", default="segformer_b0_imagenet")  # comma list
    p.add_argument("--seeds", default="73")  # comma list
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--train-limit", type=int, default=48)
    p.add_argument("--test-limit", type=int, default=None)
    p.add_argument("--inr-steps", type=int, default=2000)
    p.add_argument("--seg-steps", type=int, default=150)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    out = Path("results/cea") / args.tag
    out.mkdir(parents=True, exist_ok=True)

    plans = []
    for name in args.plan.split(","):
        plans += [(name, c) for c in PLANS[name]()]
    methods = args.methods.split(",")
    archs = args.archs.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    plan_names = args.plan.split(",")
    rows: list[dict] = []
    for seed in seeds:
        train_pool, test = load_dataset(args.dataset, args.crop, args.train_limit, args.test_limit)
        # frozen downstream segmenters (clean-trained, shared across plans)
        segmenters = {
            arch: train_torch_segmenter(train_samples=train_pool, architecture=arch, steps=args.seg_steps,
                                        crop_size=128, input_size=args.crop, seed=seed, device=device)
            for arch in archs
        }
        # learned restorers are trained PER degradation regime (standard restoration protocol)
        for plan_name in plan_names:
            plan_configs = PLANS[plan_name]()
            trained = {}
            for m in methods:
                if m in INR_VARIANTS:
                    trained[m] = train_inr(m, train_pool, plan_configs, args.inr_steps, seed, device)
                elif m == "tiny_rescnn":
                    trained[m] = train_tiny_rescnn(train_pool, plan_configs, args.inr_steps, seed, device)

            for s_idx, sample in enumerate(test):
                for c_idx, config in enumerate(plan_configs):
                    d = degrade_sample(sample, config=config, seed=seed * 2000 + s_idx * 100 + c_idx)
                    shape = d.gt.shape[:2]
                    restorations = {}
                    for m in methods:
                        if m == "bicubic":
                            restorations[m] = bicubic_restore(d.low_res, shape)
                        elif m == "uniform_sharp":
                            restorations[m] = uniform_sharp_restore(d.low_res, shape)
                        elif m in INR_VARIANTS:
                            restorations[m] = restore_with_semantic_inr(trained[m], d.low_res, shape, device=device)
                        elif m == "tiny_rescnn":
                            restorations[m] = restore_tiny_rescnn(trained[m], d.low_res, shape, device)
                        else:
                            raise ValueError(f"unknown method {m}")
                    mask_eval = _resize_mask(d.mask, (args.crop, args.crop))
                    for m, restored in restorations.items():
                        img_eval = _resize_image(restored, (args.crop, args.crop))
                        base_row = {
                            "dataset": args.dataset, "plan": plan_name, "seed": seed,
                            "sample": sample.name, "degradation": d.degradation_name, "scale": d.scale,
                            "method": m,
                            "psnr": psnr(d.gt, restored), "ssim": ssim_score(d.gt, restored),
                            "boundary_f": boundary_f_score(d.gt, restored),
                        }
                        for arch in archs:
                            row = dict(base_row)
                            row["arch"] = arch
                            row["deep_frozen_miou"] = evaluate_frozen_segmenter(segmenters[arch], img_eval, mask_eval)
                            rows.append(row)

    # write metrics
    fieldnames = ["dataset", "plan", "seed", "arch", "sample", "degradation", "scale", "method",
                  "deep_frozen_miou", "psnr", "ssim", "boundary_f"]
    with (out / "metrics.csv").open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # aggregate: per method per arch, deep mIoU delta vs bicubic (paired across sample/deg/scale/seed)
    def paired_delta(arch, method, metric, plan_filter=None):
        keys = sorted({(r["seed"], r["sample"], r["degradation"], r["scale"]) for r in rows
                       if r["arch"] == arch and (plan_filter is None or r["plan"] == plan_filter)})
        by = {(r["seed"], r["sample"], r["degradation"], r["scale"], r["method"]): r for r in rows if r["arch"] == arch}
        base, prop = [], []
        for k in keys:
            b = by.get((*k, "bicubic"))
            q = by.get((*k, method))
            if b and q:
                base.append(float(b[metric]))
                prop.append(float(q[metric]))
        if len(base) < 2:
            return None
        sig = paired_significance(np.asarray(base), np.asarray(prop), seed=73, bootstraps=2000)
        return {"n": len(base), "mean_delta": sig.mean_delta, "ttest_p": sig.ttest_p,
                "wilcoxon_p": sig.wilcoxon_p, "ci_low": sig.bootstrap_ci_low, "ci_high": sig.bootstrap_ci_high}

    summary = {"device": str(device), "datasets": args.dataset, "plans": args.plan,
               "methods": methods, "archs": archs, "seeds": seeds, "n_rows": len(rows)}
    summary["means"] = {}
    for arch in archs:
        summary["means"][arch] = {}
        for m in methods:
            mrows = [r for r in rows if r["arch"] == arch and r["method"] == m]
            summary["means"][arch][m] = {
                met: float(np.mean([float(r[met]) for r in mrows]))
                for met in ("deep_frozen_miou", "psnr", "ssim", "boundary_f")
            }
    summary["delta_vs_bicubic"] = {}
    for arch in archs:
        summary["delta_vs_bicubic"][arch] = {}
        for m in methods:
            if m == "bicubic":
                continue
            entry = {met: paired_delta(arch, m, met) for met in ("deep_frozen_miou", "psnr", "ssim", "boundary_f")}
            entry["deep_miou_by_plan"] = {pn: paired_delta(arch, m, "deep_frozen_miou", plan_filter=pn)
                                          for pn in {r["plan"] for r in rows}}
            summary["delta_vs_bicubic"][arch][m] = entry

    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"tag": args.tag, "means": summary["means"],
                      "delta_vs_bicubic": summary["delta_vs_bicubic"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
