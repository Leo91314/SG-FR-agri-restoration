"""Phase A: full strong-baseline suite under the no-leak protocol (CEA Table 3).

Evaluates SwinIR, HAT-L, Restormer, LIIF (+ Swin2SR classical/real-world) zero-shot against bicubic,
a generic learned restorer (Tiny ResCNN), and our Semantic-INR, on identical WeedsGalore crops, seeds,
and x4 degradations. No model sees GT, masks, or degradation labels at inference -- each consumes only the
degraded LR and the target shape. We report frozen downstream mIoU (SegFormer-B0), PSNR/SSIM, and BRISQUE,
with paired bootstrap CIs vs bicubic.

Honesty guardrail: any model whose weights are unavailable or that cannot be built is recorded under
``not_evaluated`` with a reason, never fabricated. LTE weights are not available on accessible mirrors and
are reported as not evaluated.

Grouping for the paper table: Classical (bicubic, Tiny ResCNN) / Transformer-restoration
(Swin2SR, SwinIR, HAT, Restormer) / Implicit-representation (LIIF) / Ours (Semantic-INR).
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import json
import time
from pathlib import Path

import numpy as np
import piq
import torch

from cea_plus.degradation import DegradationConfig, degrade_sample
from cea_plus.deep_downstream import _resize_image, _resize_mask, train_torch_segmenter
from cea_plus.downstream import evaluate_frozen_segmenter
from cea_plus.metrics import psnr, ssim_score
from cea_plus.restoration import bicubic_restore
from cea_plus.semantic_inr import restore_with_semantic_inr
from cea_plus.statistics import paired_significance
from cea_plus import strong_baselines as sb

from cea_exp import load_dataset, train_inr, train_tiny_rescnn, restore_tiny_rescnn

X4_PLAN = [
    DegradationConfig(name="x4_blur", scale=4, rain_density=0, fog_strength=0.0, blur_sigma=1.5, noise_sigma=0.01),
    DegradationConfig(name="x4_noise", scale=4, rain_density=0, fog_strength=0.0, blur_sigma=0.8, noise_sigma=0.05),
    DegradationConfig(name="x4_compound", scale=4, rain_density=20, fog_strength=0.25, blur_sigma=1.5, noise_sigma=0.04, jpeg_quality=45),
]

SWIN2SR = {
    "swin2sr_classical": "caidas/swin2SR-classical-sr-x4-64",
    "swin2sr_realworld": "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
}

GROUPS = {
    "bicubic": "classical",
    "tiny_rescnn": "classical",
    "swin2sr_classical": "transformer_restoration",
    "swin2sr_realworld": "transformer_restoration",
    "swinir": "transformer_restoration",
    "hat": "transformer_restoration",
    "restormer": "transformer_restoration",
    "liif": "implicit_representation",
    "semantic_inr": "ours",
}

NOT_EVALUATED_STATIC = {
    "lte": "not evaluated (official LTE weights only on Google Drive/Baidu; no accessible mirror)",
}


def brisque_of(img, device):
    x = torch.from_numpy(np.clip(img, 0, 1).transpose(2, 0, 1)[None]).float().to(device)
    with torch.no_grad():
        return float(piq.brisque(x, data_range=1.0, reduction="none").item())


@torch.no_grad()
def swin2sr_restore(model, low_res, shape, device):
    x = torch.from_numpy(np.clip(low_res, 0, 1).transpose(2, 0, 1)[None]).float().to(device)
    out = model(x).reconstruction.clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
    if out.shape[:2] != tuple(shape):
        out = _resize_image(out, tuple(shape))
    return out.astype(np.float32)


def main() -> None:
    out = Path("results/cea/strong_baselines")
    out.mkdir(parents=True, exist_ok=True)
    train_dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    sr_dev = torch.device("cpu")  # external SR + BRISQUE on CPU for op-coverage safety
    crop, train_limit, inr_steps, seg_steps = 256, 40, 2000, 150
    seeds = [71, 72]

    # --- load external models, honest fallback on failure ---
    not_evaluated = dict(NOT_EVALUATED_STATIC)
    external = {}  # name -> (model, restore_fn)

    from transformers import Swin2SRForImageSuperResolution

    for name, mid in SWIN2SR.items():
        try:
            m = Swin2SRForImageSuperResolution.from_pretrained(mid).to(sr_dev).eval()
            external[name] = (m, lambda mdl, lr, sh, dv: swin2sr_restore(mdl, lr, sh, dv))
        except Exception as e:  # pragma: no cover
            not_evaluated[name] = f"load error: {type(e).__name__}: {e}"

    for name, loader in sb.LOADERS.items():
        model, err = loader(sr_dev)
        if model is None:
            not_evaluated[name] = err
            print(f"[skip] {name}: {err}")
        else:
            external[name] = (model, sb.RESTORERS[name])
            print(f"[ok]   {name} loaded")

    rows = []
    t0 = time.time()
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
                for name, (model, restore_fn) in external.items():
                    restorations[name] = restore_fn(model, d.low_res, shape, sr_dev)
                for m, img in restorations.items():
                    rows.append({
                        "seed": seed, "sample": sample.name, "condition": config.name, "method": m,
                        "group": GROUPS.get(m, "other"),
                        "miou": evaluate_frozen_segmenter(seg, _resize_image(img, (crop, crop)), mask_eval),
                        "psnr": psnr(d.gt, img), "ssim": ssim_score(d.gt, img),
                        "brisque": brisque_of(img, sr_dev),
                    })
        print(f"[seed {seed}] done, elapsed {time.time() - t0:.0f}s")

    methods = [m for m in GROUPS if any(r["method"] == m for r in rows)]
    keys = sorted({(r["seed"], r["sample"], r["condition"]) for r in rows})
    by = {(r["seed"], r["sample"], r["condition"], r["method"]): r for r in rows}

    summary = {
        "seeds": seeds, "scale": "x4", "conditions": [c.name for c in X4_PLAN],
        "groups": GROUPS, "not_evaluated": not_evaluated, "means": {}, "delta_vs_bicubic": {},
    }
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
            if met == "brisque":  # lower better
                sig = paired_significance(prop, base, seed=73, bootstraps=1500)
            else:
                sig = paired_significance(base, prop, seed=73, bootstraps=1500)
            entry[met] = {"mean_delta": sig.mean_delta, "ci_low": sig.bootstrap_ci_low,
                          "ci_high": sig.bootstrap_ci_high, "ttest_p": sig.ttest_p}
        summary["delta_vs_bicubic"][m] = entry

    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out / "rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"means": summary["means"], "not_evaluated": not_evaluated,
                      "miou_delta": {m: summary["delta_vs_bicubic"][m]["miou"] for m in methods if m != "bicubic"}},
                     indent=2))


if __name__ == "__main__":
    main()
