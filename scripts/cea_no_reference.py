"""M2: no-reference image-quality cross-check (BRISQUE, no ground truth).

Full-reference PSNR/SSIM need a clean target, which is unavailable for real captured UAV frames.
As a GT-free cross-check we compute BRISQUE (lower = better perceptual quality) on bicubic vs
Semantic-INR outputs for held-out (blind) severities and the main structure/composite regimes,
with the clean image as a reference floor. A consistent BRISQUE reduction supports that the
restoration would remain beneficial under blind/real conditions where no clean target exists.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import piq

from cea_plus.degradation import DegradationConfig, degrade_sample
from cea_plus.restoration import bicubic_restore
from cea_plus.semantic_inr import restore_with_semantic_inr
from cea_plus.statistics import paired_significance

from cea_exp import PLANS, load_dataset, train_inr

HELD_OUT = {
    "interp": DegradationConfig(name="x8_interp", scale=8, rain_density=0, fog_strength=0.0, blur_sigma=1.6, noise_sigma=0.03),
    "extrap": DegradationConfig(name="x8_extrap", scale=8, rain_density=0, fog_strength=0.0, blur_sigma=2.5, noise_sigma=0.08),
}


def brisque_of(img: np.ndarray, device) -> float:
    x = torch.from_numpy(np.clip(img, 0, 1).transpose(2, 0, 1)[None]).float().to(device)
    with torch.no_grad():
        return float(piq.brisque(x, data_range=1.0, reduction="none").item())


def main() -> None:
    out = Path("results/cea/no_reference")
    out.mkdir(parents=True, exist_ok=True)
    # BRISQUE in piq runs fine on CPU; avoid MPS op-coverage issues.
    device = torch.device("cpu")
    crop, train_limit, inr_steps = 256, 40, 2000
    seeds = [71, 72]
    train_dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    conditions = dict(HELD_OUT)
    for c in PLANS["structure"]() + PLANS["composite"]():
        conditions[c.name] = c

    rows = []
    for seed in seeds:
        train_pool, test = load_dataset("weedsgalore", crop, train_limit, 16)
        inr = train_inr("semantic_inr", train_pool, PLANS["structure"](), inr_steps, seed, train_dev)
        for cond_name, config in conditions.items():
            for s_idx, sample in enumerate(test):
                d = degrade_sample(sample, config=config, seed=seed * 2000 + s_idx * 100)
                shape = d.gt.shape[:2]
                bic = bicubic_restore(d.low_res, shape)
                inr_img = restore_with_semantic_inr(inr, d.low_res, shape, device=train_dev)
                rows.append({
                    "condition": cond_name, "seed": seed, "sample": sample.name,
                    "brisque_clean": brisque_of(d.gt, device),
                    "brisque_bicubic": brisque_of(bic, device),
                    "brisque_inr": brisque_of(inr_img, device),
                })

    def agg(subset_names):
        sub = [r for r in rows if r["condition"] in subset_names]
        base = np.array([r["brisque_bicubic"] for r in sub])
        prop = np.array([r["brisque_inr"] for r in sub])
        # lower is better => improvement is bicubic - inr
        sig = paired_significance(prop, base, seed=73, bootstraps=1500)  # delta = base - prop = improvement
        return {"n": len(sub),
                "brisque_clean": float(np.mean([r["brisque_clean"] for r in sub])),
                "brisque_bicubic": float(base.mean()), "brisque_inr": float(prop.mean()),
                "brisque_improvement": sig.mean_delta, "ci_low": sig.bootstrap_ci_low,
                "ci_high": sig.bootstrap_ci_high, "ttest_p": sig.ttest_p}

    summary = {
        "metric": "BRISQUE (lower=better); improvement = BRISQUE(bicubic) - BRISQUE(INR)",
        "device": str(device), "seeds": seeds,
        "held_out_blind": {k: agg([k]) for k in HELD_OUT},
        "structure_regime": agg([c.name for c in PLANS["structure"]()]),
        "composite_regime": agg([c.name for c in PLANS["composite"]()]),
        "per_condition": {k: agg([k]) for k in conditions},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"held_out_blind": summary["held_out_blind"],
                      "structure_regime": summary["structure_regime"],
                      "composite_regime": summary["composite_regime"]}, indent=2))


if __name__ == "__main__":
    main()
