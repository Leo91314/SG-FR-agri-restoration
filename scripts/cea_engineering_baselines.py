"""Reviewer point 6: engineering and fine-tuned restoration baselines (extends Table 3).

Under the SAME no-leak x4 protocol as cea_strong_baselines.py (same seeds, WeedsGalore crops,
frozen clean-trained SegFormer-B0, paired bootstrap CIs vs bicubic), we add:
  * Engineering (no-training) baselines: CLAHE, denoise+sharpen, dehaze, unsharp mask.
  * A learned restorer fine-tuned on the agricultural training data: a compact U-Net refiner.
  * Optional: SwinIR fine-tuned on the agricultural data (guarded; recorded as not_evaluated on failure).

Goal: show that neither classic image-processing pipelines nor a restorer/SR fine-tuned on the same
agricultural data turns generic restoration into a reliable task-improver -- the conditional,
task-oriented SG-FR behaviour is what matters, not raw restorer strength.

Output: results/cea/engineering_baselines/{summary.json, rows.json}.
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

import json
import sys
import time
from pathlib import Path

import numpy as np
import piq
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from cea_plus.degradation import degrade_sample
from cea_plus.deep_downstream import _resize_image, _resize_mask, train_torch_segmenter
from cea_plus.downstream import evaluate_frozen_segmenter
from cea_plus.metrics import psnr, ssim_score
from cea_plus.restoration import (bicubic_restore, uniform_sharp_restore, clahe_restore,
                                  denoise_sharpen_restore, dehaze_restore)
from cea_plus.statistics import paired_significance

from cea_strong_baselines import X4_PLAN, brisque_of
from cea_exp import load_dataset

ENG_METHODS = {
    "clahe": clahe_restore,
    "unsharp": uniform_sharp_restore,
    "denoise_sharpen": denoise_sharpen_restore,
    "dehaze": dehaze_restore,
}
GROUPS = {
    "bicubic": "classical",
    "clahe": "engineering",
    "unsharp": "engineering",
    "denoise_sharpen": "engineering",
    "dehaze": "engineering",
    "unet_finetuned": "fine_tuned",
    "swinir_finetuned": "fine_tuned",
}


# --------------------------------------------------------------------------- compact U-Net refiner
class TinyUNet(nn.Module):
    def __init__(self, ch=24):
        super().__init__()
        def cbr(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, padding=1), nn.ReLU(inplace=True))
        self.e1 = cbr(3, ch)
        self.e2 = cbr(ch, ch * 2)
        self.pool = nn.MaxPool2d(2)
        self.b = cbr(ch * 2, ch * 4)
        self.up2 = nn.ConvTranspose2d(ch * 4, ch * 2, 2, stride=2)
        self.d2 = cbr(ch * 4, ch * 2)
        self.up1 = nn.ConvTranspose2d(ch * 2, ch, 2, stride=2)
        self.d1 = cbr(ch * 2, ch)
        self.out = nn.Conv2d(ch, 3, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        b = self.b(self.pool(e2))
        d2 = self.d2(torch.cat([self.up2(b), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(x + self.out(d1))  # residual refine of the bicubic upsample


def _t(img, device):
    return torch.from_numpy(np.clip(img, 0, 1).transpose(2, 0, 1)[None]).float().to(device)


def train_unet(train_pool, plan, steps, seed, device):
    torch.manual_seed(seed)
    net = TinyUNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    rng = np.random.default_rng(seed)
    net.train()
    for step in range(steps):
        sample = train_pool[int(rng.integers(0, len(train_pool)))]
        config = plan[int(rng.integers(0, len(plan)))]
        d = degrade_sample(sample, config=config, seed=int(rng.integers(0, 10**6)))
        up = bicubic_restore(d.low_res, d.gt.shape[:2])
        x = _t(up, device)
        y = _t(d.gt, device)
        out = net(x)
        loss = (out - y).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def restore_unet(net, low_res, shape, device):
    up = bicubic_restore(low_res, shape)
    out = net(_t(up, device)).clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
    return out.astype(np.float32)


def finetune_swinir(train_pool, plan, steps, seed, device):
    """Fine-tune SwinIR (x4) on the agricultural data; returns (model or None, err)."""
    from cea_plus import strong_baselines as sb
    model, err = sb.load_swinir(device)
    if model is None:
        return None, err
    try:
        for p in model.parameters():
            p.requires_grad_(True)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        rng = np.random.default_rng(seed)
        for step in range(steps):
            sample = train_pool[int(rng.integers(0, len(train_pool)))]
            config = plan[int(rng.integers(0, len(plan)))]
            d = degrade_sample(sample, config=config, seed=int(rng.integers(0, 10**6)))
            x = _t(d.low_res, device)
            y = _t(d.gt, device)
            out = model(x)
            if out.shape[-2:] != y.shape[-2:]:
                out = torch.nn.functional.interpolate(out, size=y.shape[-2:], mode="bicubic", align_corners=False)
            loss = (out.clamp(0, 1) - y).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        return model, None
    except Exception as e:  # pragma: no cover
        return None, f"finetune error: {type(e).__name__}: {e}"


@torch.no_grad()
def restore_swinir_ft(model, low_res, shape, device):
    x = _t(low_res, device)
    out = model(x).clamp(0, 1)
    if out.shape[-2:] != tuple(shape):
        out = torch.nn.functional.interpolate(out, size=tuple(shape), mode="bicubic", align_corners=False)
    return out[0].cpu().numpy().transpose(1, 2, 0).astype(np.float32)


def main():
    do_swinir = "--with-swinir" in sys.argv
    out = Path("results/cea/engineering_baselines")
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    crop, train_limit, seg_steps = 256, 40, 150
    unet_steps = 600
    swinir_steps = 250
    seeds = [71, 72]
    not_evaluated = {}

    rows = []
    t0 = time.time()
    for seed in seeds:
        train_pool, test = load_dataset("weedsgalore", crop, train_limit, 16)
        seg = train_torch_segmenter(train_samples=train_pool, architecture="segformer_b0_imagenet",
                                    steps=seg_steps, crop_size=128, input_size=crop, seed=seed, device=dev)
        unet = train_unet(train_pool, X4_PLAN, unet_steps, seed, dev)
        print(f"[seed {seed}] unet trained, elapsed {time.time()-t0:.0f}s", flush=True)
        swinir_ft = None
        if do_swinir:
            swinir_ft, err = finetune_swinir(train_pool, X4_PLAN, swinir_steps, seed, dev)
            if swinir_ft is None:
                not_evaluated["swinir_finetuned"] = err
                print(f"[seed {seed}] swinir ft skipped: {err}", flush=True)
            else:
                print(f"[seed {seed}] swinir fine-tuned, elapsed {time.time()-t0:.0f}s", flush=True)

        for s_idx, sample in enumerate(test):
            for config in X4_PLAN:
                d = degrade_sample(sample, config=config, seed=seed * 2000 + s_idx * 100)
                shape = d.gt.shape[:2]
                mask_eval = _resize_mask(d.mask, (crop, crop))
                restorations = {"bicubic": bicubic_restore(d.low_res, shape)}
                for name, fn in ENG_METHODS.items():
                    restorations[name] = fn(d.low_res, shape)
                restorations["unet_finetuned"] = restore_unet(unet, d.low_res, shape, dev)
                if swinir_ft is not None:
                    restorations["swinir_finetuned"] = restore_swinir_ft(swinir_ft, d.low_res, shape, dev)
                for m, img in restorations.items():
                    rows.append({
                        "seed": seed, "sample": sample.name, "condition": config.name, "method": m,
                        "group": GROUPS.get(m, "other"),
                        "miou": evaluate_frozen_segmenter(seg, _resize_image(img, (crop, crop)), mask_eval),
                        "psnr": psnr(d.gt, img), "ssim": ssim_score(d.gt, img),
                        "brisque": brisque_of(img, dev),
                    })
        print(f"[seed {seed}] eval done, elapsed {time.time()-t0:.0f}s", flush=True)

    methods = [m for m in GROUPS if any(r["method"] == m for r in rows)]
    keys = sorted({(r["seed"], r["sample"], r["condition"]) for r in rows})
    by = {(r["seed"], r["sample"], r["condition"], r["method"]): r for r in rows}

    summary = {"seeds": seeds, "scale": "x4", "conditions": [c.name for c in X4_PLAN],
               "groups": GROUPS, "not_evaluated": not_evaluated, "means": {}, "delta_vs_bicubic": {}}
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
            sig = paired_significance(prop, base, seed=73, bootstraps=1500) if met == "brisque" \
                else paired_significance(base, prop, seed=73, bootstraps=1500)
            entry[met] = {"mean_delta": sig.mean_delta, "ci_low": sig.bootstrap_ci_low,
                          "ci_high": sig.bootstrap_ci_high}
        summary["delta_vs_bicubic"][m] = entry

    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out / "rows.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"means": summary["means"],
                      "miou_delta": {m: summary["delta_vs_bicubic"][m]["miou"] for m in methods if m != "bicubic"},
                      "not_evaluated": not_evaluated}, indent=2))


if __name__ == "__main__":
    main()
