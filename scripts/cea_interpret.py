"""Phase 8: interpretability + frequency-decoupling evidence.

Trains a no-leak Semantic INR on the structure regime, then for a few test samples
dumps structure S, texture T, alpha, and semantic probability fields, plus the radial
power spectra of S vs T to show the low/high-frequency decoupling.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cea_plus.dataset import load_weedsgalore_dataset
from cea_plus.degradation import degrade_sample
from cea_plus.deep_downstream import _resize_image, _resize_mask
from cea_plus.semantic_inr import TinySemanticINR, predict_semantic_inr_fields, train_semantic_inr_steps
from cea_plus.synthesis import AgriSample

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cea_exp import PLANS, center_crop, build_batches  # reuse driver helpers


def radial_power_spectrum(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    power = np.abs(f) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), power.ravel())
    nr = np.bincount(r.ravel())
    radial = tbin / np.maximum(nr, 1)
    return radial[: min(cy, cx)]


def main() -> None:
    out = Path("results/cea/phase8_interpretability")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    crop = 256
    seed = 71

    root = Path("data/external/weedsgalore-dataset")
    train = [center_crop(s, crop) for s in load_weedsgalore_dataset(root, split="train", limit=48)]
    test = [center_crop(s, crop) for s in load_weedsgalore_dataset(root, split="test", limit=4)]
    plan = PLANS["structure"]()

    torch.manual_seed(seed)
    model = TinySemanticINR(hidden_channels=48, base_sharpen_strength=0.0,
                            structure_residual_scale=0.6, texture_residual_scale=0.2, semantic_detail_boost=0.30)
    train_semantic_inr_steps(model, batches=build_batches(train, plan, seed), steps=2000,
                             learning_rate=1e-3, task_loss_weight=0.0, semantic_loss_weight=0.03, seed=seed, device=device)

    struct_spectra, texture_spectra = [], []
    for s_idx, sample in enumerate(test):
        config = plan[0]  # x8_blur
        d = degrade_sample(sample, config=config, seed=seed * 2000 + s_idx * 100)
        shape = d.gt.shape[:2]
        fields = predict_semantic_inr_fields(model, d.low_res, shape, device=device)

        fig, ax = plt.subplots(2, 4, figsize=(16, 8))
        ax[0, 0].imshow(np.clip(_resize_image(d.low_res, shape), 0, 1)); ax[0, 0].set_title("low_res (bicubic shown)")
        ax[0, 1].imshow(np.clip(d.gt, 0, 1)); ax[0, 1].set_title("clean GT")
        ax[0, 2].imshow(np.clip(fields["restored"], 0, 1)); ax[0, 2].set_title("INR restored")
        ax[0, 3].imshow(d.mask, cmap="gray"); ax[0, 3].set_title("GT mask (ref only)")
        ax[1, 0].imshow(np.clip(fields["structure"], 0, 1)); ax[1, 0].set_title("structure S")
        tex = fields["texture"]; tex_vis = np.clip(0.5 + 4.0 * tex, 0, 1)
        ax[1, 1].imshow(tex_vis); ax[1, 1].set_title("texture T (x4)")
        ax[1, 2].imshow(fields["alpha"], cmap="viridis"); ax[1, 2].set_title("alpha")
        ax[1, 3].imshow(fields["semantic_prob"], cmap="magma"); ax[1, 3].set_title("semantic prob")
        for a in ax.ravel():
            a.axis("off")
        fig.tight_layout()
        fig.savefig(out / f"fields_{sample.name}.png", dpi=110)
        plt.close(fig)

        struct_spectra.append(radial_power_spectrum(fields["structure"].mean(axis=2)))
        texture_spectra.append(radial_power_spectrum(tex.mean(axis=2)))

    n = min(min(len(s) for s in struct_spectra), min(len(s) for s in texture_spectra))
    s_mean = np.mean([s[:n] for s in struct_spectra], axis=0)
    t_mean = np.mean([s[:n] for s in texture_spectra], axis=0)
    s_norm = s_mean / (s_mean.sum() + 1e-12)
    t_norm = t_mean / (t_mean.sum() + 1e-12)

    fig, ax = plt.subplots(figsize=(7, 5))
    freq = np.arange(n)
    ax.plot(freq, s_norm, label="structure S")
    ax.plot(freq, t_norm, label="texture T")
    ax.set_yscale("log")
    ax.set_xlabel("radial spatial frequency (low -> high)")
    ax.set_ylabel("normalized power")
    ax.set_title("Frequency decoupling: structure vs texture")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "frequency_decoupling_spectrum.png", dpi=120)
    plt.close(fig)

    low_band = slice(0, max(1, n // 4))
    high_band = slice(n // 2, n)
    summary = {
        "structure_low_freq_fraction": float(s_norm[low_band].sum()),
        "structure_high_freq_fraction": float(s_norm[high_band].sum()),
        "texture_low_freq_fraction": float(t_norm[low_band].sum()),
        "texture_high_freq_fraction": float(t_norm[high_band].sum()),
        "interpretation": "structure concentrates power at low frequencies; texture carries relatively more high-frequency power",
        "panels": sorted(str(p.name) for p in out.glob("fields_*.png")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
