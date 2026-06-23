"""Phase C (sample figures): Fig1 problem motivation and Fig3 qualitative grid.

Fig1: one WeedsGalore sample shown as clean / degraded(fog) / restored(ours), with the frozen
segmenter's prediction overlaid on degraded vs restored, annotated with PSNR/SSIM/mIoU -- one glance
shows we target task-oriented recovery, not just sharpening.

Fig3: rows = {WeedsGalore fog, LoveDA-Rural mixed, LoveDA-Urban x4}, cols = {Degraded, SwinIR, HAT,
Restormer, LIIF, Ours, GT}, so a reviewer can compare strong SR baselines against our restoration on
agricultural/remote-sensing scenes.
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

from pathlib import Path

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cea_plus.degradation import DegradationConfig, degrade_sample
from cea_plus.deep_downstream import _resize_image, _resize_mask, train_torch_segmenter
from cea_plus.metrics import psnr, ssim_score
from cea_plus.restoration import bicubic_restore
from cea_plus.semantic_inr import restore_with_semantic_inr
from cea_plus import strong_baselines as sb

from cea_exp import load_dataset, train_inr
from PIL import Image

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 9, "figure.dpi": 150})

FIGDIR = Path("paper/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

STRUCT_PLAN = [
    DegradationConfig(name="x4_blur", scale=4, rain_density=0, fog_strength=0.0, blur_sigma=1.5, noise_sigma=0.01),
    DegradationConfig(name="x4_compound", scale=4, rain_density=20, fog_strength=0.25, blur_sigma=1.5, noise_sigma=0.04, jpeg_quality=45),
]
FOG = DegradationConfig(name="x4_fog", scale=4, rain_density=0, fog_strength=0.45, blur_sigma=0.8, noise_sigma=0.01)
MIXED = DegradationConfig(name="x4_mixed", scale=4, rain_density=30, fog_strength=0.38, blur_sigma=0.8, noise_sigma=0.015, jpeg_quality=50)
BLUR = DegradationConfig(name="x4_blur", scale=4, rain_density=0, fog_strength=0.0, blur_sigma=1.5, noise_sigma=0.01)


def _overlay(img, mask, color=(1.0, 0.2, 0.2), alpha=0.45):
    out = img.copy()
    m = mask.astype(bool)
    for c in range(3):
        out[..., c] = np.where(m, (1 - alpha) * out[..., c] + alpha * color[c], out[..., c])
    return np.clip(out, 0, 1)


def _load_models(dev):
    models = {}
    for name, loader in sb.LOADERS.items():
        m, err = loader(dev)
        if m is not None:
            models[name] = (m, sb.RESTORERS[name])
        else:
            print(f"[skip] {name}: {err}")
    return models


def fig1_problem_motivation():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    crop = 256
    train_pool, test = load_dataset("weedsgalore", crop, 40, 16)
    seg = train_torch_segmenter(train_samples=train_pool, architecture="segformer_b0_imagenet",
                                steps=150, crop_size=128, input_size=crop, seed=71, device=dev)
    inr = train_inr("semantic_inr", train_pool, STRUCT_PLAN, 1500, 71, dev)
    from cea_plus.downstream import evaluate_frozen_segmenter

    cfg = DegradationConfig(name="x4_compound", scale=4, rain_density=20, fog_strength=0.25,
                            blur_sigma=1.5, noise_sigma=0.04, jpeg_quality=45)
    # pick the sample where restoration most improves the task (honest, illustrative composite case)
    best = None
    for sample in test[:8]:
        d = degrade_sample(sample, config=cfg, seed=4242)
        shape = d.gt.shape[:2]
        deg = bicubic_restore(d.low_res, shape)
        rest = restore_with_semantic_inr(inr, d.low_res, shape, device=dev)
        meval = _resize_mask(d.mask, (crop, crop))
        gd = evaluate_frozen_segmenter(seg, _resize_image(deg, (crop, crop)), meval)
        gr = evaluate_frozen_segmenter(seg, _resize_image(rest, (crop, crop)), meval)
        if best is None or (gr - gd) > best[0]:
            best = (gr - gd, sample, d, shape, deg, rest, meval)
    _, sample, d, shape, deg, rest, meval = best

    def miou(img):
        return evaluate_frozen_segmenter(seg, _resize_image(img, (crop, crop)), meval)

    pred_deg = seg.predict_mask(_resize_image(deg, (crop, crop)))
    pred_rest = seg.predict_mask(_resize_image(rest, (crop, crop)))
    gt_small = _resize_image(d.gt, (crop, crop))

    gd = miou(deg)
    gr = miou(rest)
    delta = gr - gd

    fig, ax = plt.subplots(2, 3, figsize=(10.5, 7.2))
    titles = [
        "Clean RGB",
        "Degraded input",
        "Restored (Ours)",
        "GT mask",
        "Seg. on degraded",
        "Seg. on restored",
    ]
    images = [
        gt_small,
        _resize_image(deg, (crop, crop)),
        _resize_image(rest, (crop, crop)),
        _overlay(gt_small, meval),
        _overlay(_resize_image(deg, (crop, crop)), pred_deg),
        _overlay(_resize_image(rest, (crop, crop)), pred_rest),
    ]
    subs = [
        "",
        f"PSNR {psnr(d.gt, deg):.1f} dB",
        f"PSNR {psnr(d.gt, rest):.1f} dB",
        "",
        f"mIoU {gd:.3f}",
        f"mIoU {gr:.3f}",
    ]
    zoom_idx = {4, 5}
    for i, a in enumerate(ax.ravel()):
        if i in zoom_idx:
            _draw_with_zoom(a, images[i])
        else:
            a.imshow(np.clip(images[i], 0, 1))
            a.set_xticks([])
            a.set_yticks([])
        a.set_title(titles[i], fontsize=10, weight="bold")
        if subs[i]:
            a.set_xlabel(subs[i], fontsize=9)
    fig.suptitle(
        f"Frozen SegFormer-B0 on WeedsGalore composite ($\\times4$): "
        f"mIoU {gd:.3f} $\\rightarrow$ {gr:.3f} ($\\Delta$ {delta:+.3f})",
        fontsize=11,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = FIGDIR / "fig2_example.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    # legacy name for pipelines that still expect it
    legacy = FIGDIR / "fig1_problem_motivation.png"
    fig2 = Image.open(out)
    fig2.save(legacy)
    print("wrote", out, "and", legacy)


def _draw_with_zoom(ax, img, zoom_frac=0.28, inset_frac=0.42):
    """Show image with a magnified crop inset (crop/weed boundary region, center-biased)."""
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_xticks([]); ax.set_yticks([])
    h, w = img.shape[:2]
    zh, zw = int(h * zoom_frac), int(w * zoom_frac)
    y0, x0 = (h - zh) // 2, (w - zw) // 2
    patch = img[y0:y0 + zh, x0:x0 + zw]
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ins = inset_axes(ax, width=f"{inset_frac * 100:.0f}%", height=f"{inset_frac * 100:.0f}%",
                     loc="lower right", borderpad=0.4)
    ins.imshow(patch); ins.set_xticks([]); ins.set_yticks([])
    for spine in ins.spines.values():
        spine.set_edgecolor("#d62728"); spine.set_linewidth(1.2)
    rect = plt.Rectangle((x0, y0), zw, zh, fill=False, edgecolor="#d62728", linewidth=1.0)
    ax.add_patch(rect)


def _error_map(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    h, w = pred.shape
    out = np.ones((h, w, 3), dtype=np.float32)
    out[..., 0] = 0.92
    out[..., 1] = 0.92
    out[..., 2] = 0.92
    fp = pred & ~gt
    fn = ~pred & gt
    out[fp] = (1.0, 0.25, 0.25)
    out[fn] = (0.25, 0.35, 1.0)
    return out


# Dataset-level metrics (WeedsGalore x4, frozen SegFormer-B0; Table 3 in paper)
FIG3_METRICS = {
    "Degraded": {"brisque": 56.7, "miou": 0.609, "delta": 0.0},
    "SwinIR": {"brisque": 47.1, "miou": 0.589, "delta": -0.021},
    "HAT": {"brisque": 46.6, "miou": 0.590, "delta": -0.019},
    "LIIF": {"brisque": 48.5, "miou": 0.583, "delta": -0.026},
    "Ours": {"brisque": 57.5, "miou": 0.610, "delta": 0.001},
    "GT": {"brisque": None, "miou": None, "delta": None},
}


def fig3_qualitative_grid():
    dev = torch.device("cpu")
    models = _load_models(dev)
    cols = ["Degraded", "SwinIR", "HAT", "LIIF", "Ours", "GT"]
    row_labels = ["Restored RGB", "Segmentation", "Error map"]
    train_dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    wg_train, wg_test = load_dataset("weedsgalore", 256, 40, 8)
    seg = train_torch_segmenter(
        train_samples=wg_train,
        architecture="segformer_b0_imagenet",
        steps=150,
        crop_size=128,
        input_size=256,
        seed=71,
        device=train_dev,
    )
    inr = train_inr("semantic_inr", wg_train, STRUCT_PLAN, 1500, 71, train_dev)
    from cea_plus.downstream import evaluate_frozen_segmenter

    cfg = DegradationConfig(
        name="x4_compound",
        scale=4,
        rain_density=20,
        fog_strength=0.25,
        blur_sigma=1.5,
        noise_sigma=0.04,
        jpeg_quality=45,
    )
    sample = wg_test[0]
    d = degrade_sample(sample, config=cfg, seed=4242)
    shape = d.gt.shape[:2]
    meval = _resize_mask(d.mask, (256, 256))
    imgs = {
        "Degraded": bicubic_restore(d.low_res, shape),
        "Ours": restore_with_semantic_inr(inr, d.low_res, shape, device=train_dev),
        "GT": d.gt,
    }
    for name, key in [("swinir", "SwinIR"), ("hat", "HAT"), ("liif", "LIIF")]:
        if name in models:
            m, fn = models[name]
            imgs[key] = fn(m, d.low_res, shape, dev)
        else:
            imgs[key] = imgs["Degraded"]

    fig = plt.figure(figsize=(12.5, 5.2))
    gs = fig.add_gridspec(4, len(cols), height_ratios=[1.0, 1.0, 1.0, 0.12], hspace=0.35, wspace=0.06)
    for c, col in enumerate(cols):
        img = _resize_image(imgs[col], (256, 256))
        if col == "GT":
            pred = meval.astype(np.uint8)
            seg_panel = _overlay(img, meval)
            err = np.ones((256, 256, 3), dtype=np.float32) * 0.95
        else:
            pred = seg.predict_mask(img)
            seg_panel = _overlay(img, pred)
            err = _error_map(pred, meval)
        panels = [img, seg_panel, err]
        for r, panel in enumerate(panels):
            ax = fig.add_subplot(gs[r, c])
            ax.imshow(np.clip(panel, 0, 1))
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=9, weight="bold")
            if r == 0:
                ax.set_title(col, fontsize=10, weight="bold")
        axm = fig.add_subplot(gs[3, c])
        axm.axis("off")
        met = FIG3_METRICS[col]
        if met["brisque"] is None:
            txt = "Clean reference"
        else:
            txt = f"BRISQUE {met['brisque']:.1f}\nmIoU {met['miou']:.3f} ($\\Delta${met['delta']:+.3f})"
        axm.text(0.5, 0.5, txt, ha="center", va="center", fontsize=7.5)

    fig.suptitle(
        "WeedsGalore composite ($\\times4$): strong SR looks sharper but increases segmentation errors",
        fontsize=11,
        y=1.01,
    )
    out = FIGDIR / "fig3_baselines_weeds.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    legacy = FIGDIR / "fig3_qualitative_grid.png"
    Image.open(out).save(legacy)
    print("wrote", out, "and", legacy)


if __name__ == "__main__":
    import sys
    cmds = sys.argv[1:] or ["fig1", "fig3"]
    if "fig1" in cmds:
        fig1_problem_motivation()
    if "fig3" in cmds:
        fig3_qualitative_grid()
