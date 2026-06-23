"""Loaders and restorers for strong pretrained baselines under the no-leak protocol.

All models run zero-shot (no fine-tuning, no GT/mask/degradation labels at inference) and consume only
the degraded LR image plus the target output shape -- identical access to our Semantic-INR. Each loader
returns ``None`` gracefully (with a logged reason) if weights are missing or the model cannot be built,
so the evaluation can report an honest "not evaluated (reason)" instead of fabricating numbers.

Models / sources:
- SwinIR  (Liang et al., ICCVW 2021)      : basicsr arch + official classical-SR x4 weights.
- HAT-L   (Chen et al., CVPR 2023)        : vendored arch (XPixelGroup/HAT) + HF x4 weights.
- Restormer (Zamir et al., CVPR 2022)     : vendored arch (swz30/Restormer) + blind color-denoise weights;
                                            same-resolution, applied as a refiner on the bicubic upsample.
- LIIF    (Chen et al., CVPR 2021)        : vendored RDN-LIIF (yinboc/liif) + HF weights; arbitrary scale.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from cea_plus.deep_downstream import _resize_image
from cea_plus.restoration import bicubic_restore

# basicsr (SwinIR/HAT helpers) imports a torchvision module that moved in newer releases; shim it.
import torchvision.transforms.functional as _tvf  # noqa: E402

sys.modules.setdefault("torchvision.transforms.functional_tensor", _tvf)

WEIGHTS_DIR = Path("data/weights")


def _to_tensor(img: np.ndarray, device) -> torch.Tensor:
    return torch.from_numpy(np.clip(img, 0, 1).transpose(2, 0, 1)[None]).float().to(device)


def _to_image(t: torch.Tensor, shape) -> np.ndarray:
    out = t.clamp(0, 1)[0].cpu().numpy().transpose(1, 2, 0)
    if out.shape[:2] != tuple(shape):
        out = _resize_image(out, tuple(shape))
    return out.astype(np.float32)


# --------------------------------------------------------------------------- SwinIR
def load_swinir(device, weights: Path = WEIGHTS_DIR / "swinir_classical_x4.pth"):
    if not weights.exists():
        return None, f"weights missing ({weights})"
    try:
        from basicsr.archs.swinir_arch import SwinIR

        m = SwinIR(upscale=4, in_chans=3, img_size=48, window_size=8, img_range=1.0,
                   depths=[6] * 6, embed_dim=180, num_heads=[6] * 6, mlp_ratio=2,
                   upsampler="pixelshuffle", resi_connection="1conv").eval().to(device)
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        sd = sd.get("params", sd)
        sd = {k: v for k, v in sd.items() if "attn_mask" not in k}
        m.load_state_dict(sd, strict=False)
        return m, None
    except Exception as e:  # pragma: no cover - integration guard
        return None, f"load error: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- HAT-L
def load_hat(device, weights: Path = WEIGHTS_DIR / "hat_l_x4.pth"):
    if not weights.exists():
        return None, f"weights missing ({weights})"
    try:
        from cea_plus.extern.hat_arch import HAT

        sd = torch.load(weights, map_location="cpu", weights_only=False)
        sd = sd.get("params_ema", sd.get("params", sd))
        import re

        nlayers = len({re.match(r"layers\.(\d+)\.", k).group(1) for k in sd if k.startswith("layers.")})
        embed = sd["conv_first.weight"].shape[0]
        m = HAT(upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=3, squeeze_factor=30,
                conv_scale=0.01, overlap_ratio=0.5, img_range=1.0, depths=[6] * nlayers, embed_dim=embed,
                num_heads=[6] * nlayers, mlp_ratio=2, upsampler="pixelshuffle", resi_connection="1conv").eval().to(device)
        sd = {k: v for k, v in sd.items() if "attn_mask" not in k and "relative_position_index" not in k}
        m.load_state_dict(sd, strict=False)
        return m, None
    except Exception as e:  # pragma: no cover
        return None, f"load error: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- Restormer
def load_restormer(device, weights: Path = WEIGHTS_DIR / "restormer_denoise.pth"):
    if not weights.exists():
        return None, f"weights missing ({weights})"
    try:
        from cea_plus.extern.restormer_arch import Restormer

        m = Restormer(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8], num_refinement_blocks=4,
                      heads=[1, 2, 4, 8], ffn_expansion_factor=2.66, bias=False, LayerNorm_type="BiasFree",
                      dual_pixel_task=False).eval().to(device)
        sd = torch.load(weights, map_location="cpu", weights_only=False)
        sd = sd.get("params", sd)
        m.load_state_dict(sd, strict=True)
        return m, None
    except Exception as e:  # pragma: no cover
        return None, f"load error: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- LIIF (RDN)
def load_liif(device, weights: Path = WEIGHTS_DIR / "rdn_liif.pth"):
    if not weights.exists():
        return None, f"weights missing ({weights})"
    try:
        from cea_plus.extern import liiflib

        sv = torch.load(weights, map_location="cpu", weights_only=False)
        model = liiflib.make(sv["model"], load_sd=True).eval().to(device)
        return model, None
    except Exception as e:  # pragma: no cover
        return None, f"load error: {type(e).__name__}: {e}"


# --------------------------------------------------------------------------- restorers
@torch.no_grad()
def restore_pixelshuffle_sr(model, low_res: np.ndarray, shape, device) -> np.ndarray:
    """Fixed-scale (x4) SR models: feed LR, model upscales internally."""
    x = _to_tensor(low_res, device)
    y = model(x)
    return _to_image(y, shape)


@torch.no_grad()
def restore_restormer(model, low_res: np.ndarray, shape, device) -> np.ndarray:
    """Same-resolution restorer: refine the bicubic upsample to the target shape."""
    up = bicubic_restore(low_res, shape)
    x = _to_tensor(up, device)
    y = model(x)
    return _to_image(y, shape)


@torch.no_grad()
def restore_liif(model, low_res: np.ndarray, shape, device, chunk: int = 30000) -> np.ndarray:
    """Arbitrary-scale INR: query target-resolution coords from the LR feature field."""
    from cea_plus.extern.liiflib import make_coord

    inp = _to_tensor(low_res, device)
    inp = (inp - 0.5) / 0.5  # official LIIF input normalization
    h, w = int(shape[0]), int(shape[1])
    coord = make_coord((h, w), device=device).unsqueeze(0)
    cell = torch.ones_like(coord)
    cell[:, :, 0] *= 2 / h
    cell[:, :, 1] *= 2 / w
    model.gen_feat(inp)
    n = coord.shape[1]
    preds = []
    for i in range(0, n, chunk):
        preds.append(model.query_rgb(coord[:, i:i + chunk, :], cell[:, i:i + chunk, :]))
    pred = torch.cat(preds, dim=1)
    pred = (pred * 0.5 + 0.5).clamp(0, 1)
    out = pred.view(h, w, 3).cpu().numpy()
    if out.shape[:2] != (h, w):
        out = _resize_image(out, (h, w))
    return out.astype(np.float32)


LOADERS = {
    "swinir": load_swinir,
    "hat": load_hat,
    "restormer": load_restormer,
    "liif": load_liif,
}

RESTORERS = {
    "swinir": restore_pixelshuffle_sr,
    "hat": restore_pixelshuffle_sr,
    "restormer": restore_restormer,
    "liif": restore_liif,
}

# Grouping for the CEA Table 3 (classical / transformer-restoration / implicit-representation / ours).
GROUPS = {
    "swinir": "transformer_restoration",
    "hat": "transformer_restoration",
    "restormer": "transformer_restoration",
    "liif": "implicit_representation",
}
