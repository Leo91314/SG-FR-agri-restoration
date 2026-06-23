from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class SemanticFrequencyComponents:
    restored: np.ndarray
    structure: np.ndarray
    texture: np.ndarray
    alpha: np.ndarray


def _resize(image: np.ndarray, size: tuple[int, int], resample: int = Image.Resampling.BICUBIC) -> np.ndarray:
    pil = Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0))
    return (np.asarray(pil.resize(size, resample=resample)).astype(np.float32) / 255.0).clip(0.0, 1.0)


def bicubic_restore(low_res: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    h, w = output_shape
    return _resize(low_res, (w, h), Image.Resampling.BICUBIC).astype(np.float32)


def uniform_sharp_restore(low_res: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    up = bicubic_restore(low_res, output_shape)
    smooth = ndimage.gaussian_filter(up, sigma=(1.0, 1.0, 0.0))
    return np.clip(up + 0.72 * (up - smooth), 0.0, 1.0).astype(np.float32)


def semantic_frequency_restore(
    low_res: np.ndarray,
    mask: np.ndarray,
    output_shape: tuple[int, int],
    degradation_name: Optional[str] = None,
) -> np.ndarray:
    return semantic_frequency_components(low_res, mask, output_shape, degradation_name=degradation_name).restored


def semantic_edge_aware_restore(
    low_res: np.ndarray,
    mask: np.ndarray,
    output_shape: tuple[int, int],
    degradation_name: Optional[str] = None,
) -> np.ndarray:
    components = semantic_frequency_components(low_res, mask, output_shape, degradation_name=degradation_name)
    structural = uniform_sharp_restore(low_res, output_shape)
    blend = 0.20
    restored = (1.0 - blend) * components.restored + blend * structural
    return np.clip(restored, 0.0, 1.0).astype(np.float32)


def semantic_boundary_guard_restore(
    low_res: np.ndarray,
    mask: np.ndarray,
    output_shape: tuple[int, int],
    degradation_name: Optional[str] = None,
) -> np.ndarray:
    components = semantic_frequency_components(low_res, mask, output_shape, degradation_name=degradation_name)
    up = bicubic_restore(low_res, output_shape)
    if float(up.mean()) <= 0.52:
        return components.restored

    structural = uniform_sharp_restore(low_res, output_shape)
    core = ndimage.binary_erosion(mask, iterations=4)
    if not core.any():
        core = mask
    gate = core.astype(np.float32)[..., None]
    semantic_weight = 0.35
    restored = (1.0 - semantic_weight * gate) * structural + (semantic_weight * gate) * components.restored
    return np.clip(restored, 0.0, 1.0).astype(np.float32)


def _estimate_fog_score(low_res: np.ndarray, degradation_name: Optional[str]) -> float:
    if degradation_name is not None:
        lowered = degradation_name.lower()
        if "fog" in lowered or "mixed" in lowered:
            return 1.0
        return 0.0

    mean_luma = float(low_res.mean())
    mean_saturation = float((low_res.max(axis=2) - low_res.min(axis=2)).mean())
    return float(
        np.clip((mean_luma - 0.34) / 0.16, 0.0, 1.0)
        * np.clip((0.32 - mean_saturation) / 0.12, 0.0, 1.0)
    )


def semantic_frequency_components(
    low_res: np.ndarray,
    mask: np.ndarray,
    output_shape: tuple[int, int],
    modulation: str = "semantic",
    degradation_name: Optional[str] = None,
) -> SemanticFrequencyComponents:
    up = bicubic_restore(low_res, output_shape)
    fog_score = _estimate_fog_score(low_res, degradation_name)

    mask_f = mask.astype(np.float32)[..., None]
    boundary = ndimage.binary_dilation(mask, iterations=2) ^ ndimage.binary_erosion(mask, iterations=1)
    boundary_f = boundary.astype(np.float32)[..., None]

    dehaze_candidate = np.clip((up - 0.20) / 0.80, 0.0, 1.0)
    dehazed = (1.0 - fog_score) * up + fog_score * dehaze_candidate
    smooth = ndimage.gaussian_filter(dehazed, sigma=(0.78, 0.78, 0.0))
    high = dehazed - ndimage.gaussian_filter(dehazed, sigma=(0.65, 0.65, 0.0))
    if modulation == "semantic":
        alpha = np.clip(0.04 + 0.96 * mask_f + 0.62 * boundary_f, 0.0, 1.0)
    elif modulation == "none":
        alpha = np.full_like(mask_f, 0.24, dtype=np.float32)
    elif modulation == "fixed":
        alpha = np.full_like(mask_f, 0.44, dtype=np.float32)
    elif modulation == "structure_only":
        alpha = np.zeros_like(mask_f, dtype=np.float32)
    else:
        raise ValueError(f"unknown modulation: {modulation}")
    restored = smooth + alpha * high

    crop_prior = np.array([0.11, 0.51, 0.13], dtype=np.float32)
    structure = smooth
    if modulation == "semantic":
        prior_weight = 0.12 * fog_score
        restored = (1.0 - prior_weight * mask_f) * restored + (prior_weight * mask_f) * crop_prior
        boosted = restored.copy()
        boosted[..., 1] = np.where(mask, np.maximum(boosted[..., 1], boosted[..., 0] + 0.30), boosted[..., 1])
        boost_weight = 0.60 * fog_score
        restored = (1.0 - boost_weight * mask_f) * restored + (boost_weight * mask_f) * boosted
    elif modulation == "fixed":
        prior_weight = 0.02 * fog_score
        restored = (1.0 - prior_weight * mask_f) * restored + (prior_weight * mask_f) * crop_prior
    restored = (1.0 - fog_score) * up + fog_score * restored
    edge_preserve = 0.25 * fog_score
    if edge_preserve > 0.0:
        restored = (1.0 - edge_preserve) * restored + edge_preserve * uniform_sharp_restore(low_res, output_shape)
    bicubic_preserve = 0.02 * fog_score
    if bicubic_preserve > 0.0:
        restored = (1.0 - bicubic_preserve) * restored + bicubic_preserve * up
    texture = np.clip(0.5 + 4.0 * high, 0.0, 1.0)
    return SemanticFrequencyComponents(
        restored=np.clip(restored, 0.0, 1.0).astype(np.float32),
        structure=np.clip(structure, 0.0, 1.0).astype(np.float32),
        texture=texture.astype(np.float32),
        alpha=alpha[..., 0].astype(np.float32),
    )


def restore_all_methods(
    low_res: np.ndarray,
    mask: np.ndarray,
    output_shape: tuple[int, int],
    degradation_name: Optional[str] = None,
) -> dict[str, np.ndarray]:
    return {
        "bicubic": bicubic_restore(low_res, output_shape),
        "uniform_sharp": uniform_sharp_restore(low_res, output_shape),
        "semantic_frequency": semantic_frequency_components(low_res, mask, output_shape, modulation="semantic", degradation_name=degradation_name).restored,
        "semantic_edge_aware": semantic_edge_aware_restore(low_res, mask, output_shape, degradation_name=degradation_name),
        "semantic_boundary_guard": semantic_boundary_guard_restore(low_res, mask, output_shape, degradation_name=degradation_name),
        "semantic_no_mod": semantic_frequency_components(low_res, mask, output_shape, modulation="none", degradation_name=degradation_name).restored,
        "semantic_fixed_alpha": semantic_frequency_components(low_res, mask, output_shape, modulation="fixed", degradation_name=degradation_name).restored,
        "structure_only": semantic_frequency_components(low_res, mask, output_shape, modulation="structure_only", degradation_name=degradation_name).restored,
    }
