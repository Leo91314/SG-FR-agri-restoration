from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .restoration import bicubic_restore, uniform_sharp_restore


@dataclass(frozen=True)
class NoLeakFrequencyComponents:
    restored: np.ndarray
    structure: np.ndarray
    texture: np.ndarray
    alpha: np.ndarray
    pseudo_mask: np.ndarray


def estimate_pseudo_mask_from_low_res(low_res: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    up = bicubic_restore(low_res, output_shape)
    green_excess = up[..., 1] - 0.55 * up[..., 0] - 0.45 * up[..., 2]
    threshold = max(0.06, float(np.quantile(green_excess, 0.72)))
    mask = green_excess > threshold
    if mask.any():
        mask = ndimage.binary_opening(mask, iterations=1)
        mask = ndimage.binary_closing(mask, iterations=2)
    return mask.astype(bool)


def semantic_frequency_no_leak_components(
    low_res: np.ndarray,
    output_shape: tuple[int, int],
    pseudo_mask: np.ndarray | None = None,
) -> NoLeakFrequencyComponents:
    up = bicubic_restore(low_res, output_shape)
    if pseudo_mask is None:
        pseudo_mask = estimate_pseudo_mask_from_low_res(low_res, output_shape)
    if pseudo_mask.shape != up.shape[:2]:
        raise ValueError(f"pseudo_mask shape {pseudo_mask.shape} does not match output shape {up.shape[:2]}")

    mask = pseudo_mask.astype(bool)
    mask_f = mask.astype(np.float32)[..., None]
    boundary = ndimage.binary_dilation(mask, iterations=2) ^ ndimage.binary_erosion(mask, iterations=1)
    boundary_f = boundary.astype(np.float32)[..., None]

    structure = ndimage.gaussian_filter(up, sigma=(0.78, 0.78, 0.0))
    high = up - ndimage.gaussian_filter(up, sigma=(0.65, 0.65, 0.0))
    alpha = np.clip(0.08 + 0.42 * mask_f + 0.22 * boundary_f, 0.0, 0.60)
    restored = np.clip(structure + alpha * high, 0.0, 1.0).astype(np.float32)
    texture = np.clip(0.5 + 4.0 * high, 0.0, 1.0).astype(np.float32)

    return NoLeakFrequencyComponents(
        restored=restored,
        structure=np.clip(structure, 0.0, 1.0).astype(np.float32),
        texture=texture,
        alpha=alpha[..., 0].astype(np.float32),
        pseudo_mask=mask,
    )


def semantic_frequency_no_leak_restore(
    low_res: np.ndarray,
    output_shape: tuple[int, int],
    pseudo_mask: np.ndarray | None = None,
) -> np.ndarray:
    return semantic_frequency_no_leak_components(low_res, output_shape, pseudo_mask=pseudo_mask).restored


def restore_no_leak_methods(
    low_res: np.ndarray,
    output_shape: tuple[int, int],
    pseudo_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    return {
        "bicubic": bicubic_restore(low_res, output_shape),
        "uniform_sharp": uniform_sharp_restore(low_res, output_shape),
        "semantic_frequency_no_leak": semantic_frequency_no_leak_restore(low_res, output_shape, pseudo_mask=pseudo_mask),
    }
