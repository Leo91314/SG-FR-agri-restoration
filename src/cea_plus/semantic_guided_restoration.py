from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .restoration import bicubic_restore, uniform_sharp_restore


@dataclass(frozen=True)
class GuidedFrequencyComponents:
    restored: np.ndarray
    bicubic: np.ndarray
    uniform: np.ndarray
    probability: np.ndarray
    calibrated_probability: np.ndarray
    confidence: np.ndarray
    strength: np.ndarray
    high_frequency: np.ndarray


def _guide_probability(guide_segmenter: object, image: np.ndarray) -> tuple[np.ndarray, bool]:
    if hasattr(guide_segmenter, "predict_probability"):
        prob = np.asarray(guide_segmenter.predict_probability(image), dtype=np.float32)
        return prob.clip(0.0, 1.0), True
    pseudo_mask = np.asarray(guide_segmenter.predict_mask(image), dtype=bool)
    return pseudo_mask.astype(np.float32), False


def _calibrate_probability(probability: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return probability.astype(np.float32).clip(0.0, 1.0)
    if mode == "image_minmax":
        low, high = np.quantile(probability, [0.05, 0.95])
        span = float(high - low)
        if span <= 1e-6:
            return probability.astype(np.float32).clip(0.0, 1.0)
        return ((probability - float(low)) / span).clip(0.0, 1.0).astype(np.float32)
    raise ValueError(f"unknown probability_calibration: {mode}")


def segmenter_guided_frequency_components(
    low_res: np.ndarray,
    output_shape: tuple[int, int],
    guide_segmenter: object,
    inner_boost: float = 0.20,
    outer_boost: float = -0.05,
    mask_blur_sigma: float = 2.0,
    base_strength: float = 0.72,
    uncertain_strength: float | None = None,
    confidence_gamma: float = 1.0,
    scale2_fog_fallback_strength: float | None = None,
    scale2_fog_mean_threshold: float = 0.60,
    scale2_fog_saturation_threshold: float = 0.08,
    probability_calibration: str = "none",
    bright_fog_fallback_strength: float | None = None,
    bright_fog_mean_threshold: float = 0.60,
    bright_fog_saturation_threshold: float = 0.08,
    uniform_mix_weight: float = 0.0,
) -> np.ndarray:
    if not 0.0 <= float(uniform_mix_weight) <= 1.0:
        raise ValueError("uniform_mix_weight must be in [0, 1]")
    up = bicubic_restore(low_res, output_shape)
    probability, has_probability = _guide_probability(guide_segmenter, up)
    if probability.shape != up.shape[:2]:
        raise ValueError(f"guide probability shape {probability.shape} does not match output shape {up.shape[:2]}")
    if float(confidence_gamma) <= 0.0:
        raise ValueError("confidence_gamma must be > 0")

    smooth = ndimage.gaussian_filter(up, sigma=(1.0, 1.0, 0.0))
    high = up - smooth
    calibrated_probability = _calibrate_probability(probability, probability_calibration) if has_probability else probability
    soft_mask = ndimage.gaussian_filter(calibrated_probability.astype(np.float32), sigma=float(mask_blur_sigma))[..., None]
    semantic_strength = float(base_strength) + float(inner_boost) * soft_mask + float(outer_boost) * (1.0 - soft_mask)
    if has_probability:
        confidence = np.abs(probability - 0.5) * 2.0
        confidence = np.power(confidence.clip(0.0, 1.0), float(confidence_gamma))
        reliability = ndimage.gaussian_filter(confidence.astype(np.float32), sigma=float(mask_blur_sigma))[..., None]
    else:
        reliability = np.ones((*up.shape[:2], 1), dtype=np.float32)
    fallback_strength = float(base_strength if uncertain_strength is None else uncertain_strength)
    strength = reliability * semantic_strength + (1.0 - reliability) * fallback_strength
    if scale2_fog_fallback_strength is not None:
        scale_y = float(output_shape[0]) / float(low_res.shape[0])
        scale_x = float(output_shape[1]) / float(low_res.shape[1])
        saturation = float((low_res.max(axis=2) - low_res.min(axis=2)).mean())
        bright_low_saturation = (
            abs(scale_y - 2.0) < 0.25
            and abs(scale_x - 2.0) < 0.25
            and float(low_res.mean()) >= float(scale2_fog_mean_threshold)
            and saturation <= float(scale2_fog_saturation_threshold)
        )
        if bright_low_saturation:
            strength = np.full_like(strength, float(scale2_fog_fallback_strength), dtype=np.float32)
    if bright_fog_fallback_strength is not None:
        saturation = float((low_res.max(axis=2) - low_res.min(axis=2)).mean())
        bright_low_saturation = (
            float(low_res.mean()) >= float(bright_fog_mean_threshold)
            and saturation <= float(bright_fog_saturation_threshold)
        )
        if bright_low_saturation:
            strength = np.full_like(strength, float(bright_fog_fallback_strength), dtype=np.float32)
    guided = np.clip(up + strength * high, 0.0, 1.0).astype(np.float32)
    uniform = uniform_sharp_restore(low_res, output_shape)
    if float(uniform_mix_weight) <= 0.0:
        restored = guided
    else:
        restored = np.clip(
            (1.0 - float(uniform_mix_weight)) * guided + float(uniform_mix_weight) * uniform,
            0.0,
            1.0,
        ).astype(np.float32)
    return GuidedFrequencyComponents(
        restored=restored,
        bicubic=up,
        uniform=uniform,
        probability=probability.astype(np.float32).clip(0.0, 1.0),
        calibrated_probability=calibrated_probability.astype(np.float32).clip(0.0, 1.0),
        confidence=confidence.astype(np.float32).clip(0.0, 1.0) if has_probability else np.ones(up.shape[:2], dtype=np.float32),
        strength=strength[..., 0].astype(np.float32),
        high_frequency=high.astype(np.float32),
    )


def segmenter_guided_frequency_restore(
    low_res: np.ndarray,
    output_shape: tuple[int, int],
    guide_segmenter: object,
    inner_boost: float = 0.20,
    outer_boost: float = -0.05,
    mask_blur_sigma: float = 2.0,
    base_strength: float = 0.72,
    uncertain_strength: float | None = None,
    confidence_gamma: float = 1.0,
    scale2_fog_fallback_strength: float | None = None,
    scale2_fog_mean_threshold: float = 0.60,
    scale2_fog_saturation_threshold: float = 0.08,
    probability_calibration: str = "none",
    bright_fog_fallback_strength: float | None = None,
    bright_fog_mean_threshold: float = 0.60,
    bright_fog_saturation_threshold: float = 0.08,
    uniform_mix_weight: float = 0.0,
) -> np.ndarray:
    return segmenter_guided_frequency_components(
        low_res,
        output_shape,
        guide_segmenter=guide_segmenter,
        inner_boost=inner_boost,
        outer_boost=outer_boost,
        mask_blur_sigma=mask_blur_sigma,
        base_strength=base_strength,
        uncertain_strength=uncertain_strength,
        confidence_gamma=confidence_gamma,
        scale2_fog_fallback_strength=scale2_fog_fallback_strength,
        scale2_fog_mean_threshold=scale2_fog_mean_threshold,
        scale2_fog_saturation_threshold=scale2_fog_saturation_threshold,
        probability_calibration=probability_calibration,
        bright_fog_fallback_strength=bright_fog_fallback_strength,
        bright_fog_mean_threshold=bright_fog_mean_threshold,
        bright_fog_saturation_threshold=bright_fog_saturation_threshold,
        uniform_mix_weight=uniform_mix_weight,
    ).restored
