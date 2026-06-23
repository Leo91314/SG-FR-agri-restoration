from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from PIL import Image

from .degradation import DegradationConfig, degrade_sample
from .semantic_guided_restoration import segmenter_guided_frequency_components
from .synthesis import AgriSample


@dataclass(frozen=True)
class GuidedFrequencyCaseExport:
    output_dir: Path
    summary_json: Path
    spectrum_json: Path
    restored_png: Path
    probability_png: Path
    confidence_png: Path
    strength_png: Path


def _save_rgb(path: Path, image: np.ndarray) -> None:
    array = np.uint8(np.clip(image, 0.0, 1.0) * 255.0)
    Image.fromarray(array).save(path)


def _save_heatmap(path: Path, values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        scaled = np.zeros(values.shape, dtype=np.float32)
    else:
        low = float(np.min(finite))
        high = float(np.max(finite))
        span = max(high - low, 1e-6)
        scaled = ((values - low) / span).clip(0.0, 1.0)
    red = scaled
    green = 1.0 - np.abs(scaled - 0.5) * 2.0
    blue = 1.0 - scaled
    heatmap = np.stack([red, green.clip(0.0, 1.0), blue], axis=-1)
    _save_rgb(path, heatmap)


def _high_frequency_energy(image: np.ndarray) -> float:
    gray = np.asarray(image, dtype=np.float32).mean(axis=2)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.abs(spectrum)
    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - h / 2.0) ** 2 + (xx - w / 2.0) ** 2)
    high_mask = radius >= 0.25 * min(h, w)
    return float(np.mean(magnitude[high_mask]))


def export_guided_frequency_case(
    output_dir: Path,
    sample: AgriSample,
    degradation_config: DegradationConfig,
    guide_segmenter: object,
    seed: int = 7,
    prefix: str = "case",
    **restore_kwargs,
) -> GuidedFrequencyCaseExport:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    degraded = degrade_sample(sample, config=degradation_config, seed=seed)
    components = segmenter_guided_frequency_components(
        degraded.low_res,
        degraded.gt.shape[:2],
        guide_segmenter=guide_segmenter,
        **restore_kwargs,
    )

    restored_png = output_dir / f"{prefix}_restored.png"
    probability_png = output_dir / f"{prefix}_pseudo_probability.png"
    confidence_png = output_dir / f"{prefix}_confidence.png"
    strength_png = output_dir / f"{prefix}_frequency_strength.png"
    spectrum_json = output_dir / f"{prefix}_spectrum.json"
    summary_json = output_dir / f"{prefix}_summary.json"

    _save_rgb(restored_png, components.restored)
    _save_heatmap(probability_png, components.calibrated_probability)
    _save_heatmap(confidence_png, components.confidence)
    _save_heatmap(strength_png, components.strength)

    spectrum = {
        "bicubic_high_frequency_energy": _high_frequency_energy(components.bicubic),
        "uniform_high_frequency_energy": _high_frequency_energy(components.uniform),
        "restored_high_frequency_energy": _high_frequency_energy(components.restored),
    }
    spectrum_json.write_text(json.dumps(spectrum, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "sample": sample.name,
        "degradation": degradation_config.name,
        "scale": degradation_config.scale,
        "restoration_inputs": ["low_res", "output_shape", "guide_prediction"],
        "gt_mask_usage": "visualization_only",
        "mean_probability": float(np.mean(components.probability)),
        "mean_confidence": float(np.mean(components.confidence)),
        "mean_strength": float(np.mean(components.strength)),
        "spectrum": spectrum,
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return GuidedFrequencyCaseExport(
        output_dir=output_dir,
        summary_json=summary_json,
        spectrum_json=spectrum_json,
        restored_png=restored_png,
        probability_png=probability_png,
        confidence_png=confidence_png,
        strength_png=strength_png,
    )
