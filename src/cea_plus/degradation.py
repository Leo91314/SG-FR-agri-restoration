from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .synthesis import AgriSample


@dataclass(frozen=True)
class DegradationConfig:
    name: str = "x4_mixed"
    scale: int = 4
    blur_sigma: float = 0.55
    noise_sigma: float = 0.014
    rain_density: int = 34
    fog_strength: float = 0.24
    jpeg_quality: Optional[int] = None


@dataclass(frozen=True)
class DegradedSample:
    low_res: np.ndarray
    gt: np.ndarray
    mask: np.ndarray
    degradation_name: str = "x4_mixed"
    scale: int = 4


def _resize(image: np.ndarray, size: tuple[int, int], resample: int) -> np.ndarray:
    pil = Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0))
    return (np.asarray(pil.resize(size, resample=resample)).astype(np.float32) / 255.0).clip(0.0, 1.0)


def _add_rain(image: np.ndarray, seed: int, density: int = 34) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    layer = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(layer)
    for _ in range(density):
        x = int(rng.integers(-w // 4, w))
        y = int(rng.integers(0, h))
        length = int(rng.integers(max(4, h // 9), max(5, h // 4)))
        slant = int(rng.integers(-2, 3))
        draw.line((x, y, x + slant, y + length), fill=int(rng.integers(105, 185)), width=1)
    rain = np.asarray(layer).astype(np.float32) / 255.0
    rain = ndimage.gaussian_filter(rain, sigma=(0.6, 0.2))
    return np.clip(image * (1.0 - 0.16 * rain[..., None]) + 0.34 * rain[..., None], 0.0, 1.0)


def _apply_jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    from io import BytesIO

    buffer = BytesIO()
    Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0)).save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return (np.asarray(Image.open(buffer).convert("RGB")).astype(np.float32) / 255.0).clip(0.0, 1.0)


def build_degradation_plan(scales: tuple[int, ...] = (4,), modes: tuple[str, ...] = ("mixed",)) -> list[DegradationConfig]:
    configs: list[DegradationConfig] = []
    for scale in scales:
        for mode in modes:
            if mode == "rain":
                configs.append(DegradationConfig(name=f"x{scale}_rain", scale=scale, rain_density=42, fog_strength=0.0, noise_sigma=0.008))
            elif mode == "fog":
                configs.append(DegradationConfig(name=f"x{scale}_fog", scale=scale, rain_density=0, fog_strength=0.30, noise_sigma=0.008))
            elif mode == "noise":
                configs.append(DegradationConfig(name=f"x{scale}_noise", scale=scale, rain_density=0, fog_strength=0.0, noise_sigma=0.025))
            elif mode == "blur":
                configs.append(DegradationConfig(name=f"x{scale}_blur", scale=scale, rain_density=0, fog_strength=0.0, blur_sigma=0.95, noise_sigma=0.004))
            elif mode == "jpeg":
                configs.append(DegradationConfig(name=f"x{scale}_jpeg", scale=scale, rain_density=0, fog_strength=0.0, noise_sigma=0.004, jpeg_quality=42))
            elif mode == "mixed":
                configs.append(DegradationConfig(name=f"x{scale}_mixed", scale=scale, rain_density=34, fog_strength=0.24, noise_sigma=0.014, jpeg_quality=55))
            else:
                raise ValueError(f"unknown degradation mode: {mode}")
    return configs


def degrade_sample(sample: AgriSample, scale: int = 4, seed: int = 0, config: Optional[DegradationConfig] = None) -> DegradedSample:
    config = config or DegradationConfig(scale=scale, name=f"x{scale}_mixed")
    scale = config.scale
    if scale < 1:
        raise ValueError("scale must be >= 1")
    h, w = sample.image.shape[:2]
    if h % scale or w % scale:
        raise ValueError("image size must be divisible by scale")

    blurred = ndimage.gaussian_filter(sample.image, sigma=(config.blur_sigma, config.blur_sigma, 0.0))
    low = _resize(blurred, (w // scale, h // scale), Image.Resampling.BICUBIC)
    if config.rain_density > 0:
        low = _add_rain(low, seed=seed, density=config.rain_density)
    if config.fog_strength > 0:
        low = low * (1.0 - config.fog_strength) + config.fog_strength
    if config.jpeg_quality is not None:
        low = _apply_jpeg(low, config.jpeg_quality)
    rng = np.random.default_rng(seed + 1009)
    if config.noise_sigma > 0:
        low = low + rng.normal(0.0, config.noise_sigma, low.shape).astype(np.float32)
    return DegradedSample(
        low_res=np.clip(low, 0.0, 1.0).astype(np.float32),
        gt=sample.image,
        mask=sample.mask,
        degradation_name=config.name,
        scale=scale,
    )
