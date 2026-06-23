from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class AgriSample:
    image: np.ndarray
    mask: np.ndarray
    name: str = "sample"

    def with_name(self, name: str) -> "AgriSample":
        return AgriSample(image=self.image, mask=self.mask, name=name)


def _normalize01(values: np.ndarray) -> np.ndarray:
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def make_synthetic_agri_sample(seed: int, size: int = 128) -> AgriSample:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x = xx / max(size - 1, 1)
    y = yy / max(size - 1, 1)

    soil_noise = ndimage.gaussian_filter(rng.normal(0.0, 1.0, (size, size)), sigma=2.0)
    soil_noise = _normalize01(soil_noise) - 0.5
    image = np.zeros((size, size, 3), dtype=np.float32)
    image[..., 0] = 0.42 + 0.09 * soil_noise
    image[..., 1] = 0.30 + 0.07 * soil_noise
    image[..., 2] = 0.16 + 0.05 * soil_noise

    mask = np.zeros((size, size), dtype=bool)
    row_centers = np.linspace(0.16, 0.84, 4)
    for idx, center in enumerate(row_centers):
        waviness = 0.025 * np.sin(2 * np.pi * (y * 1.7 + idx * 0.23))
        distance = np.abs(x - center - waviness)
        row = distance < 0.032
        leaf_texture = 0.5 + 0.5 * np.sin(2 * np.pi * (y * 16 + x * 4 + idx))
        row &= leaf_texture > 0.20
        mask |= row

    mask = ndimage.binary_dilation(mask, iterations=1)
    crop_noise = ndimage.gaussian_filter(rng.normal(0.0, 1.0, (size, size)), sigma=0.7)
    crop_noise = _normalize01(crop_noise) - 0.5
    edge = ndimage.binary_dilation(mask, iterations=2) ^ ndimage.binary_erosion(mask, iterations=1)

    image[mask, 0] = 0.10 + 0.04 * crop_noise[mask]
    image[mask, 1] = 0.48 + 0.18 * crop_noise[mask]
    image[mask, 2] = 0.13 + 0.05 * crop_noise[mask]
    image[edge, 1] = np.clip(image[edge, 1] + 0.13, 0.0, 1.0)
    image[edge, 0] = np.clip(image[edge, 0] - 0.03, 0.0, 1.0)

    return AgriSample(image=np.clip(image, 0.0, 1.0).astype(np.float32), mask=mask, name=f"synthetic_{seed}")
