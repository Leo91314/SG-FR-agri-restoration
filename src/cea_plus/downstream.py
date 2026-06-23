from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class FrozenPixelSegmenter:
    classifier: Any

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        features = extract_pixel_features(image)
        pred = self.classifier.predict(features)
        return pred.reshape(image.shape[:2]).astype(bool)


def extract_pixel_features(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape HxWx3")

    rgb = image.reshape(-1, 3)
    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]
    eps = 1e-6
    total = r + g + b + eps
    brightness = (r + g + b) / 3.0
    saturation = np.max(rgb, axis=1) - np.min(rgb, axis=1)
    green_excess = g - 0.55 * r - 0.45 * b
    features = np.column_stack(
        [
            r,
            g,
            b,
            r / total,
            g / total,
            b / total,
            brightness,
            saturation,
            green_excess,
            (g - r) / (g + r + eps),
            (g - b) / (g + b + eps),
            (r - b) / (r + b + eps),
        ]
    )
    return features.astype(np.float32)


def _sample_indices(mask: np.ndarray, max_pixels: int, rng: np.random.Generator) -> np.ndarray:
    flat_mask = mask.reshape(-1).astype(bool)
    positive = np.flatnonzero(flat_mask)
    negative = np.flatnonzero(~flat_mask)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("each training sample must contain both mask classes")

    per_class = max(1, max_pixels // 2)
    pos_take = min(positive.size, per_class)
    neg_take = min(negative.size, max_pixels - pos_take)
    if neg_take <= 0:
        neg_take = 1
        pos_take = max(1, max_pixels - neg_take)

    pos_selected = rng.choice(positive, size=pos_take, replace=False)
    neg_selected = rng.choice(negative, size=neg_take, replace=False)
    selected = np.concatenate([pos_selected, neg_selected])
    rng.shuffle(selected)
    return selected


def train_pixel_segmenter(
    samples: list,
    max_pixels_per_sample: int = 2000,
    seed: int = 0,
) -> FrozenPixelSegmenter:
    if not samples:
        raise ValueError("at least one training sample is required")
    if max_pixels_per_sample < 2:
        raise ValueError("max_pixels_per_sample must be >= 2")

    rng = np.random.default_rng(seed)
    feature_blocks = []
    label_blocks = []
    for sample in samples:
        if sample.image.shape[:2] != sample.mask.shape:
            raise ValueError(f"image/mask shape mismatch for {sample.name}")
        indices = _sample_indices(sample.mask, max_pixels_per_sample, rng)
        features = extract_pixel_features(sample.image)
        labels = sample.mask.reshape(-1).astype(np.uint8)
        feature_blocks.append(features[indices])
        label_blocks.append(labels[indices])

    x_train = np.vstack(feature_blocks)
    y_train = np.concatenate(label_blocks)
    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=300,
        random_state=seed,
        solver="lbfgs",
    )
    classifier.fit(x_train, y_train)
    return FrozenPixelSegmenter(classifier=classifier)


def evaluate_frozen_segmenter(segmenter: FrozenPixelSegmenter, image: np.ndarray, mask: np.ndarray) -> float:
    pred = segmenter.predict_mask(image)
    gt = mask.astype(bool)

    intersection_crop = float(np.logical_and(pred, gt).sum())
    union_crop = float(np.logical_or(pred, gt).sum())
    intersection_bg = float(np.logical_and(~pred, ~gt).sum())
    union_bg = float(np.logical_or(~pred, ~gt).sum())
    crop_iou = intersection_crop / union_crop if union_crop else 1.0
    bg_iou = intersection_bg / union_bg if union_bg else 1.0
    return float((crop_iou + bg_iou) / 2.0)


def save_segmenter(segmenter: FrozenPixelSegmenter, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(segmenter, path)


def load_segmenter(path: Path) -> FrozenPixelSegmenter:
    loaded = joblib.load(Path(path))
    if not isinstance(loaded, FrozenPixelSegmenter):
        raise TypeError("loaded object is not a FrozenPixelSegmenter")
    return loaded
