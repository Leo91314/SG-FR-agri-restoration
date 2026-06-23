from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .synthesis import AgriSample


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _read_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return (np.asarray(image).astype(np.float32) / 255.0).clip(0.0, 1.0)


def _read_mask(path: Path) -> np.ndarray:
    mask = Image.open(path).convert("L")
    values = np.asarray(mask)
    return values > 127


def _read_label_values(path: Path) -> np.ndarray:
    values = np.asarray(Image.open(path))
    if values.ndim == 3:
        values = values[..., 0]
    return values


def _image_files(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES and not item.name.startswith("."))


def _find_child_dir(parent: Path, names: tuple[str, ...]) -> Path:
    names_lower = {name.lower() for name in names}
    for item in sorted(parent.iterdir()):
        if item.is_dir() and item.name.lower() in names_lower:
            return item
    matches = sorted(item for item in parent.rglob("*") if item.is_dir() and item.name.lower() in names_lower)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"none of {names} found under {parent}")


def _normalise_loveda_split(dataset_root: Path, split: str) -> Path:
    aliases = {
        "train": "Train",
        "training": "Train",
        "val": "Val",
        "valid": "Val",
        "validation": "Val",
    }
    split_name = aliases.get(split.lower(), split)
    candidate = dataset_root / split_name
    if candidate.exists():
        return candidate
    for item in sorted(dataset_root.iterdir()):
        if item.is_dir() and item.name.lower() == split_name.lower():
            return item
    raise FileNotFoundError(f"LoveDA split not found: {dataset_root / split_name}")


def _loveda_domain_roots(split_root: Path, domain: Optional[str]) -> list[Path]:
    if domain is None or domain.lower() in {"all", "both", "*"}:
        return sorted(item for item in split_root.iterdir() if item.is_dir() and not item.name.startswith("."))
    for item in sorted(split_root.iterdir()):
        if item.is_dir() and item.name.lower() == domain.lower():
            return [item]
    raise FileNotFoundError(f"LoveDA domain not found: {split_root / domain}")


def _crop_around_mask(
    image: np.ndarray,
    mask: np.ndarray,
    crop_size: Optional[int],
    strategy: str = "mask_center",
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if crop_size is None:
        return image, mask
    h, w = mask.shape
    size = min(int(crop_size), h, w)
    if size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    if strategy == "center":
        center_y = h // 2
        center_x = w // 2
    elif strategy == "random":
        generator = rng or np.random.default_rng(0)
        top = int(generator.integers(0, h - size + 1))
        left = int(generator.integers(0, w - size + 1))
        return image[top : top + size, left : left + size], mask[top : top + size, left : left + size]
    elif strategy == "mask_center" and mask.any():
        ys, xs = np.nonzero(mask)
        center_y = int(np.mean(ys))
        center_x = int(np.mean(xs))
    elif strategy == "mask_center":
        center_y = h // 2
        center_x = w // 2
    else:
        raise ValueError(f"unknown crop_strategy: {strategy}")
    top = min(max(center_y - size // 2, 0), h - size)
    left = min(max(center_x - size // 2, 0), w - size)
    return image[top : top + size, left : left + size], mask[top : top + size, left : left + size]


def load_agri_dataset(image_dir: Path, mask_dir: Path, limit: Optional[int] = None) -> list[AgriSample]:
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"mask_dir not found: {mask_dir}")

    masks_by_stem = {path.stem: path for path in _image_files(mask_dir)}
    samples: list[AgriSample] = []
    for image_path in _image_files(image_dir):
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is None:
            continue
        image = _read_rgb(image_path)
        mask = _read_mask(mask_path)
        if image.shape[:2] != mask.shape:
            raise ValueError(f"image/mask shape mismatch for {image_path.name}")
        samples.append(AgriSample(image=image.astype(np.float32), mask=mask.astype(bool), name=image_path.stem))
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise ValueError("no paired image/mask samples found")
    return samples


def load_loveda_dataset(
    dataset_root: Path,
    split: str = "Val",
    domain: Optional[str] = "Rural",
    target_labels: tuple[int, ...] = (7,),
    limit: Optional[int] = None,
    crop_size: Optional[int] = None,
    crop_strategy: str = "mask_center",
    crop_seed: int = 0,
    require_mask: bool = True,
) -> list[AgriSample]:
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"LoveDA root not found: {dataset_root}")
    if not target_labels:
        raise ValueError("target_labels must contain at least one class id")

    split_root = _normalise_loveda_split(dataset_root, split)
    samples: list[AgriSample] = []
    rng = np.random.default_rng(crop_seed)
    for domain_root in _loveda_domain_roots(split_root, domain):
        image_dir = _find_child_dir(domain_root, ("images_png", "images", "image", "imgs"))
        mask_dir = _find_child_dir(domain_root, ("masks_png", "masks", "mask", "labels_png", "labels", "annotations"))
        masks_by_stem = {path.stem: path for path in _image_files(mask_dir)}
        for image_path in _image_files(image_dir):
            mask_path = masks_by_stem.get(image_path.stem)
            if mask_path is None:
                continue
            image = _read_rgb(image_path)
            labels = _read_label_values(mask_path)
            mask = np.isin(labels, np.asarray(target_labels))
            if image.shape[:2] != mask.shape:
                raise ValueError(f"LoveDA image/mask shape mismatch for {image_path.name}")
            image, mask = _crop_around_mask(image, mask.astype(bool), crop_size, strategy=crop_strategy, rng=rng)
            if require_mask and not mask.any():
                continue
            sample_name = f"{split_root.name}_{domain_root.name}_{image_path.stem}"
            samples.append(AgriSample(image=image.astype(np.float32), mask=mask.astype(bool), name=sample_name))
            if limit is not None and len(samples) >= limit:
                return samples
    if not samples:
        raise ValueError(f"no LoveDA samples loaded from split={split}, domain={domain}, labels={target_labels}")
    return samples


def cwfid_split_ids(dataset_root: Path, split: str) -> list[int]:
    """Return CWFID image ids for a split, with train/test leakage removed.

    The official CWFID train_test_split.yaml lists image 28 in BOTH train and test
    (a direct train/test leak). We keep the official test set intact and drop any
    overlapping id from train, so the frozen segmenter is never trained on a test image.
    """
    import yaml

    dataset_root = Path(dataset_root)
    split_file = dataset_root / "train_test_split.yaml"
    splits = yaml.safe_load(split_file.read_text(encoding="utf-8"))
    train_ids = [int(i) for i in splits.get("train", [])]
    test_ids = [int(i) for i in splits.get("test", [])]
    overlap = set(train_ids) & set(test_ids)
    key = split.lower()
    if key == "train":
        return [i for i in train_ids if i not in overlap]
    if key == "test":
        return list(test_ids)
    return [int(i) for i in splits.get(key, [])]


def load_cwfid_dataset(
    dataset_root: Path,
    split: str = "train",
    limit: Optional[int] = None,
    crop_size: Optional[int] = None,
    crop_strategy: str = "mask_center",
    crop_seed: int = 0,
    require_mask: bool = True,
) -> list[AgriSample]:
    """CWFID carrot field dataset (Haug & Ostermann 2014).

    Ground truth is the colour annotation: red=weed, green=crop. We form a binary vegetation
    (plants vs soil) target = red|green, matching the WeedsGalore "any annotated plant" mask, so the
    downstream task is directly comparable across the two agricultural datasets.
    """
    import yaml

    dataset_root = Path(dataset_root)
    image_dir = dataset_root / "images"
    annot_dir = dataset_root / "annotations"
    if not image_dir.exists() or not annot_dir.exists():
        raise FileNotFoundError(f"CWFID images/annotations not found under {dataset_root}")

    split_file = dataset_root / "train_test_split.yaml"
    if split_file.exists():
        ids = cwfid_split_ids(dataset_root, split)
    else:
        ids = sorted(int(p.stem.split("_")[0]) for p in _image_files(image_dir))

    rng = np.random.default_rng(crop_seed)
    samples: list[AgriSample] = []
    for sid in ids:
        image_path = image_dir / f"{sid:03d}_image.png"
        annot_path = annot_dir / f"{sid:03d}_annotation.png"
        if not image_path.exists() or not annot_path.exists():
            continue
        image = _read_rgb(image_path)
        annot = np.asarray(Image.open(annot_path).convert("RGB"))
        mask = (annot[..., 0] > 127) | (annot[..., 1] > 127)
        if image.shape[:2] != mask.shape:
            raise ValueError(f"CWFID image/mask shape mismatch for {image_path.name}")
        image, mask = _crop_around_mask(image, mask.astype(bool), crop_size, strategy=crop_strategy, rng=rng)
        if require_mask and not mask.any():
            continue
        samples.append(AgriSample(image=image.astype(np.float32), mask=mask.astype(bool), name=f"cwfid_{sid:03d}"))
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise ValueError(f"no CWFID samples loaded from split={split}")
    return samples


def load_cofly_dataset(
    dataset_root: Path,
    split: str = "train",
    split_index: int = 1,
    limit: Optional[int] = None,
    crop_size: Optional[int] = None,
    crop_strategy: str = "mask_center",
    crop_seed: int = 0,
    require_mask: bool = True,
) -> list[AgriSample]:
    """CoFly-WeedDB real UAV cotton-field dataset (Krestenitis et al. 2022, Zenodo 6697343).

    Real DJI Phantom-4 Pro imagery (1280x720). labels_1d encodes 0=background, 1/2/3=weed species.
    The binary target is weed vs background (labels_1d>0), i.e. UAV weed mapping -- the most
    agronomically relevant task for this dataset and a genuinely real-capture agricultural test.
    """
    dataset_root = Path(dataset_root)
    image_dir = dataset_root / "images"
    label_dir = dataset_root / "labels_1d"
    if not image_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"CoFly images/labels_1d not found under {dataset_root}")

    split_file = dataset_root / f"{split.lower()}_split{int(split_index)}.txt"
    labels_by_stem = {path.stem: path for path in _image_files(label_dir)}
    if split_file.exists():
        names = [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        stems = [Path(n).stem for n in names]
    else:
        stems = sorted(labels_by_stem)

    rng = np.random.default_rng(crop_seed)
    images_by_stem = {path.stem: path for path in _image_files(image_dir)}
    samples: list[AgriSample] = []
    for stem in stems:
        image_path = images_by_stem.get(stem)
        label_path = labels_by_stem.get(stem)
        if image_path is None or label_path is None:
            continue
        image = _read_rgb(image_path)
        labels = _read_label_values(label_path)
        mask = labels > 0
        if image.shape[:2] != mask.shape:
            raise ValueError(f"CoFly image/mask shape mismatch for {image_path.name}")
        image, mask = _crop_around_mask(image, mask.astype(bool), crop_size, strategy=crop_strategy, rng=rng)
        if require_mask and not mask.any():
            continue
        samples.append(AgriSample(image=image.astype(np.float32), mask=mask.astype(bool), name=stem[:24]))
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise ValueError(f"no CoFly samples loaded from split={split}{split_index}")
    return samples


def _read_weedsgalore_band(path: Path) -> np.ndarray:
    band = np.asarray(Image.open(path)).astype(np.float32)
    max_value = 65535.0 if band.max() > 255 else 255.0
    return np.clip(band / max_value, 0.0, 1.0)


def _find_weedsgalore_sample_root(dataset_root: Path, sample_id: str) -> Path:
    date = sample_id[:10]
    candidate = dataset_root / date
    if candidate.exists():
        return candidate
    matches = sorted(path for path in dataset_root.iterdir() if path.is_dir() and (path / "semantics" / f"{sample_id}.png").exists())
    if not matches:
        raise FileNotFoundError(f"WeedsGalore sample not found: {sample_id}")
    return matches[0]


def load_weedsgalore_dataset(dataset_root: Path, split: str = "test", limit: Optional[int] = None) -> list[AgriSample]:
    dataset_root = Path(dataset_root)
    split_path = dataset_root / "splits" / f"{split}.txt"
    if not split_path.exists():
        raise FileNotFoundError(f"WeedsGalore split not found: {split_path}")

    sample_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    samples: list[AgriSample] = []
    for sample_id in sample_ids:
        sample_root = _find_weedsgalore_sample_root(dataset_root, sample_id)
        image_dir = sample_root / "images"
        semantic_path = sample_root / "semantics" / f"{sample_id}.png"
        bands = [
            _read_weedsgalore_band(image_dir / f"{sample_id}_R.png"),
            _read_weedsgalore_band(image_dir / f"{sample_id}_G.png"),
            _read_weedsgalore_band(image_dir / f"{sample_id}_B.png"),
        ]
        image = np.stack(bands, axis=-1).astype(np.float32)
        labels = np.asarray(Image.open(semantic_path).convert("L"))
        mask = labels > 0
        samples.append(AgriSample(image=image, mask=mask.astype(bool), name=sample_id))
        if limit is not None and len(samples) >= limit:
            break
    if not samples:
        raise ValueError(f"no WeedsGalore samples loaded from split: {split}")
    return samples
