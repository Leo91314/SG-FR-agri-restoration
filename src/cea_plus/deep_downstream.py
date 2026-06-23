from dataclasses import dataclass
from pathlib import Path
import csv
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn

from .degradation import DegradationConfig, degrade_sample
from .downstream import evaluate_frozen_segmenter
from .restoration import restore_all_methods
from .statistics import paired_significance


@dataclass(frozen=True)
class FrozenTorchSegmenter:
    model: nn.Module
    device: torch.device
    input_size: int
    architecture: str

    def predict_probability(self, image: np.ndarray) -> np.ndarray:
        self.model.eval()
        h, w = image.shape[:2]
        resized = _resize_image(image, (self.input_size, self.input_size))
        tensor = _image_to_tensor(resized, self.device)
        with torch.no_grad():
            logits = _forward_logits(self.model, tensor)
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()[0]
        return prob.astype(np.float32).clip(0.0, 1.0)

    def predict_mask(self, image: np.ndarray) -> np.ndarray:
        return self.predict_probability(image) >= 0.5


@dataclass(frozen=True)
class DeepDownstreamSummary:
    sample_count: int
    metrics_path: Path
    report_path: Path
    model_path: Path


class TinySegNet(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, num_classes, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.body(image)


def _resize_image(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0))
    resized = pil.resize((shape[1], shape[0]), resample=Image.Resampling.BILINEAR)
    return (np.asarray(resized).astype(np.float32) / 255.0).clip(0.0, 1.0)


def _resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(np.uint8(mask.astype(np.uint8) * 255))
    resized = pil.resize((shape[1], shape[0]), resample=Image.Resampling.NEAREST)
    return np.asarray(resized) > 127


def _image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image.transpose(2, 0, 1)[None]).float().to(device)


def _mask_to_tensor(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.int64)[None]).long().to(device)


def _make_model(architecture: str) -> nn.Module:
    if architecture == "tiny":
        return TinySegNet(num_classes=2)
    if architecture == "deeplabv3":
        from torchvision.models.segmentation import deeplabv3_resnet50

        return deeplabv3_resnet50(weights=None, weights_backbone=None, num_classes=2, aux_loss=None)
    if architecture == "deeplabv3plus":
        import segmentation_models_pytorch as smp

        return smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=2)
    if architecture == "deeplabv3plus_imagenet":
        import segmentation_models_pytorch as smp

        model = smp.DeepLabV3Plus(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=2)
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        return model
    if architecture == "segformer_b0":
        import segmentation_models_pytorch as smp

        return smp.Segformer(encoder_name="mit_b0", encoder_weights=None, in_channels=3, classes=2)
    if architecture == "segformer_b0_imagenet":
        import segmentation_models_pytorch as smp

        model = smp.Segformer(encoder_name="mit_b0", encoder_weights="imagenet", in_channels=3, classes=2)
        for parameter in model.encoder.parameters():
            parameter.requires_grad = False
        return model
    if architecture == "deeplabv3_imagenet":
        from torchvision.models import ResNet50_Weights
        from torchvision.models.segmentation import deeplabv3_resnet50

        model = deeplabv3_resnet50(weights=None, weights_backbone=ResNet50_Weights.DEFAULT, num_classes=2, aux_loss=None)
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        return model
    raise ValueError(f"unknown architecture: {architecture}")


def _forward_logits(model: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
    output = model(tensor)
    if isinstance(output, dict):
        return output["out"]
    return output


def _freeze_batchnorm(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            child.eval()


def _crop_pair(image: np.ndarray, mask: np.ndarray, crop_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    if h < crop_size or w < crop_size:
        return _resize_image(image, (crop_size, crop_size)), _resize_mask(mask, (crop_size, crop_size))
    y = int(rng.integers(0, h - crop_size + 1))
    x = int(rng.integers(0, w - crop_size + 1))
    return image[y : y + crop_size, x : x + crop_size], mask[y : y + crop_size, x : x + crop_size]


def train_torch_segmenter(
    train_samples: list,
    architecture: str = "tiny",
    steps: int = 80,
    crop_size: int = 128,
    input_size: Optional[int] = None,
    seed: int = 0,
    device: Optional[torch.device] = None,
    learning_rate: float = 0.001,
) -> FrozenTorchSegmenter:
    if not train_samples:
        raise ValueError("at least one training sample is required")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = device or torch.device("cpu")
    input_size = input_size or crop_size
    model = _make_model(architecture).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for step in range(max(1, steps)):
        _freeze_batchnorm(model)
        sample = train_samples[step % len(train_samples)]
        crop_image, crop_mask = _crop_pair(sample.image, sample.mask, crop_size, rng)
        crop_image = _resize_image(crop_image, (input_size, input_size))
        crop_mask = _resize_mask(crop_mask, (input_size, input_size))
        logits = _forward_logits(model, _image_to_tensor(crop_image, device))
        target = _mask_to_tensor(crop_mask, device)
        positive = float(crop_mask.mean())
        weights = torch.tensor(
            [1.0 / max(1.0 - positive, 0.05), 1.0 / max(positive, 0.05)],
            dtype=torch.float32,
            device=device,
        )
        weights = weights / weights.mean()
        loss = F.cross_entropy(logits, target, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return FrozenTorchSegmenter(model=model, device=device, input_size=input_size, architecture=architecture)


def _write_deep_downstream_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["sample", "degradation", "scale", "method", "deep_frozen_miou"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_deep_downstream_report(
    path: Path,
    rows: list[dict[str, object]],
    architecture: str,
    model_path: Path,
    seed: int,
) -> None:
    methods = sorted({str(row["method"]) for row in rows})
    lines = [
        "# Deep frozen downstream report",
        "",
        f"- Architecture: `{architecture}`",
        f"- Frozen model: `{model_path}`",
        "",
        "| Method | Deep frozen mIoU |",
        "|---|---:|",
    ]
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        value = float(np.mean([float(row["deep_frozen_miou"]) for row in method_rows]))
        lines.append(f"| {method} | {value:.4f} |")
    if {"bicubic", "semantic_frequency"}.issubset(methods):
        base = _paired_values(rows, "bicubic")
        sem = _paired_values(rows, "semantic_frequency")
        lines.extend(["", "## Semantic Frequency vs Bicubic", ""])
        if len(base) >= 2:
            stats = paired_significance(base, sem, seed=seed)
            lines.extend(
                [
                    "| Metric | Mean Delta | t-test p | Wilcoxon p | Bootstrap 95% CI |",
                    "|---|---:|---:|---:|---:|",
                    f"| deep_frozen_miou | {stats.mean_delta:.6f} | {stats.ttest_p:.6g} | {stats.wilcoxon_p:.6g} | "
                    f"[{stats.bootstrap_ci_low:.6f}, {stats.bootstrap_ci_high:.6f}] |",
                ]
            )
        else:
            lines.append("- 配对样本少于 2，跳过显著性统计。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paired_values(rows: list[dict[str, object]], method: str) -> np.ndarray:
    method_rows = sorted(
        [row for row in rows if row["method"] == method],
        key=lambda item: (str(item["sample"]), str(item["degradation"]), str(item["scale"])),
    )
    return np.array([float(row["deep_frozen_miou"]) for row in method_rows], dtype=np.float64)


def run_deep_downstream_experiment(
    output_dir: Path,
    train_samples: list,
    test_samples: list,
    degradation_plan: list[DegradationConfig],
    architecture: str = "deeplabv3",
    steps: int = 80,
    crop_size: int = 128,
    eval_size: int = 256,
    seed: int = 7,
    device: Optional[torch.device] = None,
    methods: tuple[str, ...] = (
        "bicubic",
        "uniform_sharp",
        "semantic_frequency",
        "structure_only",
        "semantic_edge_aware",
        "semantic_boundary_guard",
    ),
) -> DeepDownstreamSummary:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    segmenter = train_torch_segmenter(
        train_samples=train_samples,
        architecture=architecture,
        steps=steps,
        crop_size=crop_size,
        input_size=eval_size,
        seed=seed,
        device=device,
    )
    model_path = output_dir / f"frozen_{architecture}.pt"
    torch.save({"architecture": architecture, "state_dict": segmenter.model.state_dict(), "input_size": eval_size}, model_path)

    rows: list[dict[str, object]] = []
    for sample_idx, sample in enumerate(test_samples):
        for config_idx, config in enumerate(degradation_plan):
            degraded = degrade_sample(sample, config=config, seed=seed * 1000 + sample_idx * 100 + config_idx)
            restorations = restore_all_methods(
                degraded.low_res,
                degraded.mask,
                degraded.gt.shape[:2],
                degradation_name=degraded.degradation_name,
            )
            for method in methods:
                restored = restorations[method]
                resized_image = _resize_image(restored, (eval_size, eval_size))
                resized_mask = _resize_mask(degraded.mask, (eval_size, eval_size))
                rows.append(
                    {
                        "sample": sample.name,
                        "degradation": degraded.degradation_name,
                        "scale": degraded.scale,
                        "method": method,
                        "deep_frozen_miou": evaluate_frozen_segmenter(segmenter, resized_image, resized_mask),
                    }
                )

    metrics_path = output_dir / "deep_downstream_metrics.csv"
    report_path = output_dir / "deep_downstream_report.md"
    _write_deep_downstream_metrics(metrics_path, rows)
    _write_deep_downstream_report(report_path, rows, architecture=architecture, model_path=model_path, seed=seed)
    return DeepDownstreamSummary(
        sample_count=len(_paired_values(rows, "bicubic")),
        metrics_path=metrics_path,
        report_path=report_path,
        model_path=model_path,
    )
