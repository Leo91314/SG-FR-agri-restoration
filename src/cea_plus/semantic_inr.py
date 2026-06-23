from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .restoration import bicubic_restore


@dataclass(frozen=True)
class SemanticINROutput:
    restored: torch.Tensor
    structure: torch.Tensor
    texture: torch.Tensor
    alpha: torch.Tensor
    base: torch.Tensor
    semantic_logits: torch.Tensor


def _coord_grid(batch: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    y = torch.linspace(-1.0, 1.0, height, device=device)
    x = torch.linspace(-1.0, 1.0, width, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([xx, yy], dim=0).unsqueeze(0).repeat(batch, 1, 1, 1)
    return coords


def _image_to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image.transpose(2, 0, 1)[None]).float().to(device)


def _mask_to_tensor(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(mask.astype(np.float32)[None, None]).float().to(device)


def _tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().clamp(0.0, 1.0).numpy()[0].transpose(1, 2, 0)
    return array.astype(np.float32)


def _smooth_tensor(image: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    return F.avg_pool2d(image, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


class TinySemanticINR(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 48,
        base_sharpen_strength: float = 0.0,
        structure_residual_scale: float = 0.18,
        texture_residual_scale: float = 0.16,
        semantic_detail_boost: float = 0.0,
        zero_initialize_heads: bool = True,
    ):
        super().__init__()
        if float(base_sharpen_strength) < 0.0:
            raise ValueError("base_sharpen_strength must be >= 0")
        if float(structure_residual_scale) < 0.0:
            raise ValueError("structure_residual_scale must be >= 0")
        if float(texture_residual_scale) < 0.0:
            raise ValueError("texture_residual_scale must be >= 0")
        if float(semantic_detail_boost) < 0.0:
            raise ValueError("semantic_detail_boost must be >= 0")
        self.base_sharpen_strength = float(base_sharpen_strength)
        self.structure_residual_scale = float(structure_residual_scale)
        self.texture_residual_scale = float(texture_residual_scale)
        self.semantic_detail_boost = float(semantic_detail_boost)
        self.encoder = nn.Sequential(
            nn.Conv2d(5, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.GELU(),
        )
        self.structure_head = nn.Conv2d(hidden_channels, 3, 1)
        self.texture_head = nn.Conv2d(hidden_channels, 3, 1)
        self.alpha_head = nn.Conv2d(hidden_channels, 1, 1)
        self.semantic_head = nn.Conv2d(hidden_channels, 1, 1)
        if zero_initialize_heads:
            self._zero_initialize_heads()

    def _zero_initialize_heads(self) -> None:
        for head in (self.structure_head, self.texture_head, self.alpha_head, self.semantic_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, low_res: torch.Tensor, output_shape: tuple[int, int]) -> SemanticINROutput:
        up = F.interpolate(low_res, size=output_shape, mode="bicubic", align_corners=False).clamp(0.0, 1.0)
        base = (up + self.base_sharpen_strength * (up - _smooth_tensor(up))).clamp(0.0, 1.0)
        coords = _coord_grid(up.shape[0], output_shape[0], output_shape[1], up.device)
        features = self.encoder(torch.cat([base, coords], dim=1))

        structure = (base + self.structure_residual_scale * torch.tanh(self.structure_head(features))).clamp(0.0, 1.0)
        texture = self.texture_residual_scale * torch.tanh(self.texture_head(features))
        alpha = torch.sigmoid(self.alpha_head(features))
        semantic_logits = self.semantic_head(features)
        semantic_detail = self.semantic_detail_boost * torch.sigmoid(semantic_logits) * (base - up)
        restored = (structure + alpha * texture + semantic_detail).clamp(0.0, 1.0)
        return SemanticINROutput(
            restored=restored,
            structure=structure,
            texture=texture,
            alpha=alpha,
            base=base,
            semantic_logits=semantic_logits,
        )


def semantic_inr_loss(
    output: SemanticINROutput,
    target: torch.Tensor,
    mask: torch.Tensor,
    task_loss_weight: float = 0.0,
    semantic_loss_weight: float = 0.03,
    base_consistency_loss_weight: float = 0.0,
    structure_loss_weight: float = 0.20,
    frequency_loss_weight: float = 0.20,
) -> dict[str, torch.Tensor]:
    target_structure = _smooth_tensor(target)
    target_texture = target - target_structure
    pred_texture = output.restored - _smooth_tensor(output.restored)
    reconstruction = F.l1_loss(output.restored, target)
    structure = F.l1_loss(output.structure, target_structure)
    frequency = F.l1_loss(pred_texture, target_texture)
    semantic = F.binary_cross_entropy_with_logits(output.semantic_logits, mask)
    base_consistency = F.l1_loss(output.restored, output.base)
    green_excess = output.restored[:, 1:2] - 0.55 * output.restored[:, 0:1] - 0.45 * output.restored[:, 2:3]
    task_logits = 12.0 * (green_excess - 0.18)
    task = F.binary_cross_entropy_with_logits(task_logits, mask)
    total = (
        reconstruction
        + float(structure_loss_weight) * structure
        + float(frequency_loss_weight) * frequency
        + float(semantic_loss_weight) * semantic
        + float(base_consistency_loss_weight) * base_consistency
        + float(task_loss_weight) * task
    )
    return {
        "total": total,
        "reconstruction": reconstruction,
        "structure": structure,
        "frequency": frequency,
        "semantic": semantic,
        "base_consistency": base_consistency,
        "task": task,
    }


def _learned_task_loss(output: SemanticINROutput, mask: torch.Tensor, learned_task_segmenter: object | None) -> torch.Tensor:
    if learned_task_segmenter is None:
        return output.restored.new_tensor(0.0)

    from .deep_downstream import _forward_logits, _freeze_batchnorm

    model = learned_task_segmenter.model.to(output.restored.device)
    model.eval()
    _freeze_batchnorm(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    input_size = int(learned_task_segmenter.input_size)
    resized = F.interpolate(output.restored, size=(input_size, input_size), mode="bilinear", align_corners=False)
    logits = _forward_logits(model, resized)
    target = F.interpolate(mask, size=logits.shape[-2:], mode="nearest").squeeze(1).long()
    positive = target.float().mean()
    weights = torch.stack(
        [
            1.0 / torch.clamp(1.0 - positive, min=0.05),
            1.0 / torch.clamp(positive, min=0.05),
        ]
    ).to(output.restored.device)
    weights = weights / weights.mean()
    return F.cross_entropy(logits, target, weight=weights)


def train_semantic_inr_steps(
    model: TinySemanticINR,
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    steps: int = 200,
    learning_rate: float = 1e-3,
    task_loss_weight: float = 0.0,
    semantic_loss_weight: float = 0.03,
    base_consistency_loss_weight: float = 0.0,
    structure_loss_weight: float = 0.20,
    frequency_loss_weight: float = 0.20,
    learned_task_loss_weight: float = 0.0,
    learned_task_segmenter: object | None = None,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> list[dict[str, float]]:
    if not batches:
        raise ValueError("at least one training batch is required")
    if float(learned_task_loss_weight) > 0.0 and learned_task_segmenter is None:
        raise ValueError("learned_task_segmenter is required when learned_task_loss_weight > 0")
    device = device or torch.device("cpu")
    torch.manual_seed(seed)
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    for step in range(steps):
        low_res, gt, mask = batches[step % len(batches)]
        low = _image_to_tensor(low_res, device)
        target = _image_to_tensor(gt, device)
        mask_t = _mask_to_tensor(mask, device)
        output = model(low, output_shape=gt.shape[:2])
        losses = semantic_inr_loss(
            output,
            target,
            mask_t,
            task_loss_weight=task_loss_weight,
            semantic_loss_weight=semantic_loss_weight,
            base_consistency_loss_weight=base_consistency_loss_weight,
            structure_loss_weight=structure_loss_weight,
            frequency_loss_weight=frequency_loss_weight,
        )
        learned_task = _learned_task_loss(output, mask_t, learned_task_segmenter)
        losses["learned_task"] = learned_task
        losses["total"] = losses["total"] + float(learned_task_loss_weight) * learned_task
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()
        history.append(
            {
                "total_loss": float(losses["total"].detach().cpu()),
                "reconstruction_loss": float(losses["reconstruction"].detach().cpu()),
                "structure_loss": float(losses["structure"].detach().cpu()),
                "frequency_loss": float(losses["frequency"].detach().cpu()),
                "semantic_loss": float(losses["semantic"].detach().cpu()),
                "base_consistency_loss": float(losses["base_consistency"].detach().cpu()),
                "task_loss": float(losses["task"].detach().cpu()),
                "learned_task_loss": float(losses["learned_task"].detach().cpu()),
            }
        )
    return history


@torch.no_grad()
def restore_with_semantic_inr(
    model: TinySemanticINR,
    low_res: np.ndarray,
    output_shape: tuple[int, int],
    device: Optional[torch.device] = None,
) -> np.ndarray:
    device = device or next(model.parameters()).device
    model.to(device)
    model.eval()
    low = _image_to_tensor(low_res, device)
    output = model(low, output_shape=output_shape)
    return _tensor_to_image(output.restored)


@torch.no_grad()
def predict_semantic_inr_fields(
    model: TinySemanticINR,
    low_res: np.ndarray,
    output_shape: tuple[int, int],
    device: Optional[torch.device] = None,
) -> dict[str, np.ndarray]:
    device = device or next(model.parameters()).device
    model.to(device)
    model.eval()
    low = _image_to_tensor(low_res, device)
    output = model(low, output_shape=output_shape)
    return {
        "restored": _tensor_to_image(output.restored),
        "structure": _tensor_to_image(output.structure),
        "texture": output.texture.detach().cpu().numpy()[0].transpose(1, 2, 0).astype(np.float32),
        "alpha": output.alpha.detach().cpu().numpy()[0, 0].astype(np.float32),
        "semantic_prob": torch.sigmoid(output.semantic_logits).detach().cpu().numpy()[0, 0].astype(np.float32),
    }


def bicubic_numpy(low_res: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    return bicubic_restore(low_res, output_shape)
