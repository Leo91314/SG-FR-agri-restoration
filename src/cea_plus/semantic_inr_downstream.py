from dataclasses import dataclass
from pathlib import Path
import csv
import json
from typing import Optional

import numpy as np
import torch

from .deep_downstream import _resize_image, _resize_mask, train_torch_segmenter
from .degradation import DegradationConfig, degrade_sample
from .downstream import evaluate_frozen_segmenter
from .restoration import bicubic_restore, uniform_sharp_restore
from .semantic_guided_restoration import segmenter_guided_frequency_restore
from .semantic_inr import TinySemanticINR, restore_with_semantic_inr, train_semantic_inr_steps
from .statistics import paired_significance


@dataclass(frozen=True)
class SemanticINRFrozenDownstreamSummary:
    output_dir: Path
    metrics_csv: Path
    report: Path
    summary_json: Path
    segmenter_model_path: Path
    inr_model_path: Path
    sample_count: int
    learned_task_segmenter_model_path: Optional[Path] = None
    guide_segmenter_model_path: Optional[Path] = None


def _build_training_batches(
    train_samples: list,
    degradation_plan: list[DegradationConfig],
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    batches: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for sample_idx, sample in enumerate(train_samples):
        for config_idx, config in enumerate(degradation_plan):
            degraded = degrade_sample(sample, config=config, seed=seed * 1000 + sample_idx * 100 + config_idx)
            batches.append((degraded.low_res, degraded.gt, degraded.mask))
    return batches


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = ["sample", "degradation", "scale", "method", "deep_frozen_miou"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _paired_values(rows: list[dict[str, object]], method: str) -> np.ndarray:
    method_rows = sorted(
        [row for row in rows if row["method"] == method],
        key=lambda item: (str(item["sample"]), str(item["degradation"]), str(item["scale"])),
    )
    return np.array([float(row["deep_frozen_miou"]) for row in method_rows], dtype=np.float64)


_GUIDE_PARAM_KEYS = {
    "base_strength",
    "inner_boost",
    "outer_boost",
    "mask_blur_sigma",
    "uncertain_strength",
    "confidence_gamma",
    "scale2_fog_fallback_strength",
    "scale2_fog_mean_threshold",
    "scale2_fog_saturation_threshold",
    "probability_calibration",
    "bright_fog_fallback_strength",
    "bright_fog_mean_threshold",
    "bright_fog_saturation_threshold",
    "uniform_mix_weight",
}


def _default_guide_candidate_params(
    base_strength: float,
    inner_boost: float,
    outer_boost: float,
    mask_blur_sigma: float,
    uncertain_strength: Optional[float],
    confidence_gamma: float,
    probability_calibration: str,
) -> tuple[dict[str, object], ...]:
    return (
        {
            "base_strength": base_strength,
            "inner_boost": inner_boost,
            "outer_boost": outer_boost,
            "mask_blur_sigma": mask_blur_sigma,
            "uncertain_strength": uncertain_strength,
            "confidence_gamma": confidence_gamma,
            "scale2_fog_fallback_strength": None,
            "scale2_fog_mean_threshold": 0.60,
            "scale2_fog_saturation_threshold": 0.08,
            "probability_calibration": probability_calibration,
            "bright_fog_fallback_strength": None,
            "bright_fog_mean_threshold": 0.60,
            "bright_fog_saturation_threshold": 0.08,
            "uniform_mix_weight": 0.0,
        },
        {
            "base_strength": 0.72,
            "inner_boost": 0.08,
            "outer_boost": -0.05,
            "mask_blur_sigma": mask_blur_sigma,
            "uncertain_strength": 0.0,
            "confidence_gamma": 1.0,
            "scale2_fog_fallback_strength": None,
            "scale2_fog_mean_threshold": 0.60,
            "scale2_fog_saturation_threshold": 0.08,
            "probability_calibration": probability_calibration,
            "bright_fog_fallback_strength": None,
            "bright_fog_mean_threshold": 0.60,
            "bright_fog_saturation_threshold": 0.08,
            "uniform_mix_weight": 0.0,
        },
        {
            "base_strength": 0.36,
            "inner_boost": 0.20,
            "outer_boost": -0.20,
            "mask_blur_sigma": mask_blur_sigma,
            "uncertain_strength": 0.0,
            "confidence_gamma": 1.0,
            "scale2_fog_fallback_strength": None,
            "scale2_fog_mean_threshold": 0.60,
            "scale2_fog_saturation_threshold": 0.08,
            "probability_calibration": probability_calibration,
            "bright_fog_fallback_strength": None,
            "bright_fog_mean_threshold": 0.60,
            "bright_fog_saturation_threshold": 0.08,
            "uniform_mix_weight": 0.0,
        },
        {
            "base_strength": 0.0,
            "inner_boost": 0.45,
            "outer_boost": 0.0,
            "mask_blur_sigma": mask_blur_sigma,
            "uncertain_strength": 0.0,
            "confidence_gamma": 1.0,
            "scale2_fog_fallback_strength": 0.0,
            "scale2_fog_mean_threshold": 0.60,
            "scale2_fog_saturation_threshold": 0.08,
            "probability_calibration": probability_calibration,
            "bright_fog_fallback_strength": None,
            "bright_fog_mean_threshold": 0.60,
            "bright_fog_saturation_threshold": 0.08,
            "uniform_mix_weight": 0.0,
        },
        {
            "base_strength": 0.12,
            "inner_boost": 0.48,
            "outer_boost": -0.08,
            "mask_blur_sigma": mask_blur_sigma,
            "uncertain_strength": 0.0,
            "confidence_gamma": 1.0,
            "scale2_fog_fallback_strength": None,
            "scale2_fog_mean_threshold": 0.60,
            "scale2_fog_saturation_threshold": 0.08,
            "probability_calibration": "image_minmax",
            "bright_fog_fallback_strength": None,
            "bright_fog_mean_threshold": 0.60,
            "bright_fog_saturation_threshold": 0.08,
            "uniform_mix_weight": 0.0,
        },
        {
            "base_strength": 0.0,
            "inner_boost": 0.45,
            "outer_boost": 0.0,
            "mask_blur_sigma": mask_blur_sigma,
            "uncertain_strength": 0.0,
            "confidence_gamma": 1.0,
            "scale2_fog_fallback_strength": None,
            "scale2_fog_mean_threshold": 0.60,
            "scale2_fog_saturation_threshold": 0.08,
            "probability_calibration": "image_minmax",
            "bright_fog_fallback_strength": 0.0,
            "bright_fog_mean_threshold": 0.60,
            "bright_fog_saturation_threshold": 0.08,
            "uniform_mix_weight": 0.0,
        },
        {
            "base_strength": 0.36,
            "inner_boost": 0.20,
            "outer_boost": -0.20,
            "mask_blur_sigma": mask_blur_sigma,
            "uncertain_strength": 0.0,
            "confidence_gamma": 1.0,
            "scale2_fog_fallback_strength": None,
            "scale2_fog_mean_threshold": 0.60,
            "scale2_fog_saturation_threshold": 0.08,
            "probability_calibration": "image_minmax",
            "bright_fog_fallback_strength": None,
            "bright_fog_mean_threshold": 0.60,
            "bright_fog_saturation_threshold": 0.08,
            "uniform_mix_weight": 0.50,
        },
    )


def _effective_guide_params(candidate: dict[str, object], defaults: dict[str, object]) -> dict[str, object]:
    unknown = set(candidate) - _GUIDE_PARAM_KEYS
    if unknown:
        raise ValueError(f"unknown guide candidate parameter(s): {sorted(unknown)}")
    params = dict(defaults)
    params.update(candidate)
    return params


def _select_guided_frequency_params(
    validation_samples: list,
    degradation_plan: list[DegradationConfig],
    segmenter: object,
    guide_segmenter: object,
    eval_size: int,
    seed: int,
    candidates: tuple[dict[str, object], ...],
    defaults: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], float, int, list[dict[str, object]]]:
    if not validation_samples:
        raise ValueError("guide_validation_samples must contain at least one sample")
    if not candidates:
        raise ValueError("guide_candidate_params must contain at least one candidate")

    candidate_scores: list[dict[str, object]] = []
    best_candidate: Optional[dict[str, object]] = None
    best_effective: Optional[dict[str, object]] = None
    best_score = -float("inf")
    validation_pairs = 0
    for candidate_idx, candidate in enumerate(candidates):
        effective = _effective_guide_params(dict(candidate), defaults)
        scores: list[float] = []
        for sample_idx, sample in enumerate(validation_samples):
            for config_idx, config in enumerate(degradation_plan):
                degraded = degrade_sample(sample, config=config, seed=seed * 3000 + sample_idx * 100 + config_idx)
                restored = segmenter_guided_frequency_restore(
                    degraded.low_res,
                    degraded.gt.shape[:2],
                    guide_segmenter=guide_segmenter,
                    base_strength=float(effective["base_strength"]),
                    inner_boost=float(effective["inner_boost"]),
                    outer_boost=float(effective["outer_boost"]),
                    mask_blur_sigma=float(effective["mask_blur_sigma"]),
                    uncertain_strength=None
                    if effective["uncertain_strength"] is None
                    else float(effective["uncertain_strength"]),
                    confidence_gamma=float(effective["confidence_gamma"]),
                    scale2_fog_fallback_strength=None
                    if effective["scale2_fog_fallback_strength"] is None
                    else float(effective["scale2_fog_fallback_strength"]),
                    scale2_fog_mean_threshold=float(effective["scale2_fog_mean_threshold"]),
                    scale2_fog_saturation_threshold=float(effective["scale2_fog_saturation_threshold"]),
                    probability_calibration=str(effective["probability_calibration"]),
                    bright_fog_fallback_strength=None
                    if effective["bright_fog_fallback_strength"] is None
                    else float(effective["bright_fog_fallback_strength"]),
                    bright_fog_mean_threshold=float(effective["bright_fog_mean_threshold"]),
                    bright_fog_saturation_threshold=float(effective["bright_fog_saturation_threshold"]),
                    uniform_mix_weight=float(effective["uniform_mix_weight"]),
                )
                resized_image = _resize_image(restored, (eval_size, eval_size))
                resized_mask = _resize_mask(degraded.mask, (eval_size, eval_size))
                scores.append(evaluate_frozen_segmenter(segmenter, resized_image, resized_mask))
        validation_pairs = len(scores)
        mean_score = float(np.mean(scores)) if scores else float("nan")
        candidate_scores.append(
            {
                "candidate_index": candidate_idx,
                "params": dict(candidate),
                "effective_params": effective,
                "mean_deep_frozen_miou": mean_score,
            }
        )
        if mean_score > best_score:
            best_score = mean_score
            best_candidate = dict(candidate)
            best_effective = effective
    if best_candidate is None or best_effective is None:
        raise ValueError("guide parameter selection failed")
    return best_candidate, best_effective, best_score, validation_pairs, candidate_scores


def _write_report(
    path: Path,
    rows: list[dict[str, object]],
    segmenter_architecture: str,
    segmenter_model_path: Path,
    inr_model_path: Path,
    inr_steps: int,
    learned_task_loss_weight: float,
    learned_task_segmenter_architecture: Optional[str],
    learned_task_segmenter_model_path: Optional[Path],
    guide_segmenter_architecture: Optional[str],
    guide_segmenter_model_path: Optional[Path],
    guide_selection_source: Optional[str],
    guide_selection_validation_pairs: int,
    guide_selection_best_score: Optional[float],
    guide_selected_params: Optional[dict[str, object]],
    seed: int,
) -> None:
    methods = sorted({str(row["method"]) for row in rows})
    lines = [
        "# Semantic INR frozen downstream no-leak report",
        "",
        "## Protocol",
        "",
        "- Segmenter training: clean images only.",
        "- 分割器训练图不经过任何恢复方法。",
        "- 分割器训练完成后冻结；各方法只改变评测输入图像。",
        "- Semantic INR 训练使用 reconstruction/structure/frequency/semantic BCE。",
        "- green-excess task loss 固定关闭：`task_loss_weight=0`。",
        f"- learned frozen segmenter task loss weight: `{learned_task_loss_weight}`。",
        "- 恢复推理不接收 GT mask、gt_mask 或 degradation_name。",
        "",
        "## Models",
        "",
        f"- Frozen downstream architecture: `{segmenter_architecture}`",
        f"- Frozen downstream model: `{segmenter_model_path}`",
        f"- Semantic INR model: `{inr_model_path}`",
        f"- Semantic INR steps: `{inr_steps}`",
    ]
    if learned_task_segmenter_architecture is not None and learned_task_segmenter_model_path is not None:
        lines.extend(
            [
                f"- Train-only learned task architecture: `{learned_task_segmenter_architecture}`",
                f"- Train-only learned task model: `{learned_task_segmenter_model_path}`",
            ]
        )
    if guide_segmenter_architecture is not None and guide_segmenter_model_path is not None:
        lines.extend(
            [
                f"- Inference guide segmenter architecture: `{guide_segmenter_architecture}`",
                f"- Inference guide segmenter model: `{guide_segmenter_model_path}`",
            ]
        )
    if guide_selection_source == "clean_validation_only":
        lines.extend(
            [
                "",
                "## Guide parameter selection",
                "",
                "- Source: clean validation only.",
                f"- Validation pairs: `{guide_selection_validation_pairs}`",
                f"- Best validation mIoU: `{guide_selection_best_score:.6f}`"
                if guide_selection_best_score is not None
                else "- Best validation mIoU: `nan`",
                f"- Selected params: `{json.dumps(guide_selected_params, ensure_ascii=False, sort_keys=True)}`",
            ]
        )
    lines.extend(["", "## Method means", "", "| Method | Deep frozen mIoU |", "|---|---:|"])
    for method in methods:
        values = [float(row["deep_frozen_miou"]) for row in rows if row["method"] == method]
        lines.append(f"| `{method}` | {float(np.mean(values)):.4f} |")

    comparisons = [
        ("semantic_inr_no_leak", "bicubic", "Semantic INR vs Bicubic"),
        ("semantic_guided_frequency", "bicubic", "Semantic Guided Frequency vs Bicubic"),
        ("semantic_guided_frequency", "uniform_sharp", "Semantic Guided Frequency vs Uniform Sharp"),
    ]
    for method, baseline_method, title in comparisons:
        if {baseline_method, method}.issubset(methods):
            base = _paired_values(rows, baseline_method)
            values = _paired_values(rows, method)
            lines.extend(["", f"## {title}", ""])
            if len(base) >= 2:
                stats = paired_significance(base, values, seed=seed)
                lines.extend(
                    [
                        "| Metric | Mean Delta | t-test p | Wilcoxon p | Bootstrap 95% CI |",
                        "|---|---:|---:|---:|---:|",
                        f"| deep_frozen_miou | {stats.mean_delta:.6f} | {stats.ttest_p:.6g} | {stats.wilcoxon_p:.6g} | "
                        f"[{stats.bootstrap_ci_low:.6f}, {stats.bootstrap_ci_high:.6f}] |",
                    ]
                )
            else:
                delta = float(np.mean(values - base)) if len(base) else float("nan")
                lines.append(f"- 配对样本少于 2，跳过显著性统计；mean delta = `{delta:.6f}`。")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- 该报告只验证独立冻结学习型下游证据，不使用固定绿色阈值 task mIoU。",
            "- 小样本 smoke 只能说明流程有效；论文主表仍需扩大样本、强 baseline 和跨数据集复测。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_semantic_inr_frozen_downstream(
    output_dir: Path,
    train_samples: list,
    test_samples: list,
    degradation_plan: list[DegradationConfig],
    segmenter_architecture: str = "segformer_b0",
    segmenter_steps: int = 80,
    segmenter_crop_size: int = 128,
    eval_size: int = 256,
    inr_steps: int = 240,
    hidden_channels: int = 48,
    base_sharpen_strength: float = 0.0,
    structure_residual_scale: float = 0.18,
    texture_residual_scale: float = 0.16,
    semantic_detail_boost: float = 0.0,
    base_consistency_loss_weight: float = 0.0,
    learning_rate: float = 1e-3,
    task_loss_weight: float = 0.0,
    semantic_loss_weight: float = 0.03,
    learned_task_loss_weight: float = 0.0,
    learned_task_segmenter_architecture: Optional[str] = None,
    learned_task_segmenter_steps: int = 40,
    learned_task_segmenter_crop_size: Optional[int] = None,
    learned_task_segmenter_eval_size: Optional[int] = None,
    learned_task_segmenter_seed: Optional[int] = None,
    guide_segmenter_architecture: Optional[str] = None,
    guide_segmenter_steps: int = 40,
    guide_segmenter_crop_size: Optional[int] = None,
    guide_segmenter_eval_size: Optional[int] = None,
    guide_segmenter_seed: Optional[int] = None,
    guide_validation_samples: Optional[list] = None,
    guide_candidate_params: Optional[tuple[dict[str, object], ...]] = None,
    guide_inner_boost: float = 0.20,
    guide_outer_boost: float = -0.05,
    guide_mask_blur_sigma: float = 2.0,
    guide_base_strength: float = 0.72,
    guide_uncertain_strength: Optional[float] = None,
    guide_confidence_gamma: float = 1.0,
    guide_scale2_fog_fallback_strength: Optional[float] = None,
    guide_scale2_fog_mean_threshold: float = 0.60,
    guide_scale2_fog_saturation_threshold: float = 0.08,
    guide_probability_calibration: str = "none",
    guide_bright_fog_fallback_strength: Optional[float] = None,
    guide_bright_fog_mean_threshold: float = 0.60,
    guide_bright_fog_saturation_threshold: float = 0.08,
    guide_uniform_mix_weight: float = 0.0,
    seed: int = 7,
    device: Optional[torch.device] = None,
    methods: tuple[str, ...] = ("bicubic", "uniform_sharp", "semantic_inr_no_leak"),
) -> SemanticINRFrozenDownstreamSummary:
    if float(task_loss_weight) != 0.0:
        raise ValueError("credible frozen downstream requires task_loss_weight=0")
    if float(learned_task_loss_weight) < 0.0:
        raise ValueError("learned_task_loss_weight must be >= 0")
    if guide_candidate_params is not None and guide_validation_samples is None:
        raise ValueError("guide_candidate_params requires guide_validation_samples")
    if not train_samples:
        raise ValueError("at least one training sample is required")
    if not test_samples:
        raise ValueError("at least one test sample is required")
    if not degradation_plan:
        raise ValueError("at least one degradation config is required")

    known_methods = {"bicubic", "uniform_sharp", "semantic_inr_no_leak", "semantic_guided_frequency"}
    unknown = set(methods) - known_methods
    if unknown:
        raise ValueError(f"unknown no-leak downstream method(s): {sorted(unknown)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cpu")

    segmenter = train_torch_segmenter(
        train_samples=train_samples,
        architecture=segmenter_architecture,
        steps=segmenter_steps,
        crop_size=segmenter_crop_size,
        input_size=eval_size,
        seed=seed,
        device=device,
    )
    segmenter_model_path = output_dir / f"frozen_clean_{segmenter_architecture}.pt"
    torch.save(
        {
            "architecture": segmenter_architecture,
            "state_dict": segmenter.model.state_dict(),
            "input_size": eval_size,
            "train_source": "clean",
        },
        segmenter_model_path,
    )

    learned_task_segmenter = None
    learned_task_segmenter_model_path: Optional[Path] = None
    if float(learned_task_loss_weight) > 0.0:
        learned_architecture = learned_task_segmenter_architecture or "tiny"
        learned_seed = learned_task_segmenter_seed if learned_task_segmenter_seed is not None else seed + 101
        learned_task_segmenter = train_torch_segmenter(
            train_samples=train_samples,
            architecture=learned_architecture,
            steps=learned_task_segmenter_steps,
            crop_size=learned_task_segmenter_crop_size or segmenter_crop_size,
            input_size=learned_task_segmenter_eval_size or eval_size,
            seed=learned_seed,
            device=device,
        )
        learned_task_segmenter_model_path = output_dir / f"frozen_clean_train_task_{learned_architecture}.pt"
        torch.save(
            {
                "architecture": learned_architecture,
                "state_dict": learned_task_segmenter.model.state_dict(),
                "input_size": learned_task_segmenter.input_size,
                "train_source": "clean_train_only",
                "role": "train_loss_only",
            },
            learned_task_segmenter_model_path,
        )
        learned_task_segmenter_architecture = learned_architecture

    guide_segmenter = None
    guide_segmenter_model_path: Optional[Path] = None
    guide_selection_source = "manual"
    guide_selection_validation_pairs = 0
    guide_selection_best_score: Optional[float] = None
    guide_selected_params: Optional[dict[str, object]] = None
    guide_selected_effective_params: Optional[dict[str, object]] = None
    guide_candidate_scores: list[dict[str, object]] = []
    if "semantic_guided_frequency" in methods:
        guide_architecture = guide_segmenter_architecture or "tiny"
        guide_seed = guide_segmenter_seed if guide_segmenter_seed is not None else seed + 211
        guide_segmenter = train_torch_segmenter(
            train_samples=train_samples,
            architecture=guide_architecture,
            steps=guide_segmenter_steps,
            crop_size=guide_segmenter_crop_size or segmenter_crop_size,
            input_size=guide_segmenter_eval_size or eval_size,
            seed=guide_seed,
            device=device,
        )
        guide_segmenter_model_path = output_dir / f"frozen_clean_guide_{guide_architecture}.pt"
        torch.save(
            {
                "architecture": guide_architecture,
                "state_dict": guide_segmenter.model.state_dict(),
                "input_size": guide_segmenter.input_size,
                "train_source": "clean_train_only",
                "role": "inference_pseudo_mask_guide",
            },
            guide_segmenter_model_path,
        )
        guide_segmenter_architecture = guide_architecture
        if guide_validation_samples is not None:
            defaults = {
                "base_strength": guide_base_strength,
                "inner_boost": guide_inner_boost,
                "outer_boost": guide_outer_boost,
                "mask_blur_sigma": guide_mask_blur_sigma,
                "uncertain_strength": guide_uncertain_strength,
                "confidence_gamma": guide_confidence_gamma,
                "scale2_fog_fallback_strength": guide_scale2_fog_fallback_strength,
                "scale2_fog_mean_threshold": guide_scale2_fog_mean_threshold,
                "scale2_fog_saturation_threshold": guide_scale2_fog_saturation_threshold,
                "probability_calibration": guide_probability_calibration,
                "bright_fog_fallback_strength": guide_bright_fog_fallback_strength,
                "bright_fog_mean_threshold": guide_bright_fog_mean_threshold,
                "bright_fog_saturation_threshold": guide_bright_fog_saturation_threshold,
                "uniform_mix_weight": guide_uniform_mix_weight,
            }
            candidates = guide_candidate_params or _default_guide_candidate_params(
                base_strength=guide_base_strength,
                inner_boost=guide_inner_boost,
                outer_boost=guide_outer_boost,
                mask_blur_sigma=guide_mask_blur_sigma,
                uncertain_strength=guide_uncertain_strength,
                confidence_gamma=guide_confidence_gamma,
                probability_calibration=guide_probability_calibration,
            )
            (
                guide_selected_params,
                guide_selected_effective_params,
                guide_selection_best_score,
                guide_selection_validation_pairs,
                guide_candidate_scores,
            ) = _select_guided_frequency_params(
                validation_samples=guide_validation_samples,
                degradation_plan=degradation_plan,
                segmenter=segmenter,
                guide_segmenter=guide_segmenter,
                eval_size=eval_size,
                seed=seed,
                candidates=candidates,
                defaults=defaults,
            )
            guide_base_strength = float(guide_selected_effective_params["base_strength"])
            guide_inner_boost = float(guide_selected_effective_params["inner_boost"])
            guide_outer_boost = float(guide_selected_effective_params["outer_boost"])
            guide_mask_blur_sigma = float(guide_selected_effective_params["mask_blur_sigma"])
            guide_uncertain_strength = (
                None
                if guide_selected_effective_params["uncertain_strength"] is None
                else float(guide_selected_effective_params["uncertain_strength"])
            )
            guide_confidence_gamma = float(guide_selected_effective_params["confidence_gamma"])
            guide_scale2_fog_fallback_strength = (
                None
                if guide_selected_effective_params["scale2_fog_fallback_strength"] is None
                else float(guide_selected_effective_params["scale2_fog_fallback_strength"])
            )
            guide_scale2_fog_mean_threshold = float(guide_selected_effective_params["scale2_fog_mean_threshold"])
            guide_scale2_fog_saturation_threshold = float(
                guide_selected_effective_params["scale2_fog_saturation_threshold"]
            )
            guide_probability_calibration = str(guide_selected_effective_params["probability_calibration"])
            guide_bright_fog_fallback_strength = (
                None
                if guide_selected_effective_params["bright_fog_fallback_strength"] is None
                else float(guide_selected_effective_params["bright_fog_fallback_strength"])
            )
            guide_bright_fog_mean_threshold = float(guide_selected_effective_params["bright_fog_mean_threshold"])
            guide_bright_fog_saturation_threshold = float(
                guide_selected_effective_params["bright_fog_saturation_threshold"]
            )
            guide_uniform_mix_weight = float(guide_selected_effective_params["uniform_mix_weight"])
            guide_selection_source = "clean_validation_only"

    inr_model = TinySemanticINR(
        hidden_channels=hidden_channels,
        base_sharpen_strength=base_sharpen_strength,
        structure_residual_scale=structure_residual_scale,
        texture_residual_scale=texture_residual_scale,
        semantic_detail_boost=semantic_detail_boost,
    )
    batches = _build_training_batches(train_samples, degradation_plan, seed=seed)
    history = train_semantic_inr_steps(
        inr_model,
        batches=batches,
        steps=inr_steps,
        learning_rate=learning_rate,
        task_loss_weight=0.0,
        semantic_loss_weight=semantic_loss_weight,
        base_consistency_loss_weight=base_consistency_loss_weight,
        learned_task_loss_weight=learned_task_loss_weight,
        learned_task_segmenter=learned_task_segmenter,
        seed=seed,
        device=device,
    )

    rows: list[dict[str, object]] = []
    for sample_idx, sample in enumerate(test_samples):
        for config_idx, config in enumerate(degradation_plan):
            degraded = degrade_sample(sample, config=config, seed=seed * 2000 + sample_idx * 100 + config_idx)
            restorations = {
                "bicubic": bicubic_restore(degraded.low_res, degraded.gt.shape[:2]),
                "uniform_sharp": uniform_sharp_restore(degraded.low_res, degraded.gt.shape[:2]),
                "semantic_inr_no_leak": restore_with_semantic_inr(
                    inr_model,
                    degraded.low_res,
                    degraded.gt.shape[:2],
                    device=device,
                ),
            }
            if "semantic_guided_frequency" in methods:
                restorations["semantic_guided_frequency"] = segmenter_guided_frequency_restore(
                    degraded.low_res,
                    degraded.gt.shape[:2],
                    guide_segmenter=guide_segmenter,
                    inner_boost=guide_inner_boost,
                    outer_boost=guide_outer_boost,
                    mask_blur_sigma=guide_mask_blur_sigma,
                    base_strength=guide_base_strength,
                    uncertain_strength=guide_uncertain_strength,
                    confidence_gamma=guide_confidence_gamma,
                    scale2_fog_fallback_strength=guide_scale2_fog_fallback_strength,
                    scale2_fog_mean_threshold=guide_scale2_fog_mean_threshold,
                    scale2_fog_saturation_threshold=guide_scale2_fog_saturation_threshold,
                    probability_calibration=guide_probability_calibration,
                    bright_fog_fallback_strength=guide_bright_fog_fallback_strength,
                    bright_fog_mean_threshold=guide_bright_fog_mean_threshold,
                    bright_fog_saturation_threshold=guide_bright_fog_saturation_threshold,
                    uniform_mix_weight=guide_uniform_mix_weight,
                )
            resized_mask = _resize_mask(degraded.mask, (eval_size, eval_size))
            for method in methods:
                resized_image = _resize_image(restorations[method], (eval_size, eval_size))
                rows.append(
                    {
                        "sample": sample.name,
                        "degradation": degraded.degradation_name,
                        "scale": degraded.scale,
                        "method": method,
                        "deep_frozen_miou": evaluate_frozen_segmenter(segmenter, resized_image, resized_mask),
                    }
                )

    metrics_csv = output_dir / "semantic_inr_frozen_downstream_metrics.csv"
    _write_metrics(metrics_csv, rows)

    inr_model_path = output_dir / "semantic_inr_no_leak.pt"
    torch.save(
        {
            "state_dict": inr_model.cpu().state_dict(),
            "hidden_channels": hidden_channels,
            "base_sharpen_strength": base_sharpen_strength,
            "structure_residual_scale": structure_residual_scale,
            "texture_residual_scale": texture_residual_scale,
            "semantic_detail_boost": semantic_detail_boost,
            "base_consistency_loss_weight": base_consistency_loss_weight,
            "task_loss_weight": 0.0,
            "semantic_loss_weight": semantic_loss_weight,
            "learned_task_loss_weight": learned_task_loss_weight,
            "learned_task_segmenter_architecture": learned_task_segmenter_architecture,
            "learned_task_segmenter_train_source": "clean_train_only" if learned_task_segmenter_model_path is not None else None,
            "learned_task_segmenter_role": "train_loss_only" if learned_task_segmenter_model_path is not None else None,
            "guide_segmenter_architecture": guide_segmenter_architecture,
            "guide_segmenter_train_source": "clean_train_only" if guide_segmenter_model_path is not None else None,
            "guide_segmenter_role": "inference_pseudo_mask_guide" if guide_segmenter_model_path is not None else None,
            "guide_inner_boost": guide_inner_boost if guide_segmenter_model_path is not None else None,
            "guide_outer_boost": guide_outer_boost if guide_segmenter_model_path is not None else None,
            "guide_mask_blur_sigma": guide_mask_blur_sigma if guide_segmenter_model_path is not None else None,
            "guide_base_strength": guide_base_strength if guide_segmenter_model_path is not None else None,
            "guide_uncertain_strength": guide_uncertain_strength if guide_segmenter_model_path is not None else None,
            "guide_confidence_gamma": guide_confidence_gamma if guide_segmenter_model_path is not None else None,
            "guide_scale2_fog_fallback_strength": guide_scale2_fog_fallback_strength
            if guide_segmenter_model_path is not None
            else None,
            "guide_scale2_fog_mean_threshold": guide_scale2_fog_mean_threshold
            if guide_segmenter_model_path is not None
            else None,
            "guide_scale2_fog_saturation_threshold": guide_scale2_fog_saturation_threshold
            if guide_segmenter_model_path is not None
            else None,
            "guide_probability_calibration": guide_probability_calibration if guide_segmenter_model_path is not None else None,
            "guide_bright_fog_fallback_strength": guide_bright_fog_fallback_strength
            if guide_segmenter_model_path is not None
            else None,
            "guide_bright_fog_mean_threshold": guide_bright_fog_mean_threshold
            if guide_segmenter_model_path is not None
            else None,
            "guide_bright_fog_saturation_threshold": guide_bright_fog_saturation_threshold
            if guide_segmenter_model_path is not None
            else None,
            "guide_uniform_mix_weight": guide_uniform_mix_weight if guide_segmenter_model_path is not None else None,
            "guide_selection_source": guide_selection_source if guide_segmenter_model_path is not None else None,
            "guide_selection_validation_pairs": guide_selection_validation_pairs if guide_segmenter_model_path is not None else None,
            "guide_selection_best_score": guide_selection_best_score,
            "guide_selected_params": guide_selected_params,
            "guide_selected_effective_params": guide_selected_effective_params,
            "guide_candidate_scores": guide_candidate_scores,
            "train_batches": len(batches),
        },
        inr_model_path,
    )

    report = output_dir / "semantic_inr_frozen_downstream_report.md"
    _write_report(
        report,
        rows,
        segmenter_architecture=segmenter_architecture,
        segmenter_model_path=segmenter_model_path,
        inr_model_path=inr_model_path,
        inr_steps=inr_steps,
        learned_task_loss_weight=learned_task_loss_weight,
        learned_task_segmenter_architecture=learned_task_segmenter_architecture,
        learned_task_segmenter_model_path=learned_task_segmenter_model_path,
        guide_segmenter_architecture=guide_segmenter_architecture,
        guide_segmenter_model_path=guide_segmenter_model_path,
        guide_selection_source=guide_selection_source if guide_segmenter_model_path is not None else None,
        guide_selection_validation_pairs=guide_selection_validation_pairs,
        guide_selection_best_score=guide_selection_best_score,
        guide_selected_params=guide_selected_params,
        seed=seed,
    )

    base = _paired_values(rows, "bicubic")
    sem = _paired_values(rows, "semantic_inr_no_leak")
    mean_delta = float(np.mean(sem - base)) if len(base) and len(sem) == len(base) else float("nan")
    guided = _paired_values(rows, "semantic_guided_frequency")
    guided_mean_delta = float(np.mean(guided - base)) if len(base) and len(guided) == len(base) else float("nan")
    guided_stats = paired_significance(base, guided, seed=seed) if len(base) >= 2 and len(guided) == len(base) else None
    uniform = _paired_values(rows, "uniform_sharp")
    guided_vs_uniform_delta = (
        float(np.mean(guided - uniform)) if len(uniform) and len(guided) == len(uniform) else float("nan")
    )
    guided_vs_uniform_stats = (
        paired_significance(uniform, guided, seed=seed) if len(uniform) >= 2 and len(guided) == len(uniform) else None
    )
    summary_json = output_dir / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "segmenter_architecture": segmenter_architecture,
                "segmenter_train_source": "clean",
                "segmenter_steps": segmenter_steps,
                "inr_steps": inr_steps,
                "base_sharpen_strength": base_sharpen_strength,
                "structure_residual_scale": structure_residual_scale,
                "texture_residual_scale": texture_residual_scale,
                "semantic_detail_boost": semantic_detail_boost,
                "base_consistency_loss_weight": base_consistency_loss_weight,
                "train_batches": len(batches),
                "test_pairs": len(base),
                "task_loss_weight": 0.0,
                "semantic_loss_weight": semantic_loss_weight,
                "learned_task_loss_weight": learned_task_loss_weight,
                "learned_task_segmenter_architecture": learned_task_segmenter_architecture,
                "learned_task_segmenter_train_source": "clean_train_only" if learned_task_segmenter_model_path is not None else None,
                "learned_task_segmenter_role": "train_loss_only" if learned_task_segmenter_model_path is not None else None,
                "guide_segmenter_architecture": guide_segmenter_architecture,
                "guide_segmenter_train_source": "clean_train_only" if guide_segmenter_model_path is not None else None,
                "guide_segmenter_role": "inference_pseudo_mask_guide" if guide_segmenter_model_path is not None else None,
                "guide_inner_boost": guide_inner_boost if guide_segmenter_model_path is not None else None,
                "guide_outer_boost": guide_outer_boost if guide_segmenter_model_path is not None else None,
                "guide_mask_blur_sigma": guide_mask_blur_sigma if guide_segmenter_model_path is not None else None,
                "guide_base_strength": guide_base_strength if guide_segmenter_model_path is not None else None,
                "guide_uncertain_strength": guide_uncertain_strength if guide_segmenter_model_path is not None else None,
                "guide_confidence_gamma": guide_confidence_gamma if guide_segmenter_model_path is not None else None,
                "guide_scale2_fog_fallback_strength": guide_scale2_fog_fallback_strength
                if guide_segmenter_model_path is not None
                else None,
                "guide_scale2_fog_mean_threshold": guide_scale2_fog_mean_threshold
                if guide_segmenter_model_path is not None
                else None,
                "guide_scale2_fog_saturation_threshold": guide_scale2_fog_saturation_threshold
                if guide_segmenter_model_path is not None
                else None,
                "guide_probability_calibration": guide_probability_calibration
                if guide_segmenter_model_path is not None
                else None,
                "guide_bright_fog_fallback_strength": guide_bright_fog_fallback_strength
                if guide_segmenter_model_path is not None
                else None,
                "guide_bright_fog_mean_threshold": guide_bright_fog_mean_threshold
                if guide_segmenter_model_path is not None
                else None,
                "guide_bright_fog_saturation_threshold": guide_bright_fog_saturation_threshold
                if guide_segmenter_model_path is not None
                else None,
                "guide_uniform_mix_weight": guide_uniform_mix_weight
                if guide_segmenter_model_path is not None
                else None,
                "guide_selection_source": guide_selection_source if guide_segmenter_model_path is not None else None,
                "guide_selection_validation_pairs": guide_selection_validation_pairs if guide_segmenter_model_path is not None else None,
                "guide_selection_best_score": guide_selection_best_score,
                "guide_selected_params": guide_selected_params,
                "guide_selected_effective_params": guide_selected_effective_params,
                "guide_candidate_scores": guide_candidate_scores,
                "initial_reconstruction_loss": history[0]["reconstruction_loss"],
                "final_reconstruction_loss": history[-1]["reconstruction_loss"],
                "final_base_consistency_loss": history[-1]["base_consistency_loss"],
                "final_learned_task_loss": history[-1]["learned_task_loss"],
                "semantic_inr_vs_bicubic_deep_frozen_miou_delta": mean_delta,
                "semantic_guided_frequency_vs_bicubic_deep_frozen_miou_delta": guided_mean_delta,
                "semantic_guided_frequency_vs_bicubic_deep_frozen_miou_ttest_p": None
                if guided_stats is None
                else guided_stats.ttest_p,
                "semantic_guided_frequency_vs_bicubic_deep_frozen_miou_wilcoxon_p": None
                if guided_stats is None
                else guided_stats.wilcoxon_p,
                "semantic_guided_frequency_vs_bicubic_deep_frozen_miou_bootstrap_ci_low": None
                if guided_stats is None
                else guided_stats.bootstrap_ci_low,
                "semantic_guided_frequency_vs_bicubic_deep_frozen_miou_bootstrap_ci_high": None
                if guided_stats is None
                else guided_stats.bootstrap_ci_high,
                "semantic_guided_frequency_vs_uniform_sharp_deep_frozen_miou_delta": guided_vs_uniform_delta,
                "semantic_guided_frequency_vs_uniform_sharp_deep_frozen_miou_ttest_p": None
                if guided_vs_uniform_stats is None
                else guided_vs_uniform_stats.ttest_p,
                "semantic_guided_frequency_vs_uniform_sharp_deep_frozen_miou_wilcoxon_p": None
                if guided_vs_uniform_stats is None
                else guided_vs_uniform_stats.wilcoxon_p,
                "semantic_guided_frequency_vs_uniform_sharp_deep_frozen_miou_bootstrap_ci_low": None
                if guided_vs_uniform_stats is None
                else guided_vs_uniform_stats.bootstrap_ci_low,
                "semantic_guided_frequency_vs_uniform_sharp_deep_frozen_miou_bootstrap_ci_high": None
                if guided_vs_uniform_stats is None
                else guided_vs_uniform_stats.bootstrap_ci_high,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return SemanticINRFrozenDownstreamSummary(
        output_dir=output_dir,
        metrics_csv=metrics_csv,
        report=report,
        summary_json=summary_json,
        segmenter_model_path=segmenter_model_path,
        inr_model_path=inr_model_path,
        sample_count=len(base),
        learned_task_segmenter_model_path=learned_task_segmenter_model_path,
        guide_segmenter_model_path=guide_segmenter_model_path,
    )
