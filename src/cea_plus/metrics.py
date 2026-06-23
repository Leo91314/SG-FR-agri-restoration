import math

import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity


def psnr(gt: np.ndarray, pred: np.ndarray) -> float:
    mse = float(np.mean((gt.astype(np.float32) - pred.astype(np.float32)) ** 2))
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim_score(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(structural_similarity(gt, pred, channel_axis=-1, data_range=1.0))


def _validate_edge_quantile(edge_quantile: float) -> float:
    edge_quantile = float(edge_quantile)
    if not np.isfinite(edge_quantile) or edge_quantile <= 0.0 or edge_quantile >= 1.0:
        raise ValueError("edge_quantile must be a finite value between 0 and 1")
    return edge_quantile


def _edge_map(image: np.ndarray, edge_quantile: float = 0.82) -> np.ndarray:
    edge_quantile = _validate_edge_quantile(edge_quantile)
    gray = image.mean(axis=2)
    sx = ndimage.sobel(gray, axis=1)
    sy = ndimage.sobel(gray, axis=0)
    mag = np.hypot(sx, sy)
    threshold = float(np.quantile(mag, edge_quantile))
    return mag > threshold


def boundary_f_score(gt: np.ndarray, pred: np.ndarray, tolerance: int = 1, edge_quantile: float = 0.82) -> float:
    edge_quantile = _validate_edge_quantile(edge_quantile)
    gt_edges = _edge_map(gt, edge_quantile=edge_quantile)
    pred_edges = _edge_map(pred, edge_quantile=edge_quantile)
    gt_near = ndimage.binary_dilation(gt_edges, iterations=tolerance)
    pred_near = ndimage.binary_dilation(pred_edges, iterations=tolerance)

    tp_pred = float(np.logical_and(pred_edges, gt_near).sum())
    tp_gt = float(np.logical_and(gt_edges, pred_near).sum())
    pred_count = float(pred_edges.sum())
    gt_count = float(gt_edges.sum())
    if pred_count == 0.0 or gt_count == 0.0:
        return 0.0
    precision = tp_pred / pred_count
    recall = tp_gt / gt_count
    if precision + recall == 0.0:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def proxy_segmentation_miou(gt_mask: np.ndarray, image: np.ndarray) -> float:
    green_excess = image[..., 1] - 0.55 * image[..., 0] - 0.45 * image[..., 2]
    pred = green_excess > 0.18
    gt = gt_mask.astype(bool)

    intersection_crop = float(np.logical_and(pred, gt).sum())
    union_crop = float(np.logical_or(pred, gt).sum())
    intersection_bg = float(np.logical_and(~pred, ~gt).sum())
    union_bg = float(np.logical_or(~pred, ~gt).sum())
    crop_iou = intersection_crop / union_crop if union_crop else 1.0
    bg_iou = intersection_bg / union_bg if union_bg else 1.0
    return float((crop_iou + bg_iou) / 2.0)
