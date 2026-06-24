# Reproducibility guide — SG-FR (IEEE Access)

This document maps each paper table/figure to the script that regenerates it, lists fixed seeds,
checkpoints, and statistical defaults.

## Environment

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-paper.txt
pip install -e .
export PYTHONPATH=src
```

Python **3.12** is recommended (3.9 fails on `object | None` type syntax in some scripts).
Hardware: experiments were run on Apple MPS; CUDA/CPU are supported.

## Random seeds

| Component | Seeds | Notes |
|---|---|---|
| WeedsGalore main | 71, 72, 73 | `scripts/cea_exp.py --seeds 71,72,73` |
| CWFID / CoFly | 71, 72 | 2 seeds in revision stats |
| SG-FR (INR) training | same as row seed | 2000 steps per regime |
| Frozen segmenter | same as row seed | 150 clean steps (default) |
| Bootstrap CIs | seed 73 (patch-level), 17 (image-level) | `cea_plus.statistics.paired_significance` |
| Degradation per tile | `seed * 2000 + sample_idx * 100 + config_idx` | deterministic |

## Pretrained checkpoints (Hugging Face)

Strong baselines in Table `tab:baselines` load public weights via `scripts/cea_strong_baselines.py`:

| Model | HF repo / path |
|---|---|
| SwinIR | `JingyunLiang/SwinIR` (classical SR) |
| HAT | `Xiangtao/HAT` |
| Restormer | `swz30/Restormer` |
| Swin2SR | `mv-lab/swin2sr-classical-sr-x4-64` |
| SegFormer-B0 | `nvidia/segformer-b0-finetuned-ade-512-512` (ImageNet init in code) |

Fine-tuned SwinIR/U-Net baselines are trained in-repo (`cea_engineering_baselines.py`).

## Table → script mapping

| Paper artifact | Script | Output |
|---|---|---|
| Table `tab:mainresults` | `scripts/cea_exp.py` + `scripts/cea_revision_stats.py` | `results/cea/dataset_*/` |
| Table `tab:baselines` | `scripts/cea_strong_baselines.py`, `cea_engineering_baselines.py`, `cea_no_reference.py` | `results/cea/strong_baselines/` |
| Table `tab:correlation` | `scripts/cea_revision_stats.py` | stdout / revision JSON |
| Table `tab:ablation` | `scripts/cea_exp.py` (INR variants) | `results/cea/phase4_ablation/` |
| Table `tab:scope` (LoveDA) | `scripts/cea_exp.py --dataset loveda_rural,loveda_urban` | `results/cea/dataset_loveda_*/` |
| Table `tab:date_disjoint` | `scripts/cea_date_disjoint_sanity.py` | `results/cea/date_disjoint_sanity/summary.json` |
| Table `tab:near_dup` | `scripts/cea_near_duplicate_sanity.py` | `results/cea/near_duplicate_sanity/summary.json` |
| Table `tab:seg_budget` | `scripts/cea_segmenter_budget_ablation.py` | `results/cea/segmenter_budget_ablation/summary.json` |
| Table `tab:pseudo_source` | `scripts/cea_semantic_source.py` | `results/cea/phase6_semantic_source/summary.json` |
| Table `tab:fullmatrix` | `scripts/cea_table2_matrix.py` | `results/cea/full_matrix/` |
| Figs 1–5 | `scripts/cea_figures.py`, `cea_sat_figures.py`, `cea_report.py` | `paper/figures/` |

## Bootstrap protocol

- **Image-level CI** (main claims): per-image mean of patch-level ΔmIoU, then percentile bootstrap
  (3000 resamples, 2.5/97.5 quantiles). Pairing unit = image (not patch).
- **Patch-level CI** (supplementary): `paired_significance(..., bootstraps=2000)`.
- Significance tests: paired t-test and Wilcoxon on patch pairs (reported in CSV summaries).

## BRISQUE

Computed with `piq.brisque` (see `scripts/cea_no_reference.py`). Version pinned in
`requirements-paper.txt`. Lower is better.

## Split integrity

- **CWFID**: image 28 removed from training (`cea_plus.dataset.cwfid_split_ids`).
- **WeedsGalore / CoFly**: official splits retained; content overlap disclosed; sensitivity in
  `cea_near_duplicate_sanity.py` and `cea_date_disjoint_sanity.py`.
- Automated checks: `pytest tests/test_split_integrity.py`.

## Minimal reproduction (WeedsGalore structure+composite)

```bash
PYTHONPATH=src python3.12 scripts/cea_exp.py \
  --tag reproduce_main \
  --dataset weedsgalore \
  --plan structure,composite \
  --methods bicubic,semantic_inr \
  --archs segformer_b0_imagenet \
  --seeds 71,72,73
```
