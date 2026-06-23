# SG-FR: Task-Oriented Semantic-Guided Restoration for Agricultural UAV Segmentation

Source code and evaluation pipeline for the paper:

> **Task-Oriented Semantic-Guided Restoration for Agricultural UAV Crop/Weed Segmentation under Structure and Composite Degradation**
> Minghao Li, School of Mathematics and Statistics, Anqing Normal University. Submitted to *IEEE Access*.

This repository studies **when** a lightweight, regime-specific blind restorer with training-only
semantic supervision (SG-FR) helps downstream crop/weed segmentation, instead of optimizing
image-quality metrics (PSNR/SSIM/BRISQUE) in isolation. Restoration is scored by the mIoU of a
**frozen, clean-trained** segmenter (SegFormer-B0, DeepLabV3+) under a **paired held-out evaluation
protocol** (overlap-controlled where identifiable) with image-level bootstrap CIs.

## Key idea

A compact restorer (25.7K parameters, ~2.4 ms per 256x256 tile) decomposes its output into a
low-frequency structure field and a high-frequency texture field with a per-pixel modulation map:

```
I_hat = B(I_d) + alpha ⊙ H(I_d)
```

A crop-semantic head supplies **auxiliary multi-task supervision during training only** (inert at
inference: no masks, no degradation labels at test time; one model per degradation regime). The restorer is never trained against the evaluation mIoU.

## Repository layout

```
src/cea_plus/      # restoration models, degradation, metrics, downstream eval, experiment runners
scripts/           # experiment drivers, baselines, analysis, figure generation
tests/             # unit / smoke tests (pytest)
pyproject.toml     # package metadata; sets PYTHONPATH=src for tests
```

Note: raw datasets, trained weights, and result dumps are **not** tracked here (see `.gitignore`).
Public datasets are obtained from their original repositories (links below).

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install numpy scipy scikit-image pillow matplotlib torch torchvision
```

Python >= 3.9. Experiments were run on Apple MPS; CUDA/CPU also work.

## Run tests

```bash
python3 -m pytest -q
```

## Run the smoke experiment (synthetic data)

```bash
PYTHONPATH=src python3 -m cea_plus.run_experiment --out results/smoke --samples 24 --seed 7 --image-size 128
```

## Run on WeedsGalore (official structure)

```bash
PYTHONPATH=src python3 -m cea_plus.run_experiment \
  --weedsgalore-root data/external/weedsgalore-dataset \
  --split test \
  --out results/weedsgalore_test_matrix \
  --scales 2,4 \
  --modes rain,fog,noise,blur,jpeg,mixed
```

## Datasets

| Dataset | Role | Source |
|---|---|---|
| WeedsGalore | primary UAV maize crop/weed | original repository |
| CWFID | ground carrot/weed replication | original repository |
| CoFly-WeedDB | real UAV cotton replication | Zenodo DOI 10.5281/zenodo.6697343 |
| LoveDA | out-of-domain scope test | original repository |

## Reproducibility

All experiments use fixed random seeds, frozen downstream segmenters, and paired significance testing.
Following peer review we report **image-level / seed-level percentile bootstrap 95% CIs** (the pairing
unit is the image, not the patch, to avoid inflated significance from non-independent patches). Key
revision scripts:

- `scripts/cea_revision_stats.py` — re-aggregated image/seed-level CIs and the stratified
  quality-gain correlation (ΔPSNR/ΔSSIM/ΔBoundaryF/ΔBRISQUE vs ΔmIoU).
- `scripts/cea_engineering_baselines.py` — classical engineering baselines (CLAHE, unsharp, denoise+
  sharpen, dark-channel dehaze) and fine-tuned learned restorers (compact U-Net, fine-tuned SwinIR).
- `scripts/cea_semantic_source.py` — GT vs SegFormer/DeepLab pseudo-mask supervision with cross-evaluator check.
- `scripts/cea_date_disjoint_sanity.py` — WeedsGalore date-disjoint robustness (train May 25/30, test June 6/15).
- `tests/test_split_integrity.py` — train/test split-integrity checks. The CWFID official split lists
  image 28 in both train and test; `cea_plus.dataset.cwfid_split_ids` removes this leak from training.
  WeedsGalore (multi-temporal) and CoFly (single-flight) residual leakage is disclosed in the paper.

Figure- and table-generation scripts (`cea_figures.py`, `cea_report.py`, `paper_tables.py`) also live in
`scripts/`.

## Citation

```bibtex
@article{li2026sgfr,
  title   = {Task-Oriented Semantic-Guided Restoration for Agricultural UAV Vegetation Segmentation under Structure and Composite Degradation},
  author  = {Li, Minghao},
  journal = {IEEE Access},
  year    = {2026}
}
```

## License

MIT License (see [LICENSE](LICENSE)).
