import argparse
from pathlib import Path

import torch

from .dataset import load_agri_dataset, load_weedsgalore_dataset
from .deep_baselines import run_deep_baseline_experiment
from .degradation import build_degradation_plan


def _parse_scales(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_modes(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CEA_plus deep baseline experiments")
    parser.add_argument("--out", type=Path, default=Path("results/deep_baselines"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--weedsgalore-root", type=Path, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--scales", type=str, default="4")
    parser.add_argument("--modes", type=str, default="fog")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--include-swinir", action="store_true")
    parser.add_argument("--swinir-repo", type=Path, default=Path("external_baselines/SwinIR"))
    parser.add_argument("--swinir-weight-x2", type=Path, default=None)
    parser.add_argument("--swinir-weight-x4", type=Path, default=None)
    parser.add_argument("--swinir-device", type=str, default="cpu")
    parser.add_argument("--swinir-tile", type=int, default=64)
    args = parser.parse_args()

    if args.weedsgalore_root is not None:
        train_samples = load_weedsgalore_dataset(args.weedsgalore_root, split=args.train_split, limit=args.train_limit)
        test_samples = load_weedsgalore_dataset(args.weedsgalore_root, split=args.test_split, limit=args.test_limit)
    elif args.image_dir is not None or args.mask_dir is not None:
        if args.image_dir is None or args.mask_dir is None:
            parser.error("--image-dir and --mask-dir must be provided together")
        samples = load_agri_dataset(args.image_dir, args.mask_dir)
        if len(samples) < 2:
            parser.error("local dataset needs at least two samples")
        train_count = args.train_limit or max(1, len(samples) // 2)
        train_samples = samples[:train_count]
        test_samples = samples[train_count:]
        if args.test_limit is not None:
            test_samples = test_samples[: args.test_limit]
    else:
        parser.error("--weedsgalore-root or --image-dir/--mask-dir must be provided")

    plan = build_degradation_plan(scales=_parse_scales(args.scales), modes=_parse_modes(args.modes))
    swinir_weight_paths = {}
    if args.swinir_weight_x2 is not None:
        swinir_weight_paths[2] = args.swinir_weight_x2
    if args.swinir_weight_x4 is not None:
        swinir_weight_paths[4] = args.swinir_weight_x4
    summary = run_deep_baseline_experiment(
        output_dir=args.out,
        train_samples=train_samples,
        test_samples=test_samples,
        degradation_plan=plan,
        seed=args.seed,
        steps=args.steps,
        include_swinir=args.include_swinir,
        swinir_repo_dir=args.swinir_repo,
        swinir_weight_paths=swinir_weight_paths or None,
        swinir_device=torch.device(args.swinir_device),
        swinir_tile=args.swinir_tile,
    )
    print(f"rows={summary.row_count}")
    print(f"metrics={summary.metrics_path}")
    print(f"status={summary.status_path}")
    print(f"model={summary.model_path}")


if __name__ == "__main__":
    main()
