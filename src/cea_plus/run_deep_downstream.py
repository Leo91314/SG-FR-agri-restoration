import argparse
from pathlib import Path

from .dataset import load_agri_dataset, load_loveda_dataset, load_weedsgalore_dataset
from .degradation import build_degradation_plan
from .deep_downstream import run_deep_downstream_experiment


def _parse_scales(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_modes(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_methods(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_labels(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CEA_plus deep frozen downstream segmentation experiment")
    parser.add_argument("--out", type=Path, default=Path("results/deep_downstream"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--architecture", type=str, default="deeplabv3plus")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--eval-size", type=int, default=160)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--weedsgalore-root", type=Path, default=None)
    parser.add_argument("--loveda-root", type=Path, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--split", type=str, default="Val")
    parser.add_argument("--domain", type=str, default="Rural")
    parser.add_argument("--target-labels", type=str, default="7")
    parser.add_argument("--loveda-crop-size", type=int, default=512)
    parser.add_argument("--loveda-crop-strategy", type=str, default="mask_center")
    parser.add_argument("--loveda-crop-seed", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--scales", type=str, default="4")
    parser.add_argument("--modes", type=str, default="mixed")
    parser.add_argument(
        "--methods",
        type=str,
        default="bicubic,uniform_sharp,semantic_frequency,structure_only,semantic_edge_aware,semantic_boundary_guard",
    )
    args = parser.parse_args()

    if args.weedsgalore_root is not None:
        train_samples = load_weedsgalore_dataset(args.weedsgalore_root, split=args.train_split, limit=args.train_limit)
        test_samples = load_weedsgalore_dataset(args.weedsgalore_root, split=args.test_split, limit=args.test_limit)
    elif args.loveda_root is not None:
        samples = load_loveda_dataset(
            args.loveda_root,
            split=args.split,
            domain=args.domain,
            target_labels=_parse_labels(args.target_labels),
            limit=None,
            crop_size=args.loveda_crop_size,
            crop_strategy=args.loveda_crop_strategy,
            crop_seed=args.loveda_crop_seed,
        )
        train_count = args.train_limit or max(1, len(samples) // 2)
        train_samples = samples[:train_count]
        test_samples = samples[train_count:]
        if args.test_limit is not None:
            test_samples = test_samples[: args.test_limit]
    elif args.image_dir is not None or args.mask_dir is not None:
        if args.image_dir is None or args.mask_dir is None:
            parser.error("--image-dir and --mask-dir must be provided together")
        samples = load_agri_dataset(args.image_dir, args.mask_dir)
        if len(samples) < 2:
            parser.error("local image/mask dataset needs at least two samples for train/test split")
        train_count = args.train_limit or max(1, len(samples) // 2)
        train_samples = samples[:train_count]
        test_samples = samples[train_count:]
        if args.test_limit is not None:
            test_samples = test_samples[: args.test_limit]
    else:
        parser.error("--weedsgalore-root, --loveda-root, or --image-dir/--mask-dir must be provided")

    plan = build_degradation_plan(scales=_parse_scales(args.scales), modes=_parse_modes(args.modes))
    summary = run_deep_downstream_experiment(
        output_dir=args.out,
        train_samples=train_samples,
        test_samples=test_samples,
        degradation_plan=plan,
        architecture=args.architecture,
        steps=args.steps,
        crop_size=args.crop_size,
        eval_size=args.eval_size,
        seed=args.seed,
        methods=_parse_methods(args.methods),
    )
    print(f"samples={summary.sample_count}")
    print(f"metrics={summary.metrics_path}")
    print(f"report={summary.report_path}")
    print(f"model={summary.model_path}")


if __name__ == "__main__":
    main()
