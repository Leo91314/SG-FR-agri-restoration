import argparse
from pathlib import Path

from .dataset import load_agri_dataset, load_weedsgalore_dataset
from .degradation import build_degradation_plan
from .downstream_eval import run_downstream_experiment


def _parse_scales(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_modes(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CEA_plus frozen downstream segmentation experiment")
    parser.add_argument("--out", type=Path, default=Path("results/downstream"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--weedsgalore-root", type=Path, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--test-split", type=str, default="test")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--scales", type=str, default="4")
    parser.add_argument("--modes", type=str, default="mixed")
    parser.add_argument("--max-pixels-per-sample", type=int, default=2000)
    args = parser.parse_args()

    if args.weedsgalore_root is not None:
        train_samples = load_weedsgalore_dataset(dataset_root=args.weedsgalore_root, split=args.train_split, limit=args.train_limit)
        test_samples = load_weedsgalore_dataset(dataset_root=args.weedsgalore_root, split=args.test_split, limit=args.test_limit)
    elif args.image_dir is not None or args.mask_dir is not None:
        if args.image_dir is None or args.mask_dir is None:
            parser.error("--image-dir and --mask-dir must be provided together")
        samples = load_agri_dataset(image_dir=args.image_dir, mask_dir=args.mask_dir, limit=None)
        if len(samples) < 2:
            parser.error("local image/mask dataset needs at least two samples for train/test split")
        train_count = args.train_limit or max(1, len(samples) // 2)
        train_samples = samples[:train_count]
        test_samples = samples[train_count:]
        if args.test_limit is not None:
            test_samples = test_samples[: args.test_limit]
    else:
        parser.error("--weedsgalore-root or --image-dir/--mask-dir must be provided")

    plan = build_degradation_plan(scales=_parse_scales(args.scales), modes=_parse_modes(args.modes))
    summary = run_downstream_experiment(
        output_dir=args.out,
        train_samples=train_samples,
        test_samples=test_samples,
        degradation_plan=plan,
        max_pixels_per_sample=args.max_pixels_per_sample,
        seed=args.seed,
    )
    print(f"samples={summary.sample_count}")
    print(f"best_method={summary.best_method}")
    print("significant_metrics=" + ",".join(summary.significant_metrics))
    print(f"model={summary.model_path}")


if __name__ == "__main__":
    main()
