import argparse
from pathlib import Path

from .dataset import load_agri_dataset, load_loveda_dataset, load_weedsgalore_dataset
from .degradation import build_degradation_plan
from .pipeline import run_dataset_experiment, run_experiment


def _parse_scales(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_modes(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_labels(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CEA_plus smoke restoration experiment")
    parser.add_argument("--out", type=Path, default=Path("results/smoke"))
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--mask-dir", type=Path, default=None)
    parser.add_argument("--weedsgalore-root", type=Path, default=None)
    parser.add_argument("--loveda-root", type=Path, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--domain", type=str, default="Rural")
    parser.add_argument("--target-labels", type=str, default="7")
    parser.add_argument("--crop-size", type=int, default=None)
    parser.add_argument("--crop-strategy", type=str, default="mask_center")
    parser.add_argument("--crop-seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scales", type=str, default="4")
    parser.add_argument("--modes", type=str, default="mixed")
    args = parser.parse_args()

    if args.weedsgalore_root is not None:
        samples = load_weedsgalore_dataset(dataset_root=args.weedsgalore_root, split=args.split, limit=args.limit)
        plan = build_degradation_plan(scales=_parse_scales(args.scales), modes=_parse_modes(args.modes))
        summary = run_dataset_experiment(output_dir=args.out, samples=samples, degradation_plan=plan, seed=args.seed)
    elif args.loveda_root is not None:
        samples = load_loveda_dataset(
            dataset_root=args.loveda_root,
            split=args.split,
            domain=args.domain,
            target_labels=_parse_labels(args.target_labels),
            limit=args.limit,
            crop_size=args.crop_size,
            crop_strategy=args.crop_strategy,
            crop_seed=args.crop_seed,
        )
        plan = build_degradation_plan(scales=_parse_scales(args.scales), modes=_parse_modes(args.modes))
        summary = run_dataset_experiment(output_dir=args.out, samples=samples, degradation_plan=plan, seed=args.seed)
    elif args.image_dir is not None or args.mask_dir is not None:
        if args.image_dir is None or args.mask_dir is None:
            parser.error("--image-dir and --mask-dir must be provided together")
        samples = load_agri_dataset(image_dir=args.image_dir, mask_dir=args.mask_dir, limit=args.limit)
        plan = build_degradation_plan(scales=_parse_scales(args.scales), modes=_parse_modes(args.modes))
        summary = run_dataset_experiment(output_dir=args.out, samples=samples, degradation_plan=plan, seed=args.seed)
    else:
        summary = run_experiment(output_dir=args.out, samples=args.samples, seed=args.seed, image_size=args.image_size)
    print(f"samples={summary.sample_count}")
    print(f"best_method={summary.best_method}")
    print("significant_metrics=" + ",".join(summary.significant_metrics))


if __name__ == "__main__":
    main()
