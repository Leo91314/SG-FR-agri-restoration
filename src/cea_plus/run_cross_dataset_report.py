import argparse
from pathlib import Path

from .cross_dataset import CrossDatasetSpec, write_cross_dataset_report


def _parse_dataset_spec(value: str) -> CrossDatasetSpec:
    parts = value.split("|", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError("--dataset must use NAME|CSV_PATH or NAME|CSV_PATH|DESCRIPTION")
    name, csv_path = parts[0].strip(), parts[1].strip()
    description = parts[2].strip() if len(parts) == 3 else ""
    if not name:
        raise argparse.ArgumentTypeError("dataset name cannot be empty")
    return CrossDatasetSpec(name=name, metrics_csv=Path(csv_path), description=description)


def _parse_degradations(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_modes(value: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Write cross-dataset CEA_plus generalization report")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", action="append", type=_parse_dataset_spec, required=True)
    parser.add_argument("--include-degradations", type=_parse_degradations, default=())
    parser.add_argument("--include-modes", type=_parse_modes, default=())
    args = parser.parse_args()

    summary = write_cross_dataset_report(
        dataset_specs=args.dataset,
        output_dir=args.out,
        include_degradations=args.include_degradations,
        include_modes=args.include_modes,
    )
    print(f"report={summary.report}")
    print(f"method_means={summary.method_means}")
    print(f"paired_deltas={summary.paired_deltas}")
    print(f"summary_json={summary.summary_json}")


if __name__ == "__main__":
    main()
