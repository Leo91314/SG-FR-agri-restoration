from dataclasses import dataclass
from pathlib import Path
import csv
import importlib.util
import time

from .dataset import load_agri_dataset, load_weedsgalore_dataset
from .deep_baselines import official_baseline_specs
from .degradation import DegradationConfig, degrade_sample
from .restoration import restore_all_methods
from .synthesis import make_synthetic_agri_sample


@dataclass(frozen=True)
class Stage2Summary:
    real_dataset_count: int
    report_path: Path


DATASET_SOURCES = [
    (
        "WeedsGalore",
        "UAV crop/weed segmentation, CC BY 4.0, downloaded package: weedsgalore-dataset.zip",
        "https://doidata.gfz.de/weedsgalore_e_celikkan_2024/",
    ),
    (
        "LoveDA",
        "Remote-sensing semantic segmentation with rural subset; larger archive, useful secondary benchmark",
        "https://zenodo.org/records/5706578",
    ),
    (
        "Agriculture-Vision",
        "Aerial agricultural pattern segmentation; large benchmark, requires storage planning",
        "https://registry.opendata.aws/intelinair_agriculture_vision/",
    ),
]


BASELINES = [
    ("Bicubic", "local", True, "implemented"),
    ("Uniform sharp", "local", True, "implemented"),
    ("Semantic frequency", "local", True, "implemented"),
    ("DeepLabv3+", "torchvision", importlib.util.find_spec("torchvision") is not None, "available" if importlib.util.find_spec("torchvision") else "missing"),
]


def _find_local_dataset(root: Path) -> tuple[int, str]:
    weedsgalore = root / "data" / "external" / "weedsgalore-dataset"
    if weedsgalore.exists():
        try:
            return len(load_weedsgalore_dataset(weedsgalore, split="test")), str(weedsgalore)
        except Exception:
            pass
    image_dir = root / "dataset" / "images"
    mask_dir = root / "dataset" / "masks"
    if image_dir.exists() and mask_dir.exists():
        samples = load_agri_dataset(image_dir=image_dir, mask_dir=mask_dir)
        return len(samples), "dataset/images + dataset/masks"
    return 0, "未发现"


def _write_dataset_sources(path: Path) -> None:
    lines = ["# 数据源清单", ""]
    for name, description, url in DATASET_SOURCES:
        lines.append(f"## {name}")
        lines.append("")
        lines.append(f"- 说明：{description}")
        lines.append(f"- 官方链接：{url}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_baseline_status(path: Path, root: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["baseline", "source", "available", "status"])
        for row in BASELINES:
            writer.writerow(row)
        for spec in official_baseline_specs(root / "external_baselines"):
            writer.writerow([spec.name, spec.official_url, spec.status == "available", spec.status])


def _write_efficiency(path: Path) -> None:
    sample = make_synthetic_agri_sample(seed=901, size=128)
    degraded = degrade_sample(sample, config=DegradationConfig(name="x4_mixed", scale=4), seed=902)
    rows = []
    for method in restore_all_methods(degraded.low_res, degraded.mask, degraded.gt.shape[:2]):
        started = time.perf_counter()
        for _ in range(5):
            restore_all_methods(degraded.low_res, degraded.mask, degraded.gt.shape[:2])[method]
        elapsed = time.perf_counter() - started
        rows.append((method, elapsed * 1000.0 / 5.0))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "avg_ms"])
        writer.writerows((method, f"{avg_ms:.4f}") for method, avg_ms in rows)


def write_stage2_readiness(root: Path, output_dir: Path, run_efficiency: bool = False) -> Stage2Summary:
    root = Path(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_count, dataset_location = _find_local_dataset(root)
    official_specs = official_baseline_specs(root / "external_baselines")
    official_status = "，".join(f"{spec.name}: {spec.status}" for spec in official_specs)
    missing_official = [spec.name for spec in official_specs if spec.status != "available"]
    _write_dataset_sources(output_dir / "dataset_sources.md")
    _write_baseline_status(output_dir / "baseline_status.csv", root=root)
    if run_efficiency:
        _write_efficiency(output_dir / "efficiency.csv")

    report_path = output_dir / "stage2_readiness_report.md"
    lines = [
        "# Stage 2 Readiness Report",
        "",
        f"- 真实数据目录：{dataset_location}",
        f"- 可加载真实样本数：{dataset_count}",
        f"- 官方深度 baseline：{official_status}。",
        "- 已可运行：本地 bicubic、uniform_sharp、semantic_frequency 与消融方法。",
        "",
        "## 下一步",
        "",
        "1. 复查 WeedsGalore test split 真实矩阵实验结果，确定语义先验过强的修正方案。",
        f"2. 下载或接入剩余官方权重：{', '.join(missing_official) if missing_official else '无'}。",
        "3. 训练或接入冻结 DeepLabv3+ 下游分割模型。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return Stage2Summary(real_dataset_count=dataset_count, report_path=report_path)
