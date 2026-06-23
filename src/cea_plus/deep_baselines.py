from dataclasses import dataclass
from pathlib import Path
import csv
import importlib.util
import subprocess
import sys
import types
from typing import Optional

import numpy as np
from PIL import Image
import requests
import torch
import torch.nn.functional as F
from torch import nn

from .degradation import DegradationConfig, degrade_sample
from .metrics import boundary_f_score, proxy_segmentation_miou, psnr, ssim_score
from .restoration import bicubic_restore
from .synthesis import AgriSample


@dataclass(frozen=True)
class OfficialBaselineSpec:
    name: str
    official_url: str
    task: str
    local_dir: Path
    weight_hint: str

    @property
    def status(self) -> str:
        if not self.local_dir.exists():
            return "missing"
        weights = list(self.local_dir.rglob("*.pth")) + list(self.local_dir.rglob("*.pt"))
        return "available" if weights else "code_only"


@dataclass(frozen=True)
class SwinIRWeightInfo:
    scale: int
    filename: str
    url: str
    local_path: Path


@dataclass(frozen=True)
class Swin2SRWeightInfo:
    scale: int
    filename: str
    url: str
    local_path: Path


@dataclass(frozen=True)
class HATWeightInfo:
    scale: int
    filename: str
    gdrive_id: str
    local_path: Path


@dataclass(frozen=True)
class OfficialSwinIRBaseline:
    model: nn.Module
    scale: int
    weight_path: Path
    device: torch.device
    tile: Optional[int] = None
    tile_overlap: int = 16

    def restore(self, low_res: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
        tensor = _to_tensor(low_res.astype(np.float32), self.device)
        with torch.no_grad():
            output = _run_swinir_inference(
                tensor,
                model=self.model,
                scale=self.scale,
                tile=self.tile,
                tile_overlap=self.tile_overlap,
                window_size=8,
            )
        restored = output.squeeze(0).float().cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
        if restored.shape[:2] != output_shape:
            restored = _resize_float_image(restored, output_shape)
        return restored.astype(np.float32)


@dataclass(frozen=True)
class OfficialSwin2SRBaseline:
    model: nn.Module
    scale: int
    weight_path: Path
    device: torch.device
    tile: Optional[int] = None
    tile_overlap: int = 16

    def restore(self, low_res: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
        tensor = _to_tensor(low_res.astype(np.float32), self.device)
        with torch.no_grad():
            output = _run_swinir_inference(
                tensor,
                model=self.model,
                scale=self.scale,
                tile=self.tile,
                tile_overlap=self.tile_overlap,
                window_size=8,
            )
        restored = output.squeeze(0).float().cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
        if restored.shape[:2] != output_shape:
            restored = _resize_float_image(restored, output_shape)
        return restored.astype(np.float32)


@dataclass(frozen=True)
class OfficialHATBaseline:
    model: nn.Module
    scale: int
    weight_path: Path
    device: torch.device
    tile: Optional[int] = None
    tile_overlap: int = 16

    def restore(self, low_res: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
        tensor = _to_tensor(low_res.astype(np.float32), self.device)
        with torch.no_grad():
            output = _run_swinir_inference(
                tensor,
                model=self.model,
                scale=self.scale,
                tile=self.tile,
                tile_overlap=self.tile_overlap,
                window_size=16,
            )
        restored = output.squeeze(0).float().cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
        if restored.shape[:2] != output_shape:
            restored = _resize_float_image(restored, output_shape)
        return restored.astype(np.float32)


@dataclass(frozen=True)
class OfficialRestormerBaseline:
    model: nn.Module
    weight_path: Path
    device: torch.device
    tile: Optional[int] = 256
    tile_overlap: int = 32

    def restore(self, low_res: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
        up = bicubic_restore(low_res.astype(np.float32), output_shape)
        tensor = _to_tensor(up, self.device)
        with torch.no_grad():
            output = _run_restormer_inference(
                tensor,
                model=self.model,
                tile=self.tile,
                tile_overlap=self.tile_overlap,
                multiple_of=8,
            )
        restored = output.squeeze(0).float().cpu().clamp(0.0, 1.0).numpy().transpose(1, 2, 0)
        return restored.astype(np.float32)


@dataclass(frozen=True)
class OfficialLIIFBaseline:
    model: nn.Module
    make_coord: object
    batched_predict: object
    weight_path: Path
    device: torch.device
    eval_bsize: int = 30000

    def restore(self, low_res: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
        h, w = output_shape
        image = torch.from_numpy(low_res.transpose(2, 0, 1)).float().to(self.device)
        coord = self.make_coord((h, w)).to(self.device)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2.0 / h
        cell[:, 1] *= 2.0 / w
        original_cuda = torch.Tensor.cuda
        torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor.to(self.device)  # type: ignore[assignment]
        with torch.no_grad():
            try:
                pred = self.batched_predict(
                    self.model,
                    ((image - 0.5) / 0.5).unsqueeze(0),
                    coord.unsqueeze(0),
                    cell.unsqueeze(0),
                    bsize=self.eval_bsize,
                )[0]
            finally:
                torch.Tensor.cuda = original_cuda  # type: ignore[assignment]
        restored = (pred * 0.5 + 0.5).clamp(0.0, 1.0).view(h, w, 3).cpu().numpy()
        return restored.astype(np.float32)


@dataclass(frozen=True)
class OfficialLTEBaseline:
    model: nn.Module
    make_coord: object
    batched_predict: object
    weight_path: Path
    device: torch.device
    eval_bsize: int = 30000

    def restore(self, low_res: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
        h, w = output_shape
        image = torch.from_numpy(low_res.transpose(2, 0, 1)).float().to(self.device)
        coord = self.make_coord((h, w)).to(self.device)
        cell = torch.ones_like(coord)
        cell[:, 0] *= 2.0 / h
        cell[:, 1] *= 2.0 / w
        original_cuda = torch.Tensor.cuda
        torch.Tensor.cuda = lambda tensor, *args, **kwargs: tensor.to(self.device)  # type: ignore[assignment]
        with torch.no_grad():
            try:
                pred = self.batched_predict(
                    self.model,
                    ((image - 0.5) / 0.5).unsqueeze(0),
                    coord.unsqueeze(0),
                    cell.unsqueeze(0),
                    bsize=self.eval_bsize,
                )[0]
            finally:
                torch.Tensor.cuda = original_cuda  # type: ignore[assignment]
        restored = (pred * 0.5 + 0.5).clamp(0.0, 1.0).view(h, w, 3).cpu().numpy()
        return restored.astype(np.float32)


@dataclass(frozen=True)
class TinyResCNNBaseline:
    model: nn.Module
    model_path: Path

    def restore(self, low_res: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
        up = bicubic_restore(low_res, output_shape)
        device = next(self.model.parameters()).device
        with torch.no_grad():
            tensor = torch.from_numpy(up.transpose(2, 0, 1)[None]).float().to(device)
            restored = self.model(tensor).clamp(0.0, 1.0).cpu().numpy()[0].transpose(1, 2, 0)
        return restored.astype(np.float32)


@dataclass(frozen=True)
class DeepBaselineSummary:
    row_count: int
    metrics_path: Path
    status_path: Path
    report_path: Path
    model_path: Path


@dataclass(frozen=True)
class RestormerBaselineSummary:
    row_count: int
    metrics_path: Path
    report_path: Path
    weight_path: Path


@dataclass(frozen=True)
class HATBaselineSummary:
    row_count: int
    metrics_path: Path
    report_path: Path
    weight_paths: dict[int, Path]


class TinyResCNN(nn.Module):
    def __init__(self, width: int = 24) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width, 3, 3, padding=1),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image + 0.35 * torch.tanh(self.body(image))


def swinir_lightweight_weight_info(repo_dir: Path = Path("external_baselines/SwinIR"), scale: int = 4) -> SwinIRWeightInfo:
    if scale not in {2, 4}:
        raise ValueError("official lightweight SwinIR baseline currently supports x2 and x4")
    filename = f"002_lightweightSR_DIV2K_s64w8_SwinIR-S_x{scale}.pth"
    url = f"https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/{filename}"
    return SwinIRWeightInfo(
        scale=scale,
        filename=filename,
        url=url,
        local_path=Path(repo_dir) / "model_zoo" / "swinir" / filename,
    )


def swin2sr_classical_weight_info(repo_dir: Path = Path("external_baselines/swin2sr"), scale: int = 4) -> Swin2SRWeightInfo:
    if scale not in {2, 4}:
        raise ValueError("official classical Swin2SR baseline currently supports x2 and x4")
    filename = f"Swin2SR_ClassicalSR_X{scale}_64.pth"
    url = f"https://github.com/mv-lab/swin2sr/releases/download/v0.0.1/{filename}"
    return Swin2SRWeightInfo(
        scale=scale,
        filename=filename,
        url=url,
        local_path=Path(repo_dir) / "model_zoo" / "swin2sr" / filename,
    )


def hat_imagenet_weight_info(repo_dir: Path = Path("external_baselines/HAT"), scale: int = 4) -> HATWeightInfo:
    gdrive_ids = {
        2: "11WDyK4MMcRapHs_aKJKHaYAFsb29SoCw",
        4: "1cxls85ZE7kalhNy47eBJI_L_Lwf9hxRI",
    }
    if scale not in gdrive_ids:
        raise ValueError("official ImageNet-pretrained HAT baseline currently supports x2 and x4")
    filename = f"HAT_SRx{scale}_ImageNet-pretrain.pth"
    return HATWeightInfo(
        scale=scale,
        filename=filename,
        gdrive_id=gdrive_ids[scale],
        local_path=Path(repo_dir) / "experiments" / "pretrained_models" / filename,
    )


def ensure_swinir_lightweight_weight(repo_dir: Path = Path("external_baselines/SwinIR"), scale: int = 4) -> Path:
    info = swinir_lightweight_weight_info(repo_dir=repo_dir, scale=scale)
    if info.local_path.exists() and info.local_path.stat().st_size > 0:
        return info.local_path
    info.local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(info.url, stream=True, timeout=120)
        response.raise_for_status()
        with info.local_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    except requests.exceptions.RequestException:
        if info.local_path.exists():
            info.local_path.unlink()
        subprocess.run(["curl", "-L", "--fail", info.url, "-o", str(info.local_path)], check=True)
    return info.local_path


def ensure_swin2sr_classical_weight(repo_dir: Path = Path("external_baselines/swin2sr"), scale: int = 4) -> Path:
    info = swin2sr_classical_weight_info(repo_dir=repo_dir, scale=scale)
    if info.local_path.exists() and info.local_path.stat().st_size > 0:
        return info.local_path
    info.local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(info.url, stream=True, timeout=120)
        response.raise_for_status()
        with info.local_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    except requests.exceptions.RequestException:
        if info.local_path.exists():
            info.local_path.unlink()
        subprocess.run(["curl", "-L", "--fail", info.url, "-o", str(info.local_path)], check=True)
    return info.local_path


def ensure_hat_imagenet_weight(repo_dir: Path = Path("external_baselines/HAT"), scale: int = 4) -> Path:
    info = hat_imagenet_weight_info(repo_dir=repo_dir, scale=scale)
    if info.local_path.exists() and info.local_path.stat().st_size > 0:
        return info.local_path
    info.local_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown  # type: ignore

        result = gdown.download(id=info.gdrive_id, output=str(info.local_path), quiet=False)
        if result is None:
            raise RuntimeError("gdown did not return a downloaded path")
    except Exception:
        if info.local_path.exists():
            info.local_path.unlink()
        subprocess.run(
            [sys.executable, "-m", "gdown", "--id", info.gdrive_id, "-O", str(info.local_path)],
            check=True,
        )
    return info.local_path


def _load_swinir_class(repo_dir: Path):
    module_path = Path(repo_dir) / "models" / "network_swinir.py"
    if not module_path.exists():
        raise FileNotFoundError(f"SwinIR network file not found: {module_path}")
    spec = importlib.util.spec_from_file_location("cea_plus_official_swinir_network", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load SwinIR network from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SwinIR


def _load_swin2sr_class(repo_dir: Path):
    module_path = Path(repo_dir) / "models" / "network_swin2sr.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Swin2SR network file not found: {module_path}")
    spec = importlib.util.spec_from_file_location("cea_plus_official_swin2sr_network", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Swin2SR network from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Swin2SR


class _HATArchRegistryShim:
    def register(self, *args, **kwargs):
        def decorator(model_class):
            return model_class

        return decorator


def _install_hat_basicsr_shims() -> dict[str, object]:
    module_names = ("basicsr", "basicsr.utils", "basicsr.utils.registry", "basicsr.archs", "basicsr.archs.arch_util")
    old_modules: dict[str, object] = {name: sys.modules.get(name) for name in module_names}

    basicsr = types.ModuleType("basicsr")
    utils = types.ModuleType("basicsr.utils")
    registry = types.ModuleType("basicsr.utils.registry")
    archs = types.ModuleType("basicsr.archs")
    arch_util = types.ModuleType("basicsr.archs.arch_util")

    registry.ARCH_REGISTRY = _HATArchRegistryShim()

    def to_2tuple(value):
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        return (value, value)

    def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
        return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

    arch_util.to_2tuple = to_2tuple
    arch_util.trunc_normal_ = trunc_normal_
    basicsr.utils = utils
    basicsr.archs = archs
    utils.registry = registry
    archs.arch_util = arch_util

    sys.modules["basicsr"] = basicsr
    sys.modules["basicsr.utils"] = utils
    sys.modules["basicsr.utils.registry"] = registry
    sys.modules["basicsr.archs"] = archs
    sys.modules["basicsr.archs.arch_util"] = arch_util
    return old_modules


def _restore_hat_basicsr_shims(old_modules: dict[str, object]) -> None:
    for name, module in old_modules.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_hat_class(repo_dir: Path):
    module_path = Path(repo_dir) / "hat" / "archs" / "hat_arch.py"
    if not module_path.exists():
        raise FileNotFoundError(f"HAT architecture file not found: {module_path}")
    spec = importlib.util.spec_from_file_location("cea_plus_official_hat_arch", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load HAT architecture from {module_path}")
    module = importlib.util.module_from_spec(spec)
    old_modules = _install_hat_basicsr_shims()
    try:
        spec.loader.exec_module(module)
    finally:
        _restore_hat_basicsr_shims(old_modules)
    return module.HAT


def _build_swinir_lightweight_model(repo_dir: Path, scale: int, device: torch.device) -> nn.Module:
    swinir_class = _load_swinir_class(repo_dir)
    model = swinir_class(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffledirect",
        resi_connection="1conv",
    )
    return model.to(device)


def _build_swin2sr_classical_model(repo_dir: Path, scale: int, device: torch.device) -> nn.Module:
    swin2sr_class = _load_swin2sr_class(repo_dir)
    model = swin2sr_class(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    return model.to(device)


def _build_hat_imagenet_model(repo_dir: Path, scale: int, device: torch.device) -> nn.Module:
    hat_class = _load_hat_class(repo_dir)
    model = hat_class(
        upscale=scale,
        in_chans=3,
        img_size=64,
        window_size=16,
        compress_ratio=3,
        squeeze_factor=30,
        conv_scale=0.01,
        overlap_ratio=0.5,
        img_range=1.0,
        depths=[6, 6, 6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2,
        upsampler="pixelshuffle",
        resi_connection="1conv",
    )
    return model.to(device)


def load_official_swinir_baseline(
    repo_dir: Path = Path("external_baselines/SwinIR"),
    scale: int = 4,
    weight_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    tile: Optional[int] = 64,
    tile_overlap: int = 16,
) -> OfficialSwinIRBaseline:
    device = device or torch.device("cpu")
    weight_path = Path(weight_path) if weight_path is not None else ensure_swinir_lightweight_weight(repo_dir, scale)
    model = _build_swinir_lightweight_model(Path(repo_dir), scale=scale, device=device)
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return OfficialSwinIRBaseline(
        model=model,
        scale=scale,
        weight_path=weight_path,
        device=device,
        tile=tile,
        tile_overlap=tile_overlap,
    )


def load_official_swin2sr_baseline(
    repo_dir: Path = Path("external_baselines/swin2sr"),
    scale: int = 4,
    weight_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    tile: Optional[int] = 64,
    tile_overlap: int = 16,
) -> OfficialSwin2SRBaseline:
    device = device or torch.device("cpu")
    weight_path = Path(weight_path) if weight_path is not None else ensure_swin2sr_classical_weight(repo_dir, scale)
    model = _build_swin2sr_classical_model(Path(repo_dir), scale=scale, device=device)
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return OfficialSwin2SRBaseline(
        model=model,
        scale=scale,
        weight_path=Path(weight_path),
        device=device,
        tile=tile,
        tile_overlap=tile_overlap,
    )


def load_official_hat_baseline(
    repo_dir: Path = Path("external_baselines/HAT"),
    scale: int = 4,
    weight_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    tile: Optional[int] = 64,
    tile_overlap: int = 16,
) -> OfficialHATBaseline:
    device = device or torch.device("cpu")
    weight_path = Path(weight_path) if weight_path is not None else ensure_hat_imagenet_weight(repo_dir, scale)
    model = _build_hat_imagenet_model(Path(repo_dir), scale=scale, device=device)
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict) and "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
    elif isinstance(checkpoint, dict) and "params" in checkpoint:
        state_dict = checkpoint["params"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return OfficialHATBaseline(
        model=model,
        scale=scale,
        weight_path=Path(weight_path),
        device=device,
        tile=tile,
        tile_overlap=tile_overlap,
    )


def _load_restormer_class(repo_dir: Path):
    module_path = Path(repo_dir) / "basicsr" / "models" / "archs" / "restormer_arch.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Restormer architecture file not found: {module_path}")
    spec = importlib.util.spec_from_file_location("cea_plus_official_restormer_arch", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Restormer architecture from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Restormer


def _build_restormer_model(repo_dir: Path, device: torch.device) -> nn.Module:
    restormer_class = _load_restormer_class(repo_dir)
    model = restormer_class(
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=4,
        heads=[1, 2, 4, 8],
        ffn_expansion_factor=2.66,
        bias=False,
        LayerNorm_type="WithBias",
        dual_pixel_task=False,
    )
    return model.to(device)


def load_official_restormer_baseline(
    repo_dir: Path = Path("external_baselines/Restormer"),
    weight_path: Path = Path("external_baselines/Restormer/Deraining/pretrained_models/deraining.pth"),
    device: Optional[torch.device] = None,
    tile: Optional[int] = 256,
    tile_overlap: int = 32,
) -> OfficialRestormerBaseline:
    device = device or torch.device("cpu")
    model = _build_restormer_model(Path(repo_dir), device=device)
    checkpoint = torch.load(Path(weight_path), map_location=device)
    state_dict = checkpoint["params"] if isinstance(checkpoint, dict) and "params" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return OfficialRestormerBaseline(
        model=model,
        weight_path=Path(weight_path),
        device=device,
        tile=tile,
        tile_overlap=tile_overlap,
    )


def load_official_liif_baseline(
    repo_dir: Path = Path("external_baselines/LIIF"),
    weight_path: Path = Path("external_baselines/LIIF/save/edsr-baseline-liif.pth"),
    device: Optional[torch.device] = None,
    eval_bsize: int = 30000,
) -> OfficialLIIFBaseline:
    repo_dir = Path(repo_dir)
    device = device or torch.device("cpu")
    repo_string = str(repo_dir.resolve())
    remove_path = False
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
        remove_path = True
    old_modules = {name: sys.modules.get(name) for name in ("models", "utils", "test")}
    for name in old_modules:
        sys.modules.pop(name, None)
    try:
        import models  # type: ignore
        from test import batched_predict  # type: ignore
        from utils import make_coord  # type: ignore
    finally:
        for name, module in old_modules.items():
            if module is not None:
                sys.modules[name] = module
        if remove_path:
            try:
                sys.path.remove(repo_string)
            except ValueError:
                pass

    checkpoint = torch.load(Path(weight_path), map_location=device)
    model = models.make(checkpoint["model"], load_sd=True).to(device)  # type: ignore[name-defined]
    model.eval()
    return OfficialLIIFBaseline(
        model=model,
        make_coord=make_coord,
        batched_predict=batched_predict,
        weight_path=Path(weight_path),
        device=device,
        eval_bsize=eval_bsize,
    )


def load_official_lte_baseline(
    repo_dir: Path = Path("external_baselines/LTE"),
    weight_path: Path = Path("external_baselines/LTE/save/edsr-baseline-lte.pth"),
    device: Optional[torch.device] = None,
    eval_bsize: int = 30000,
) -> OfficialLTEBaseline:
    repo_dir = Path(repo_dir)
    device = device or torch.device("cpu")
    if device.type == "mps":
        device = torch.device("cpu")
    repo_string = str(repo_dir.resolve())
    remove_path = False
    if repo_string not in sys.path:
        sys.path.insert(0, repo_string)
        remove_path = True
    old_modules = {name: sys.modules.get(name) for name in ("models", "utils", "test", "datasets")}
    for name in old_modules:
        sys.modules.pop(name, None)
    try:
        import models  # type: ignore
        from test import batched_predict  # type: ignore
        from utils import make_coord  # type: ignore
    finally:
        for name, module in old_modules.items():
            if module is not None:
                sys.modules[name] = module
            else:
                sys.modules.pop(name, None)
        if remove_path:
            try:
                sys.path.remove(repo_string)
            except ValueError:
                pass

    checkpoint = torch.load(Path(weight_path), map_location=device)
    model = models.make(checkpoint["model"], load_sd=True).to(device)  # type: ignore[name-defined]
    model.eval()
    return OfficialLTEBaseline(
        model=model,
        make_coord=make_coord,
        batched_predict=batched_predict,
        weight_path=Path(weight_path),
        device=device,
        eval_bsize=eval_bsize,
    )


def _resize_float_image(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0))
    resized = pil.resize((output_shape[1], output_shape[0]), resample=Image.Resampling.BICUBIC)
    return (np.asarray(resized).astype(np.float32) / 255.0).clip(0.0, 1.0)


def _run_swinir_inference(
    image: torch.Tensor,
    model: nn.Module,
    scale: int,
    tile: Optional[int],
    tile_overlap: int,
    window_size: int,
) -> torch.Tensor:
    _, _, h_old, w_old = image.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    image = torch.cat([image, torch.flip(image, [2])], 2)[:, :, : h_old + h_pad, :]
    image = torch.cat([image, torch.flip(image, [3])], 3)[:, :, :, : w_old + w_pad]
    output = _swinir_forward(image, model=model, scale=scale, tile=tile, tile_overlap=tile_overlap, window_size=window_size)
    return output[..., : h_old * scale, : w_old * scale]


def _swinir_forward(
    image: torch.Tensor,
    model: nn.Module,
    scale: int,
    tile: Optional[int],
    tile_overlap: int,
    window_size: int,
) -> torch.Tensor:
    if tile is None:
        return model(image)
    batch, channels, height, width = image.size()
    tile_size = min(tile, height, width)
    if tile_size % window_size != 0:
        raise ValueError("tile must be a multiple of SwinIR window_size")
    tile_overlap = min(tile_overlap, max(0, tile_size - window_size))
    stride = tile_size - tile_overlap
    if stride <= 0:
        raise ValueError("tile_overlap must be smaller than tile")
    h_indices = list(range(0, height - tile_size, stride)) + [height - tile_size]
    w_indices = list(range(0, width - tile_size, stride)) + [width - tile_size]
    output_sum = torch.zeros(batch, channels, height * scale, width * scale).type_as(image)
    output_weight = torch.zeros_like(output_sum)
    for h_idx in h_indices:
        for w_idx in w_indices:
            patch = image[..., h_idx : h_idx + tile_size, w_idx : w_idx + tile_size]
            patch_output = model(patch)
            patch_weight = torch.ones_like(patch_output)
            output_sum[..., h_idx * scale : (h_idx + tile_size) * scale, w_idx * scale : (w_idx + tile_size) * scale].add_(patch_output)
            output_weight[..., h_idx * scale : (h_idx + tile_size) * scale, w_idx * scale : (w_idx + tile_size) * scale].add_(patch_weight)
    return output_sum.div_(output_weight)


def _run_restormer_inference(
    image: torch.Tensor,
    model: nn.Module,
    tile: Optional[int],
    tile_overlap: int,
    multiple_of: int = 8,
) -> torch.Tensor:
    _, _, height, width = image.shape
    pad_h = (multiple_of - height % multiple_of) % multiple_of
    pad_w = (multiple_of - width % multiple_of) % multiple_of
    padded = F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")
    output = _restormer_forward(padded, model=model, tile=tile, tile_overlap=tile_overlap, multiple_of=multiple_of)
    return output[..., :height, :width]


def _restormer_forward(
    image: torch.Tensor,
    model: nn.Module,
    tile: Optional[int],
    tile_overlap: int,
    multiple_of: int,
) -> torch.Tensor:
    if tile is None:
        return model(image)
    batch, channels, height, width = image.shape
    tile_size = min(tile, height, width)
    if tile_size % multiple_of != 0:
        raise ValueError("tile must be a multiple of Restormer padding multiple")
    tile_overlap = min(tile_overlap, max(0, tile_size - multiple_of))
    stride = tile_size - tile_overlap
    if stride <= 0:
        raise ValueError("tile_overlap must be smaller than tile")
    h_indices = list(range(0, height - tile_size, stride)) + [height - tile_size]
    w_indices = list(range(0, width - tile_size, stride)) + [width - tile_size]
    output_sum = torch.zeros(batch, channels, height, width).type_as(image)
    output_weight = torch.zeros_like(output_sum)
    for h_idx in h_indices:
        for w_idx in w_indices:
            patch = image[..., h_idx : h_idx + tile_size, w_idx : w_idx + tile_size]
            patch_output = model(patch)
            patch_weight = torch.ones_like(patch_output)
            output_sum[..., h_idx : h_idx + tile_size, w_idx : w_idx + tile_size].add_(patch_output)
            output_weight[..., h_idx : h_idx + tile_size, w_idx : w_idx + tile_size].add_(patch_weight)
    return output_sum.div_(output_weight)


def official_baseline_specs(root: Path = Path("external_baselines")) -> list[OfficialBaselineSpec]:
    root = Path(root)
    return [
        OfficialBaselineSpec(
            name="SwinIR",
            official_url="https://github.com/JingyunLiang/SwinIR",
            task="image super-resolution / restoration",
            local_dir=root / "SwinIR",
            weight_hint="classical or real-world SR weights from official release/model zoo",
        ),
        OfficialBaselineSpec(
            name="Swin2SR",
            official_url="https://github.com/mv-lab/swin2sr",
            task="SwinV2 image super-resolution / restoration",
            local_dir=root / "swin2sr",
            weight_hint="official GitHub release checkpoints, e.g. Swin2SR_ClassicalSR_X2_64.pth and X4",
        ),
        OfficialBaselineSpec(
            name="HAT",
            official_url="https://github.com/XPixelGroup/HAT",
            task="hybrid attention transformer super-resolution",
            local_dir=root / "HAT",
            weight_hint="official Google Drive/Baidu checkpoints, e.g. HAT_SRx4_ImageNet-pretrain.pth",
        ),
        OfficialBaselineSpec(
            name="Restormer",
            official_url="https://github.com/swz30/Restormer",
            task="deraining / deblurring / denoising restoration",
            local_dir=root / "Restormer",
            weight_hint="official release checkpoints, e.g. deraining.pth or gaussian_color_denoising_blind.pth",
        ),
        OfficialBaselineSpec(
            name="LIIF",
            official_url="https://github.com/yinboc/liif",
            task="arbitrary-scale image super-resolution",
            local_dir=root / "LIIF",
            weight_hint="official pretrained LIIF-EDSR/RDN checkpoint",
        ),
        OfficialBaselineSpec(
            name="LTE",
            official_url="https://github.com/jaewon-lee-b/lte",
            task="implicit image super-resolution with local texture estimator",
            local_dir=root / "LTE",
            weight_hint="official pretrained LTE checkpoint",
        ),
    ]


def _to_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(image.transpose(2, 0, 1)[None]).float().to(device)


def train_tiny_rescnn_baseline(
    train_samples: list[AgriSample],
    degradation_plan: list[DegradationConfig],
    model_path: Path,
    seed: int = 0,
    steps: int = 80,
    learning_rate: float = 0.004,
) -> TinyResCNNBaseline:
    if not train_samples:
        raise ValueError("at least one training sample is required")
    if not degradation_plan:
        raise ValueError("at least one degradation config is required")

    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = TinyResCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    pairs = []
    for sample_idx, sample in enumerate(train_samples):
        for config_idx, config in enumerate(degradation_plan):
            degraded = degrade_sample(sample, config=config, seed=seed * 1000 + sample_idx * 100 + config_idx)
            bicubic = bicubic_restore(degraded.low_res, degraded.gt.shape[:2])
            pairs.append((_to_tensor(bicubic, device), _to_tensor(degraded.gt, device)))

    model.train()
    for step in range(max(1, steps)):
        x_train, y_train = pairs[step % len(pairs)]
        optimizer.zero_grad(set_to_none=True)
        pred = model(x_train)
        loss = torch.mean((pred - y_train) ** 2)
        loss.backward()
        optimizer.step()

    model.eval()
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "seed": seed, "steps": steps}, model_path)
    return TinyResCNNBaseline(model=model, model_path=model_path)


def _write_official_status(path: Path, root: Path = Path("external_baselines")) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["baseline", "official_url", "task", "local_dir", "status", "weight_hint"])
        for spec in official_baseline_specs(root=root):
            writer.writerow([spec.name, spec.official_url, spec.task, spec.local_dir, spec.status, spec.weight_hint])


def _write_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sample", "degradation", "scale", "method", "psnr", "ssim", "boundary_f", "task_miou"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_metric_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _append_metric_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = ["sample", "degradation", "scale", "method", "psnr", "ssim", "boundary_f", "task_miou"]
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _completed_hat_pairs(rows: list[dict[str, object]]) -> set[tuple[str, str, str]]:
    methods_by_pair: dict[tuple[str, str, str], set[str]] = {}
    for row in rows:
        key = (str(row["sample"]), str(row["degradation"]), str(row["scale"]))
        methods_by_pair.setdefault(key, set()).add(str(row["method"]))
    return {key for key, methods in methods_by_pair.items() if {"bicubic", "hat_imagenet"}.issubset(methods)}


def _write_report(path: Path, rows: list[dict[str, object]], model_path: Path, status_path: Path) -> None:
    methods = sorted({str(row["method"]) for row in rows})
    lines = [
        "# Deep Baseline Report",
        "",
        "## Baselines",
        "",
        f"- Tiny ResCNN model: `{model_path}`",
        f"- Official baseline status: `{status_path}`",
        f"- Methods: {', '.join(methods)}",
        "",
        "| Method | PSNR | SSIM | Boundary F | Task mIoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        lines.append(
            f"| {method} | {np.mean([float(row['psnr']) for row in method_rows]):.4f} | "
            f"{np.mean([float(row['ssim']) for row in method_rows]):.4f} | "
            f"{np.mean([float(row['boundary_f']) for row in method_rows]):.4f} | "
            f"{np.mean([float(row['task_miou']) for row in method_rows]):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_deep_baseline_experiment(
    output_dir: Path,
    train_samples: list[AgriSample],
    test_samples: list[AgriSample],
    degradation_plan: list[DegradationConfig],
    seed: int = 7,
    steps: int = 80,
    include_swinir: bool = False,
    swinir_repo_dir: Path = Path("external_baselines/SwinIR"),
    swinir_weight_paths: Optional[dict[int, Path]] = None,
    swinir_device: Optional[torch.device] = None,
    swinir_tile: Optional[int] = 64,
) -> DeepBaselineSummary:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "tiny_rescnn.pt"
    baseline = train_tiny_rescnn_baseline(
        train_samples=train_samples,
        degradation_plan=degradation_plan,
        model_path=model_path,
        seed=seed,
        steps=steps,
    )
    swinir_by_scale: dict[int, OfficialSwinIRBaseline] = {}
    if include_swinir:
        swinir_device = swinir_device or torch.device("cpu")
        weight_paths = swinir_weight_paths or {}
        for scale in sorted({config.scale for config in degradation_plan}):
            weight_path = weight_paths.get(scale)
            swinir_by_scale[scale] = load_official_swinir_baseline(
                repo_dir=swinir_repo_dir,
                scale=scale,
                weight_path=weight_path,
                device=swinir_device,
                tile=swinir_tile,
            )

    rows: list[dict[str, object]] = []
    for sample_idx, sample in enumerate(test_samples):
        for config_idx, config in enumerate(degradation_plan):
            degraded = degrade_sample(sample, config=config, seed=seed * 1000 + sample_idx * 100 + config_idx)
            outputs = {
                "bicubic": bicubic_restore(degraded.low_res, degraded.gt.shape[:2]),
                "tiny_rescnn": baseline.restore(degraded.low_res, degraded.gt.shape[:2]),
            }
            if include_swinir:
                outputs["swinir_lightweight"] = swinir_by_scale[config.scale].restore(degraded.low_res, degraded.gt.shape[:2])
            for method, restored in outputs.items():
                rows.append(
                    {
                        "sample": sample.name,
                        "degradation": degraded.degradation_name,
                        "scale": degraded.scale,
                        "method": method,
                        "psnr": psnr(degraded.gt, restored),
                        "ssim": ssim_score(degraded.gt, restored),
                        "boundary_f": boundary_f_score(degraded.gt, restored),
                        "task_miou": proxy_segmentation_miou(degraded.mask, restored),
                    }
                )

    metrics_path = output_dir / "deep_baseline_metrics.csv"
    status_path = output_dir / "official_baseline_status.csv"
    report_path = output_dir / "deep_baseline_report.md"
    _write_metrics(metrics_path, rows)
    _write_official_status(status_path)
    _write_report(report_path, rows, model_path, status_path)
    return DeepBaselineSummary(
        row_count=len(rows),
        metrics_path=metrics_path,
        status_path=status_path,
        report_path=report_path,
        model_path=model_path,
    )


def _write_restormer_report(path: Path, rows: list[dict[str, object]], weight_path: Path) -> None:
    methods = sorted({str(row["method"]) for row in rows})
    lines = [
        "# Restormer Baseline Report",
        "",
        f"- Weight: `{weight_path}`",
        f"- Methods: {', '.join(methods)}",
        "",
        "| Method | PSNR | SSIM | Boundary F | Task mIoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        lines.append(
            f"| {method} | {np.mean([float(row['psnr']) for row in method_rows]):.4f} | "
            f"{np.mean([float(row['ssim']) for row in method_rows]):.4f} | "
            f"{np.mean([float(row['boundary_f']) for row in method_rows]):.4f} | "
            f"{np.mean([float(row['task_miou']) for row in method_rows]):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_hat_report(path: Path, rows: list[dict[str, object]], weight_paths: dict[int, Path]) -> None:
    methods = sorted({str(row["method"]) for row in rows})
    weight_lines = ", ".join(f"x{scale}: `{path}`" for scale, path in sorted(weight_paths.items()))
    lines = [
        "# HAT Baseline Report",
        "",
        f"- Weights: {weight_lines}",
        f"- Methods: {', '.join(methods)}",
        "",
        "| Method | PSNR | SSIM | Boundary F | Task mIoU |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        lines.append(
            f"| {method} | {np.mean([float(row['psnr']) for row in method_rows]):.4f} | "
            f"{np.mean([float(row['ssim']) for row in method_rows]):.4f} | "
            f"{np.mean([float(row['boundary_f']) for row in method_rows]):.4f} | "
            f"{np.mean([float(row['task_miou']) for row in method_rows]):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_restormer_baseline_experiment(
    output_dir: Path,
    test_samples: list[AgriSample],
    degradation_plan: list[DegradationConfig],
    weight_path: Path = Path("external_baselines/Restormer/Deraining/pretrained_models/deraining.pth"),
    repo_dir: Path = Path("external_baselines/Restormer"),
    seed: int = 7,
    device: Optional[torch.device] = None,
    tile: Optional[int] = 256,
    tile_overlap: int = 32,
) -> RestormerBaselineSummary:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = load_official_restormer_baseline(
        repo_dir=repo_dir,
        weight_path=weight_path,
        device=device or torch.device("cpu"),
        tile=tile,
        tile_overlap=tile_overlap,
    )

    rows: list[dict[str, object]] = []
    for sample_idx, sample in enumerate(test_samples):
        for config_idx, config in enumerate(degradation_plan):
            degraded = degrade_sample(sample, config=config, seed=seed * 1000 + sample_idx * 100 + config_idx)
            outputs = {
                "bicubic": bicubic_restore(degraded.low_res, degraded.gt.shape[:2]),
                "restormer_deraining": baseline.restore(degraded.low_res, degraded.gt.shape[:2]),
            }
            for method, restored in outputs.items():
                rows.append(
                    {
                        "sample": sample.name,
                        "degradation": degraded.degradation_name,
                        "scale": degraded.scale,
                        "method": method,
                        "psnr": psnr(degraded.gt, restored),
                        "ssim": ssim_score(degraded.gt, restored),
                        "boundary_f": boundary_f_score(degraded.gt, restored),
                        "task_miou": proxy_segmentation_miou(degraded.mask, restored),
                    }
                )

    metrics_path = output_dir / "restormer_metrics.csv"
    report_path = output_dir / "restormer_report.md"
    _write_metrics(metrics_path, rows)
    _write_restormer_report(report_path, rows, weight_path=Path(weight_path))
    return RestormerBaselineSummary(
        row_count=len(rows),
        metrics_path=metrics_path,
        report_path=report_path,
        weight_path=Path(weight_path),
    )


def run_hat_baseline_experiment(
    output_dir: Path,
    test_samples: list[AgriSample],
    degradation_plan: list[DegradationConfig],
    weight_paths: Optional[dict[int, Path]] = None,
    repo_dir: Path = Path("external_baselines/HAT"),
    seed: int = 7,
    device: Optional[torch.device] = None,
    tile: Optional[int] = 64,
    tile_overlap: int = 16,
    resume: bool = False,
) -> HATBaselineSummary:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "hat_metrics.csv"
    report_path = output_dir / "hat_report.md"
    device = device or torch.device("cpu")
    configured_weight_paths = weight_paths or {}
    baselines_by_scale: dict[int, OfficialHATBaseline] = {}
    resolved_weight_paths: dict[int, Path] = {}
    for scale in sorted({config.scale for config in degradation_plan}):
        weight_path = configured_weight_paths.get(scale)
        baseline = load_official_hat_baseline(
            repo_dir=repo_dir,
            scale=scale,
            weight_path=weight_path,
            device=device,
            tile=tile,
            tile_overlap=tile_overlap,
        )
        baselines_by_scale[scale] = baseline
        resolved_weight_paths[scale] = baseline.weight_path

    rows: list[dict[str, object]] = _read_metric_rows(metrics_path) if resume else []
    completed_pairs = _completed_hat_pairs(rows) if resume else set()
    for sample_idx, sample in enumerate(test_samples):
        for config_idx, config in enumerate(degradation_plan):
            pair_key = (sample.name, config.name, str(config.scale))
            if pair_key in completed_pairs:
                continue
            degraded = degrade_sample(sample, config=config, seed=seed * 1000 + sample_idx * 100 + config_idx)
            outputs = {
                "bicubic": bicubic_restore(degraded.low_res, degraded.gt.shape[:2]),
                "hat_imagenet": baselines_by_scale[config.scale].restore(degraded.low_res, degraded.gt.shape[:2]),
            }
            new_rows: list[dict[str, object]] = []
            for method, restored in outputs.items():
                new_rows.append(
                    {
                        "sample": sample.name,
                        "degradation": degraded.degradation_name,
                        "scale": degraded.scale,
                        "method": method,
                        "psnr": psnr(degraded.gt, restored),
                        "ssim": ssim_score(degraded.gt, restored),
                        "boundary_f": boundary_f_score(degraded.gt, restored),
                        "task_miou": proxy_segmentation_miou(degraded.mask, restored),
                    }
                )
            rows.extend(new_rows)
            if resume:
                _append_metric_rows(metrics_path, new_rows)
                completed_pairs.add(pair_key)
                if device.type == "mps" and hasattr(torch, "mps"):
                    torch.mps.empty_cache()

    if not resume:
        _write_metrics(metrics_path, rows)
    _write_hat_report(report_path, rows, resolved_weight_paths)
    return HATBaselineSummary(
        row_count=len(rows),
        metrics_path=metrics_path,
        report_path=report_path,
        weight_paths=resolved_weight_paths,
    )
