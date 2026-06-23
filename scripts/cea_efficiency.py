"""Phase 9: efficiency analysis (params / inference time / peak memory).

Reports parameter counts and wall-clock inference time for the Semantic INR vs the
generic learned restorer baseline (Tiny ResCNN), on a fixed 256x256 restoration task.
FLOPs are reported via ptflops/thop when available; otherwise documented as unavailable.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from cea_plus.restoration import bicubic_restore
from cea_plus.semantic_inr import TinySemanticINR
from cea_plus.deep_baselines import TinyResCNN


def count_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def time_callable(fn, warmup: int = 3, iters: int = 20) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    out = Path("results/cea/phase9_efficiency")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    shape = (256, 256)
    low = np.random.RandomState(0).rand(32, 32, 3).astype(np.float32)
    up = bicubic_restore(low, shape)

    inr = TinySemanticINR(hidden_channels=48).to(device).eval()
    rescnn = TinyResCNN(width=24).to(device).eval()

    low_t = torch.from_numpy(low.transpose(2, 0, 1)[None]).float().to(device)
    up_t = torch.from_numpy(up.transpose(2, 0, 1)[None]).float().to(device)

    @torch.no_grad()
    def run_inr():
        _ = inr(low_t, output_shape=shape)
        if device.type == "mps":
            torch.mps.synchronize()

    @torch.no_grad()
    def run_rescnn():
        _ = rescnn(up_t)
        if device.type == "mps":
            torch.mps.synchronize()

    results = {
        "device": str(device),
        "output_shape": list(shape),
        "methods": {
            "semantic_inr": {
                "params": count_params(inr),
                "params_million": round(count_params(inr) / 1e6, 4),
                "inference_s_per_image": round(time_callable(run_inr), 5),
            },
            "tiny_rescnn": {
                "params": count_params(rescnn),
                "params_million": round(count_params(rescnn) / 1e6, 4),
                "inference_s_per_image": round(time_callable(run_rescnn), 5),
            },
        },
    }

    # optional FLOPs
    flops = {}
    try:
        from ptflops import get_model_complexity_info  # type: ignore

        class INRWrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                return self.m(x, output_shape=shape).restored

        macs, _ = get_model_complexity_info(INRWrap(inr).to(device), (3, 32, 32),
                                            as_strings=False, print_per_layer_stat=False, verbose=False)
        flops["semantic_inr_gflops"] = round(2 * macs / 1e9, 4)
        macs2, _ = get_model_complexity_info(rescnn, (3, 256, 256),
                                             as_strings=False, print_per_layer_stat=False, verbose=False)
        flops["tiny_rescnn_gflops"] = round(2 * macs2 / 1e9, 4)
    except Exception as exc:  # noqa: BLE001
        flops["note"] = f"FLOPs unavailable ({type(exc).__name__}); report params + wall-clock only"
    results["flops"] = flops

    (out / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
