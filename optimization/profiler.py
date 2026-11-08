"""Model profiling utilities for ISP neural network components.

Provides parameter counting, model-size measurement, FLOPs estimation,
latency benchmarking, memory profiling, and multi-model comparison.
"""

from __future__ import annotations

import io
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Parameter counting
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters.

    Returns:
        ``{'total': int, 'trainable': int, 'non_trainable': int}``
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": total - trainable,
    }


# ---------------------------------------------------------------------------
# Model size
# ---------------------------------------------------------------------------

def measure_model_size(model: nn.Module) -> float:
    """Measure model checkpoint size in MB by serialising to an in-memory buffer.

    Args:
        model: A PyTorch model.

    Returns:
        Size in megabytes.
    """
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024.0 * 1024.0)


# ---------------------------------------------------------------------------
# FLOPs / MACs estimation
# ---------------------------------------------------------------------------

def _readable(count: int) -> str:
    """Convert a large integer to a human-readable string (K / M / G / T)."""
    for suffix in ["", "K", "M", "G", "T"]:
        if abs(count) < 1000.0:
            return f"{count:.2f} {suffix}".strip()
        count /= 1000  # type: ignore[assignment]
    return f"{count:.2f} P"


def estimate_flops(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 4, 256, 256),
) -> Dict[str, Any]:
    """Estimate FLOPs and MACs for *model* via forward-hook counting.

    Counted layer types: ``Conv2d``, ``ConvTranspose2d``, ``Linear``,
    ``BatchNorm2d``.

    Args:
        model: A PyTorch model.
        input_shape: Shape of a single input tensor (including batch dim).

    Returns:
        ``{'total_macs': int, 'total_flops': int,
           'macs_readable': str, 'flops_readable': str}``
    """
    model.eval()
    macs_list: List[int] = []
    hooks: List[torch.utils.hooks.RemovableHook] = []

    def _conv_hook(module: nn.Module, inp: Any, out: Any) -> None:
        assert isinstance(module, (nn.Conv2d, nn.ConvTranspose2d))
        batch, out_c, out_h, out_w = out.shape
        kernel_ops = module.kernel_size[0] * module.kernel_size[1] * (module.in_channels // module.groups)
        total = batch * out_c * out_h * out_w * kernel_ops
        if module.bias is not None:
            total += batch * out_c * out_h * out_w
        macs_list.append(total)

    def _linear_hook(module: nn.Module, inp: Any, out: Any) -> None:
        assert isinstance(module, nn.Linear)
        batch = inp[0].shape[0]
        total = batch * module.in_features * module.out_features
        if module.bias is not None:
            total += batch * module.out_features
        macs_list.append(total)

    def _bn_hook(module: nn.Module, inp: Any, out: Any) -> None:
        assert isinstance(module, nn.BatchNorm2d)
        # BN: 2 multiplies + 2 adds per element (mean, var, scale, shift)
        numel = inp[0].numel()
        macs_list.append(numel * 2)

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            hooks.append(m.register_forward_hook(_conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))
        elif isinstance(m, nn.BatchNorm2d):
            hooks.append(m.register_forward_hook(_bn_hook))

    device = next(model.parameters(), torch.tensor(0.0)).device
    dummy = torch.randn(*input_shape, device=device)
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    total_macs = sum(macs_list)
    total_flops = total_macs * 2  # 1 MAC ≈ 2 FLOPs

    return {
        "total_macs": total_macs,
        "total_flops": total_flops,
        "macs_readable": _readable(total_macs),
        "flops_readable": _readable(total_flops),
    }


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------

def measure_latency(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 4, 256, 256),
    num_runs: int = 100,
    warmup: int = 10,
    device: str = "cpu",
) -> Dict[str, float]:
    """Measure inference latency.

    Runs *warmup* warm-up iterations then times *num_runs* iterations.

    Args:
        model: A PyTorch model.
        input_shape: Input tensor shape.
        num_runs: Number of timed iterations.
        warmup: Number of warm-up iterations (untimed).
        device: ``'cpu'`` or ``'cuda'``.

    Returns:
        ``{'mean_ms': float, 'std_ms': float, 'min_ms': float,
           'max_ms': float, 'fps': float}``
    """
    model = model.to(device).eval()
    dummy = torch.randn(*input_shape, device=device)
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    # Warm-up
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
            if use_cuda:
                torch.cuda.synchronize()

    # Timed runs
    timings: List[float] = []
    with torch.no_grad():
        for _ in range(num_runs):
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # ms

    import numpy as np

    arr = np.array(timings)
    mean_ms = float(arr.mean())
    return {
        "mean_ms": mean_ms,
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def measure_memory(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 4, 256, 256),
    device: str = "cpu",
) -> Dict[str, float]:
    """Measure peak memory consumption during inference.

    For CUDA: uses ``torch.cuda.max_memory_allocated``.
    For CPU: uses ``tracemalloc``.

    Args:
        model: A PyTorch model.
        input_shape: Input tensor shape.
        device: ``'cpu'`` or ``'cuda'``.

    Returns:
        ``{'peak_memory_mb': float, 'model_memory_mb': float}``
    """
    model_mb = measure_model_size(model)
    model = model.to(device).eval()

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        dummy = torch.randn(*input_shape, device=device)
        with torch.no_grad():
            model(dummy)

        torch.cuda.synchronize(device)
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_mb = peak_bytes / (1024.0 * 1024.0)
    else:
        # CPU profiling via tracemalloc
        tracemalloc.start()
        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            model(dummy)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_bytes / (1024.0 * 1024.0)

    return {
        "peak_memory_mb": round(peak_mb, 4),
        "model_memory_mb": round(model_mb, 4),
    }


# ---------------------------------------------------------------------------
# Full profile
# ---------------------------------------------------------------------------

def full_profile(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 4, 256, 256),
    device: str = "cpu",
) -> Dict[str, Any]:
    """Run complete profiling: parameters, size, FLOPs, latency, memory.

    Args:
        model: A PyTorch model.
        input_shape: Input tensor shape.
        device: Inference device.

    Returns:
        Combined dictionary of all profiling metrics.
    """
    result: Dict[str, Any] = {}
    result["parameters"] = count_parameters(model)
    result["model_size_mb"] = measure_model_size(model)
    result["flops"] = estimate_flops(model, input_shape)
    result["latency"] = measure_latency(model, input_shape, device=device)
    result["memory"] = measure_memory(model, input_shape, device=device)
    return result


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def profile_comparison(
    models: Dict[str, nn.Module],
    input_shape: Tuple[int, ...] = (1, 4, 256, 256),
    device: str = "cpu",
) -> Dict[str, Dict[str, Any]]:
    """Profile and compare multiple models.

    Args:
        models: Mapping of model names to instances.
        input_shape: Input tensor shape used for all models.
        device: Inference device.

    Returns:
        Mapping of model names to their full profiling results.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for name, model in models.items():
        results[name] = full_profile(model, input_shape, device)
    return results
