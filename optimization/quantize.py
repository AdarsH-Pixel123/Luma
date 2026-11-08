"""Model quantization utilities for ISP neural network components.

Provides dynamic / static INT8 quantization, FP16 conversion, and
model-size comparison helpers.
"""

from __future__ import annotations

import io
from typing import Dict

import torch
import torch.nn as nn
import torch.quantization as quant


# ---------------------------------------------------------------------------
# Dynamic quantization
# ---------------------------------------------------------------------------

def quantize_dynamic(model: nn.Module) -> nn.Module:
    """Apply dynamic quantization (INT8 weights) to *model*.

    Quantizes ``nn.Linear`` and ``nn.Conv2d`` layers.  The returned model
    runs with INT8 weight arithmetic at inference time; activations remain
    in floating point.

    Args:
        model: A PyTorch model in eval mode.

    Returns:
        The dynamically-quantized model.
    """
    model.eval()
    quantized = quant.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear, nn.Conv2d},
        dtype=torch.qint8,
    )
    return quantized


# ---------------------------------------------------------------------------
# Static quantization
# ---------------------------------------------------------------------------

def quantize_static(
    model: nn.Module,
    calibration_data: torch.Tensor,
) -> nn.Module:
    """Apply post-training static quantization.

    Uses *calibration_data* to profile activation ranges so that both
    weights **and** activations are quantized to INT8.

    .. note::
       The model should expose ``fuse_model()`` for best results.  If the
       method exists it is called automatically.

    Args:
        model: A PyTorch model.  Will be mutated (prepared in-place).
        calibration_data: Representative input tensor ``(N, C, H, W)``.

    Returns:
        The statically-quantized model.
    """
    model.eval()

    # Optional layer fusion (conv+bn+relu, etc.)
    if hasattr(model, "fuse_model"):
        model.fuse_model()  # type: ignore[attr-defined]

    # Attach a default qconfig
    model.qconfig = quant.get_default_qconfig("x86")  # type: ignore[assignment]

    # Insert observers
    quant.prepare(model, inplace=True)

    # Calibration pass
    with torch.no_grad():
        # Support both a single batch and a dataset-sized tensor
        if calibration_data.dim() == 4 and calibration_data.shape[0] > 1:
            batch_size = min(32, calibration_data.shape[0])
            for start in range(0, calibration_data.shape[0], batch_size):
                batch = calibration_data[start : start + batch_size]
                model(batch)
        else:
            model(calibration_data)

    # Convert to quantized model
    quant.convert(model, inplace=True)
    return model


# ---------------------------------------------------------------------------
# FP16 conversion
# ---------------------------------------------------------------------------

def convert_to_fp16(model: nn.Module) -> nn.Module:
    """Convert *model* to FP16 (half precision).

    Args:
        model: A PyTorch model.

    Returns:
        The same model with all parameters and buffers cast to ``float16``.
    """
    return model.half()


# ---------------------------------------------------------------------------
# Size comparison
# ---------------------------------------------------------------------------

def compare_model_sizes(models: Dict[str, nn.Module]) -> Dict[str, float]:
    """Compare model sizes in MB for different precision variants.

    Each model is serialised to an in-memory buffer and the byte count is
    measured—this gives the *on-disk checkpoint* size which correlates well
    with memory footprint.

    Args:
        models: Mapping of descriptive names to model instances, e.g.
            ``{'FP32': model, 'FP16': model_fp16, 'INT8': model_int8}``.

    Returns:
        Mapping of the same names to sizes in megabytes.
    """
    sizes: Dict[str, float] = {}
    for name, model in models.items():
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        size_mb = buf.tell() / (1024.0 * 1024.0)
        sizes[name] = round(size_mb, 4)
    return sizes
