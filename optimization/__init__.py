"""Optimization module for the Hybrid Learned Smartphone ISP pipeline.

Provides model quantization, profiling, and performance comparison utilities.
"""

from .quantize import quantize_dynamic, quantize_static, convert_to_fp16, compare_model_sizes
from .profiler import (
    count_parameters,
    measure_model_size,
    estimate_flops,
    measure_latency,
    measure_memory,
    full_profile,
    profile_comparison,
)

__all__ = [
    "quantize_dynamic",
    "quantize_static",
    "convert_to_fp16",
    "compare_model_sizes",
    "count_parameters",
    "measure_model_size",
    "estimate_flops",
    "measure_latency",
    "measure_memory",
    "full_profile",
    "profile_comparison",
]
