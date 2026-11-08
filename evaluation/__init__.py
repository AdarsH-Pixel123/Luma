"""Evaluation module for the Hybrid Learned Smartphone ISP pipeline.

Provides image quality metrics (PSNR, SSIM, Delta E 2000, LPIPS)
and visual comparison utilities.
"""

from .metrics import compute_psnr, compute_ssim, compute_delta_e, compute_lpips, evaluate_all
from .compare import create_comparison_grid, create_metrics_table, plot_metric_comparison

__all__ = [
    "compute_psnr",
    "compute_ssim",
    "compute_delta_e",
    "compute_lpips",
    "evaluate_all",
    "create_comparison_grid",
    "create_metrics_table",
    "plot_metric_comparison",
]
