"""
Hybrid Learned Smartphone ISP — Demo & Benchmark
=================================================
Compares three pipelines side-by-side:

1. **Classical ISP** — traditional signal-processing chain.
2. **Pure Neural ISP** — MobileUNet end-to-end (RAW → sRGB).
3. **Hybrid ISP** — Classical pre + Neural core + Classical post.

Usage::

    # Single image
    python demo.py --image path/to/image.jpg

    # Single image with a trained checkpoint
    python demo.py --image path/to/image.jpg --model_path checkpoints/best.pth

    # Batch of images
    python demo.py --image_dir path/to/images/ --output_dir outputs/

    # Choose noise level for synthetic degradation
    python demo.py --image path/to/image.jpg --noise_level high
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Optional

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Module imports
from datasets.degradation import rgb_to_synthetic_bayer, degrade_raw
from datasets.utils import bayer_to_4ch
from classical_isp.pipeline import ClassicalISP
from hybrid_isp.pipeline import HybridISP
from evaluation.metrics import evaluate_all
from evaluation.compare import (
    create_comparison_grid,
    create_metrics_table,
    plot_metric_comparison,
)


# ---------------------------------------------------------------------------
# Pure Neural ISP helper
# ---------------------------------------------------------------------------

def run_neural_isp(
    packed_bayer: np.ndarray,
    model: "torch.nn.Module",
    device: str = "cpu",
) -> np.ndarray:
    """Run a pure neural ISP (direct packed-RAW → sRGB, no classical stages).

    Args:
        packed_bayer: ``(H/2, W/2, 4)`` packed Bayer planes.
        model:        A ``MobileUNet`` (or compatible) instance, already on
                      *device* and in eval mode.
        device:       ``'cpu'`` or ``'cuda'``.

    Returns:
        ``(H, W, 3)`` ``uint8`` sRGB image.
    """
    import torch

    with torch.no_grad():
        tensor = (
            torch.from_numpy(packed_bayer.transpose(2, 0, 1))
            .unsqueeze(0)
            .float()
            .to(device)
        )

        # Pad to multiple of 8 for the encoder
        _, _, h, w = tensor.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            tensor = torch.nn.functional.pad(
                tensor, (0, pad_w, 0, pad_h), mode="reflect"
            )

        output = model(tensor)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, : h * 2, : w * 2]

        result = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)

    return result


# ---------------------------------------------------------------------------
# Single-image demo
# ---------------------------------------------------------------------------

def demo_single_image(
    image_path: str,
    model_path: Optional[str] = None,
    output_dir: str = "outputs",
    noise_level: str = "medium",
) -> Dict[str, dict]:
    """Run a full three-way comparison on a single image.

    Args:
        image_path:  Path to a ground-truth RGB image (JPEG, PNG, …).
        model_path:  Optional path to a trained neural ISP checkpoint.
        output_dir:  Directory for saving outputs.
        noise_level: Synthetic noise level — ``'low'``, ``'medium'``, or
                     ``'high'``.

    Returns:
        Dictionary ``{pipeline_name: metrics_dict}``.
    """
    import torch
    from neural_isp.mobile_unet import MobileUNet

    os.makedirs(output_dir, exist_ok=True)

    # ---- Load ground-truth RGB -----------------------------------------
    gt_rgb = (
        np.array(Image.open(image_path).convert("RGB")).astype(np.float32)
        / 255.0
    )

    # ---- Generate synthetic degraded RAW --------------------------------
    synthetic_bayer = rgb_to_synthetic_bayer(gt_rgb)
    degraded_bayer, deg_params = degrade_raw(
        synthetic_bayer, noise_level=noise_level
    )

    results: Dict[str, np.ndarray] = {}
    timings: Dict[str, float] = {}

    # ---- 1. Classical ISP -----------------------------------------------
    t0 = time.time()
    classical = ClassicalISP()
    classical_output = classical.process(degraded_bayer)
    timings["Classical ISP"] = (time.time() - t0) * 1000.0
    results["Classical ISP"] = classical_output

    # ---- 2. Pure Neural ISP ---------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    neural_model = MobileUNet(in_channels=4, base_filters=32)
    if model_path and os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        neural_model.load_state_dict(state)
    neural_model.to(device).eval()

    packed = bayer_to_4ch(degraded_bayer)  # (H/2, W/2, 4)

    t0 = time.time()
    neural_output = run_neural_isp(packed, neural_model, device)
    timings["Neural ISP"] = (time.time() - t0) * 1000.0
    results["Neural ISP"] = neural_output

    # ---- 3. Hybrid ISP --------------------------------------------------
    t0 = time.time()
    hybrid = HybridISP(model_path=model_path, device=device)
    hybrid_output = hybrid.process(degraded_bayer)
    timings["Hybrid ISP"] = (time.time() - t0) * 1000.0
    results["Hybrid ISP"] = hybrid_output

    # ---- Ground truth as uint8 ------------------------------------------
    gt_uint8 = (gt_rgb * 255.0).astype(np.uint8)

    # ---- Evaluate all pipelines -----------------------------------------
    metrics: Dict[str, dict] = {}
    for name, output in results.items():
        # Crop to common size (Bayer packing may lose 1 pixel)
        h = min(output.shape[0], gt_uint8.shape[0])
        w = min(output.shape[1], gt_uint8.shape[1])
        pred = output[:h, :w].astype(np.float32) / 255.0
        target = gt_uint8[:h, :w].astype(np.float32) / 255.0
        metrics[name] = evaluate_all(pred, target)
        metrics[name]["latency_ms"] = timings[name]

    # ---- Visualise degraded input (Bayer as greyscale) ------------------
    degraded_vis = np.clip(degraded_bayer, 0.0, 1.0)
    degraded_vis = (
        np.stack([degraded_vis] * 3, axis=-1) * 255.0
    ).astype(np.uint8)

    # ---- Build comparison outputs ---------------------------------------
    images: Dict[str, np.ndarray] = {
        "Degraded RAW": degraded_vis,
        "Classical ISP": results["Classical ISP"],
        "Neural ISP": results["Neural ISP"],
        "Hybrid ISP": results["Hybrid ISP"],
        "Ground Truth": gt_uint8,
    }

    basename = os.path.splitext(os.path.basename(image_path))[0]

    create_comparison_grid(
        images,
        save_path=os.path.join(output_dir, f"{basename}_comparison.png"),
    )
    create_metrics_table(
        metrics,
        save_path=os.path.join(output_dir, f"{basename}_metrics"),
    )
    plot_metric_comparison(
        metrics,
        save_path=os.path.join(output_dir, f"{basename}_charts.png"),
    )

    # ---- Pretty-print results -------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"Results for: {image_path}")
    print(f"{'=' * 60}")
    for name, m in metrics.items():
        print(f"\n{name}:")
        for k, v in m.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    return metrics


# ---------------------------------------------------------------------------
# Batch processing helper
# ---------------------------------------------------------------------------

def demo_batch(
    image_dir: str,
    model_path: Optional[str],
    output_dir: str,
    noise_level: str,
) -> None:
    """Process every image in *image_dir* and print aggregate metrics."""
    all_metrics: Dict[str, Dict[str, dict]] = {}
    supported = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(supported):
            continue
        path = os.path.join(image_dir, fname)
        metrics = demo_single_image(path, model_path, output_dir, noise_level)
        all_metrics[fname] = metrics

    if not all_metrics:
        print("No images found in", image_dir)
        return

    # ---- Aggregate averages per pipeline --------------------------------
    pipelines = list(next(iter(all_metrics.values())).keys())
    n = len(all_metrics)

    print(f"\n{'=' * 60}")
    print(f"AVERAGE METRICS ACROSS {n} IMAGES")
    print(f"{'=' * 60}")

    for pipeline in pipelines:
        avg: Dict[str, float] = {}
        for fname_metrics in all_metrics.values():
            for k, v in fname_metrics[pipeline].items():
                if isinstance(v, (int, float)):
                    avg[k] = avg.get(k, 0.0) + float(v)
        print(f"\n{pipeline} (avg over {n} images):")
        for k, v in avg.items():
            print(f"  {k}: {v / n:.4f}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Learned Smartphone ISP — Demo & Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image", type=str, default=None, help="Path to a single test image."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory of test images for batch processing.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a trained neural ISP checkpoint (.pth).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory for results (default: outputs/).",
    )
    parser.add_argument(
        "--noise_level",
        type=str,
        default="medium",
        choices=["low", "medium", "high"],
        help="Synthetic noise level for degradation (default: medium).",
    )
    args = parser.parse_args()

    if args.image:
        demo_single_image(
            args.image, args.model_path, args.output_dir, args.noise_level
        )
    elif args.image_dir:
        demo_batch(
            args.image_dir, args.model_path, args.output_dir, args.noise_level
        )
    else:
        parser.print_help()
        print("\nPlease provide --image or --image_dir.")


if __name__ == "__main__":
    main()
