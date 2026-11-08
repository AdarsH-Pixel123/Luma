"""
Robustness and Degradation Recovery Benchmark
=============================================
Stress-tests Classical ISP, Pure Neural ISP, Hybrid ISP, and Adaptive Hybrid ISP
under extreme conditions:
  1. Noise level sweep: [clean, low, medium, high, extreme]
  2. Illumination sweep: [1.0x (normal), 0.5x, 0.25x, 0.1x, 0.05x (extreme low-light)]

Demonstrates performance degradation & recovery to satisfy advanced challenge requirements.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.degradation import rgb_to_synthetic_bayer, add_poisson_gaussian_noise, apply_exposure_change
from datasets.utils import bayer_to_4ch
from classical_isp.pipeline import ClassicalISP
from hybrid_isp.pipeline import HybridISP
from hybrid_isp.adaptive import AdaptiveHybridISP
from neural_isp.mobile_unet import MobileUNet
from evaluation.metrics import evaluate_all


def run_neural(packed_bayer, model, device='cpu'):
    with torch.no_grad():
        tensor = torch.from_numpy(packed_bayer.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
        _, _, h, w = tensor.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
        output = model(tensor)
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :h*2, :w*2]
        result = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        return np.clip(result * 255, 0, 255).astype(np.uint8)


def evaluate_pipelines_on_raw(degraded_bayer, gt_rgb, classical_isp, neural_model, hybrid_isp, adaptive_isp, device='cpu'):
    gt_uint8 = (gt_rgb * 255).astype(np.uint8)
    packed = bayer_to_4ch(degraded_bayer)
    
    # 1. Classical ISP
    t0 = time.time()
    out_classical = classical_isp.process(degraded_bayer)
    lat_classical = (time.time() - t0) * 1000

    # 2. Pure Neural ISP
    t0 = time.time()
    out_neural = run_neural(packed, neural_model, device)
    lat_neural = (time.time() - t0) * 1000

    # 3. Hybrid ISP
    t0 = time.time()
    out_hybrid = hybrid_isp.process(degraded_bayer)
    lat_hybrid = (time.time() - t0) * 1000

    # 4. Adaptive Hybrid ISP
    t0 = time.time()
    out_adaptive, info = adaptive_isp.process_with_info(degraded_bayer)
    lat_adaptive = (time.time() - t0) * 1000

    outputs = {
        'Classical ISP': (out_classical, lat_classical),
        'Pure Neural ISP': (out_neural, lat_neural),
        'Hybrid ISP': (out_hybrid, lat_hybrid),
        'Adaptive Hybrid ISP': (out_adaptive, lat_adaptive)
    }

    metrics = {}
    for name, (out, lat) in outputs.items():
        h = min(out.shape[0], gt_uint8.shape[0])
        w = min(out.shape[1], gt_uint8.shape[1])
        pred_f = out[:h, :w].astype(np.float32) / 255.0
        targ_f = gt_uint8[:h, :w].astype(np.float32) / 255.0
        m = evaluate_all(pred_f, targ_f)
        m['latency_ms'] = lat
        metrics[name] = m

    return metrics


def run_benchmark(image_path: str, model_path: str = None, output_dir: str = 'outputs'):
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize models
    classical = ClassicalISP()
    hybrid = HybridISP(model_path=model_path, device=device)
    adaptive = AdaptiveHybridISP(model_path=model_path, device=device)

    neural = MobileUNet()
    if model_path and os.path.exists(model_path):
        ckpt = torch.load(model_path, map_location=device)
        neural.load_state_dict(ckpt.get('model_state_dict', ckpt))
    neural.to(device).eval()

    # Load test image
    gt_rgb = np.array(Image.open(image_path).convert('RGB')).astype(np.float32) / 255.0
    base_bayer = rgb_to_synthetic_bayer(gt_rgb)

    # -------------------------------------------------------------
    # Experiment 1: Noise Level Sweep
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: NOISE LEVEL SWEEP (Shot + Read Noise)")
    print("=" * 70)

    noise_levels = [
        ("Clean", 0.0, 0.0),
        ("Low", 0.01, 0.005),
        ("Medium", 0.03, 0.015),
        ("High", 0.06, 0.03),
        ("Extreme", 0.10, 0.05),
    ]

    noise_results = {p: {'psnr': [], 'ssim': [], 'delta_e': []} for p in 
                     ['Classical ISP', 'Pure Neural ISP', 'Hybrid ISP', 'Adaptive Hybrid ISP']}

    noise_labels = [n[0] for n in noise_levels]

    for label, shot, read in noise_levels:
        if shot == 0.0 and read == 0.0:
            noisy_bayer = base_bayer.copy()
        else:
            noisy_bayer = add_poisson_gaussian_noise(base_bayer, shot_noise=shot, read_noise=read)

        metrics = evaluate_pipelines_on_raw(noisy_bayer, gt_rgb, classical, neural, hybrid, adaptive, device)
        print(f"\nCondition: Noise [{label}]")
        for p, m in metrics.items():
            print(f"  {p:<22} -> PSNR: {m['psnr']:>6.2f} dB | SSIM: {m['ssim']:>6.4f} | Delta E: {m['delta_e']:>6.2f}")
            noise_results[p]['psnr'].append(m['psnr'])
            noise_results[p]['ssim'].append(m['ssim'])
            noise_results[p]['delta_e'].append(m['delta_e'])

    # Plot Noise Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics_keys = [('psnr', 'PSNR (dB) ↑', 'Peak Signal-to-Noise Ratio'),
                    ('ssim', 'SSIM ↑', 'Structural Similarity Index'),
                    ('delta_e', 'Delta E 2000 ↓', 'Color Difference (Lower is Better)')]

    colors = {'Classical ISP': '#e63946', 'Pure Neural ISP': '#457b9d', 
              'Hybrid ISP': '#2a9d8f', 'Adaptive Hybrid ISP': '#f4a261'}
    markers = {'Classical ISP': 'o', 'Pure Neural ISP': 's', 
               'Hybrid ISP': '^', 'Adaptive Hybrid ISP': 'D'}

    for ax, (mkey, ylabel, title) in zip(axes, metrics_keys):
        for p in noise_results:
            ax.plot(noise_labels, noise_results[p][mkey], marker=markers[p], 
                    color=colors[p], linewidth=2.5, label=p)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Sensor Noise Level', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    plt.tight_layout()
    noise_plot_path = os.path.join(output_dir, 'robustness_noise_curve.png')
    plt.savefig(noise_plot_path, dpi=200)
    plt.close()
    print(f"\n[SAVED] Noise robustness plot -> {noise_plot_path}")

    # -------------------------------------------------------------
    # Experiment 2: Low-Light Illumination Attenuation Sweep
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: ILLUMINATION ATTENUATION SWEEP (Low-Light Stress)")
    print("=" * 70)

    lux_levels = [
        ("100% (Normal)", 1.0),
        ("50%", 0.5),
        ("25%", 0.25),
        ("10% (Low-Light)", 0.1),
        ("5% (Extreme Dark)", 0.05),
    ]

    lux_results = {p: {'psnr': [], 'ssim': [], 'delta_e': []} for p in 
                   ['Classical ISP', 'Pure Neural ISP', 'Hybrid ISP', 'Adaptive Hybrid ISP']}

    lux_labels = [l[0] for l in lux_levels]

    for label, factor in lux_levels:
        dim_bayer = np.clip(base_bayer * factor, 0.0, 1.0)
        # Add slight sensor noise proportional to low-light
        noisy_dim_bayer = add_poisson_gaussian_noise(dim_bayer, shot_noise=0.015, read_noise=0.008)

        metrics = evaluate_pipelines_on_raw(noisy_dim_bayer, gt_rgb, classical, neural, hybrid, adaptive, device)
        print(f"\nCondition: Illumination [{label}]")
        for p, m in metrics.items():
            print(f"  {p:<22} -> PSNR: {m['psnr']:>6.2f} dB | SSIM: {m['ssim']:>6.4f} | Delta E: {m['delta_e']:>6.2f}")
            lux_results[p]['psnr'].append(m['psnr'])
            lux_results[p]['ssim'].append(m['ssim'])
            lux_results[p]['delta_e'].append(m['delta_e'])

    # Plot Illumination Curves
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (mkey, ylabel, title) in zip(axes, metrics_keys):
        for p in lux_results:
            ax.plot(lux_labels, lux_results[p][mkey], marker=markers[p], 
                    color=colors[p], linewidth=2.5, label=p)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Scene Illumination Level', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()

    plt.tight_layout()
    lux_plot_path = os.path.join(output_dir, 'robustness_illumination_curve.png')
    plt.savefig(lux_plot_path, dpi=200)
    plt.close()
    print(f"[SAVED] Illumination robustness plot -> {lux_plot_path}")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ISP Robustness Benchmark')
    parser.add_argument('--image', type=str, default='sample_data/color_chart.png', help='Path to test image')
    parser.add_argument('--model_path', type=str, default='checkpoints/mobile_unet_best.pth', help='Path to trained weights')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    run_benchmark(args.image, args.model_path, args.output_dir)

