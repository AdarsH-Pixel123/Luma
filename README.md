# LUMA: Lightweight Unified Mobile Architecture for Smartphone Hybrid ISP

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**LUMA** (**L**ightweight **U**nified **M**obile **A**rchitecture) is an end-to-end, mobile-constrained **Hybrid Image Signal Processor (ISP)** that bridges physics-based classical image processing and deep learning restoration to reconstruct degraded Bayer RAW sensor data into high-fidelity sRGB images under extreme computational and memory bandwidth constraints.

---

##  System Architecture

Modern mobile ISPs must balance image quality with strict thermal, memory bandwidth, and latency budgets on NPUs/DSPs. Pure deep learning ISPs (e.g. full PyNET) are too compute-heavy (>50 GFLOPs), while classical ISPs fail in extreme noise and non-linear lighting.

### Division of Labor in LUMA:
1. **Classical Physics Domain (Linear RAW space):** Exact, zero-cost operations: Black Level Subtraction (BLS) and linear Bayer-domain White Balance (WB) gains.
2. **Neural Restoration Core (Sub-sampled Bayer space):** Packed 4-channel $(H/2, W/2, 4)$ processing cuts MACs by **4×** compared to full-resolution convolutions. Handles joint Poisson-Gaussian denoising and edge-preserving demosaicing via **MobileUNet** (141K parameters, 0.31 MB FP16) or **PyNETLite** (110K parameters, 0.27 MB FP16).
3. **Hybrid Color & Dynamic Range Domain:** Physics-grounded $3\times3$ Color Correction Matrix (CCM), Filmic tone mapping, guided-filter sharpening, and standard sRGB gamma encoding.

```
Degraded Bayer RAW (H, W)
         │
   [Black Level Subtraction & Normalization]
         │
   [Linear White Balance Gains in Bayer Domain]
         │
   [Pack to 4-Channel Sub-sampled Bayer (H/2, W/2, 4)]
         │
   ┌──────────────────────────────────────────────┐
   │  Lightweight Neural Core (MobileUNet / PyNET) │  <-- ~0.5 GFLOPs, ~20ms
   │  Joint Denoise + Demosaic via PixelShuffle   │
   └──────────────────────────────────────────────┘
         │
   Linear RGB (H, W, 3)
         │
   [Learned / Constrained Color Correction Matrix (CCM)]
         │
   [Filmic Tone Mapping + sRGB Gamma Encoding]
         │
   [Luma Guided-Filter Sharpening]
         │
   High-Quality 8-bit sRGB Image
```

---

##  Repository Structure

```text
CV Hackathon/
├── classical_isp/           # Traditional Modular ISP Pipeline
│   ├── black_level.py       # Black level subtraction & normalization
│   ├── white_balance.py     # Gray World, White Patch, Bayer-domain WB gains
│   ├── demosaic.py          # Malvar-He-Cutler (MHC) & Bilinear demosaicing
│   ├── denoise.py           # Bilateral filtering & Non-Local Means (NLM)
│   ├── color_correction.py  # 3x3 CCM application & least-squares patch fitting
│   ├── tone_mapping.py      # Filmic curve, Reinhard, and sRGB piecewise gamma
│   ├── sharpening.py        # Guided filter sharpening & unsharp masking
│   └── pipeline.py          # Full ClassicalISP orchestrator & step extractor
│
├── neural_isp/              # Deep Learning Architectures & Training
│   ├── mobile_unet.py       # Mobile-optimized U-Net with Depthwise-Separable Convs
│   ├── pynet_lite.py        # Multi-scale 3-level lightweight PyNET
│   ├── losses.py            # Composite ISPLoss: L1 + (1-SSIM) + Delta E + LPIPS
│   └── train.py             # Mixed-precision training loop with cosine warmup
│
├── hybrid_isp/              # Hybrid ISP Orchestration
│   ├── pipeline.py          # HybridISP (Classical Pre -> Neural Core -> Classical Post)
│   └── adaptive.py          # AdaptiveHybridISP (noise-level & illumination estimation)
│
├── optimization/            # Mobile Edge Profiling & Efficiency
│   ├── quantize.py          # INT8 Dynamic/Static Quantization & FP16 conversion
│   └── profiler.py          # FLOPs/MACs hook, latency, peak RAM & param counter
│
├── evaluation/              # Quantitative Metrics & Visual Reporting
│   ├── metrics.py           # PSNR, SSIM, CIE Delta E 2000 (ΔE₀₀), LPIPS
│   └── compare.py           # Multi-pipeline comparison grids & metric charts
│
├── datasets/                # RAW Data Loading & Degradation Engine
│   ├── degradation.py       # Poisson-Gaussian noise, WB shifts, exposure drops
│   ├── raw_dataset.py       # SyntheticRAWDataset & Real RAWDataset loaders
│   └── utils.py             # Bayer packing/unpacking & spatial augmentations
│
├── checkpoints/             # Saved model weights (.pth) & training logs
├── outputs/                 # Output comparison grids, metric plots & tables
├── sample_data/             # Sample color chart & training/val image sets
├── demo.py                  # End-to-end 3-way benchmark runner
├── profile_models.py        # Model efficiency & quantization profiling script
└── requirements.txt         # Dependencies
```

---

##  Model Efficiency & Profiling Benchmarks

Benchmarked on CPU (mobile device simulation):

| Model | Parameters | Size (FP32) | Size (FP16) | Compute (MACs) | Compute (FLOPs) | Latency (CPU) | Throughput |
|---|---|---|---|---|---|---|---|
| **MobileUNet** | **141,616** | 0.59 MB | **0.31 MB** | 226.36 M | **452.72 M** | **20.9 ms** | **47.6 FPS** |
| **PyNETLite** | **110,462** | 0.48 MB | **0.27 MB** | 401.71 M | **803.41 M** | **27.4 ms** | **36.5 FPS** |

*Both models achieve sub-0.5 MB footprints in FP16 with sub-30ms execution on edge CPU.*

---

##  Quickstart & Usage

### 0. Launch Production Web Studio (Modern Minimalist UI)
Launch the full interactive production frontend with real-time before/after split comparator, 400% zoom loupe, live telemetry, and stage inspection:
```bash
# Windows One-Click:
run_studio.bat

# Or via Python / PowerShell:
python server.py
```
Open **`http://localhost:8000`** in any browser.

---

### 1. Run Model Efficiency & Quantization Profiler
```bash
python profile_models.py
```

### 2. Run the 3-Way Pipeline Benchmark
Compares **Classical ISP**, **Pure Neural ISP**, and **Hybrid ISP** against ground truth:
```bash
python demo.py --image sample_data/color_chart.png --model_path checkpoints/mobile_unet_best.pth --output_dir outputs
```
This generates:
- `outputs/color_chart_comparison.png`: Side-by-side visual comparison grid.
- `outputs/color_chart_metrics.txt`: Table comparing PSNR, SSIM, Delta E 2000, and Latency.
- `outputs/color_chart_charts.png`: Metric bar charts across all three pipelines.

### 3. Train / Fine-Tune Neural Models
Train the lightweight neural core on any directory of RGB images (converts to synthetic Bayer with realistic Poisson-Gaussian noise on-the-fly):
```bash
python neural_isp/train.py \
    --model mobile_unet \
    --data_dir sample_data/train \
    --val_dir sample_data/val \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3 \
    --crop_size 128 \
    --save_dir checkpoints
```

### 4. Adaptive Processing Under Adverse Conditions
```python
from hybrid_isp.adaptive import AdaptiveHybridISP
import numpy as np

# Automatically adjusts denoising strength, WB, and tone curves based on noise & lux:
adaptive_isp = AdaptiveHybridISP(model_path="checkpoints/mobile_unet_best.pth")
output_rgb = adaptive_isp.process(raw_bayer_mosaic)
```

---

##  Evaluation Metrics Supported
- **PSNR (Peak Signal-to-Noise Ratio)**: Overall signal reconstruction accuracy.
- **SSIM (Structural Similarity Index)**: High-frequency texture and structural fidelity.
- **CIE $\Delta E_{00}$ (Delta E 2000)**: Perceptual color accuracy in CIELAB space under D65 illuminant.
- **LPIPS**: Perceptual deep feature similarity distance.
- **Latency (ms) & Peak Memory (MB)**: Hardware constraint profiling.

