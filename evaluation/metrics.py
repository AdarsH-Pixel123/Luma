"""Image quality metrics for ISP pipeline evaluation.

Implements PSNR, SSIM, CIE Delta E 2000, and LPIPS perceptual distance.
All metrics operate on numpy arrays (H, W, C) in [0, 1] float or [0, 255] uint8
unless otherwise noted.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float64(img: np.ndarray, max_val: float = 1.0) -> np.ndarray:
    """Ensure image is float64 in [0, 1]."""
    img = img.astype(np.float64)
    if img.max() > 1.0 + 1e-6 or max_val > 1.0:
        img = img / max(max_val, img.max())
    return np.clip(img, 0.0, 1.0)


def _gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
    """Create a 1-D Gaussian kernel."""
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    kernel = np.exp(-0.5 * (coords / sigma) ** 2)
    return kernel / kernel.sum()


def _gaussian_filter_2d(img: np.ndarray, win_size: int, sigma: float) -> np.ndarray:
    """Apply separable Gaussian filter to a 2-D image."""
    from scipy.ndimage import uniform_filter  # noqa: F401 – fallback import check

    kernel = _gaussian_kernel_1d(win_size, sigma)
    # Separable convolution: rows then columns
    out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), axis=1, arr=img)
    out = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), axis=0, arr=out)
    return out


# ---------------------------------------------------------------------------
# PSNR
# ---------------------------------------------------------------------------

def compute_psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Both inputs should be float [0, 1] or uint8 [0, 255].
    Auto-detects range from *max_val* or dtype.

    Args:
        pred: Predicted image array.
        target: Ground-truth image array.
        max_val: Maximum pixel value (default 1.0 for float images).

    Returns:
        PSNR in dB.  Returns ``float('inf')`` when MSE == 0.
    """
    if pred.dtype == np.uint8 or target.dtype == np.uint8:
        max_val = 255.0

    pred_f = pred.astype(np.float64)
    target_f = target.astype(np.float64)

    mse = np.mean((pred_f - target_f) ** 2)
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10((max_val ** 2) / mse)


# ---------------------------------------------------------------------------
# SSIM
# ---------------------------------------------------------------------------

def compute_ssim(pred: np.ndarray, target: np.ndarray, win_size: int = 11) -> float:
    """Compute Structural Similarity Index (Wang et al., 2004).

    Implements the full SSIM formula with Gaussian weighting and the
    C1 / C2 stabilisation constants.

    Args:
        pred: Predicted image, shape ``(H, W)`` or ``(H, W, C)``.
        target: Ground-truth image, same shape as *pred*.
        win_size: Side length of the Gaussian window (must be odd).

    Returns:
        Mean SSIM value in [-1, 1].
    """
    if pred.dtype == np.uint8 or target.dtype == np.uint8:
        L = 255.0
    else:
        L = 1.0

    pred_f = pred.astype(np.float64)
    target_f = target.astype(np.float64)

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    sigma = 1.5  # standard SSIM default

    def _ssim_channel(p: np.ndarray, t: np.ndarray) -> float:
        mu_p = _gaussian_filter_2d(p, win_size, sigma)
        mu_t = _gaussian_filter_2d(t, win_size, sigma)

        mu_p_sq = mu_p * mu_p
        mu_t_sq = mu_t * mu_t
        mu_pt = mu_p * mu_t

        sigma_p_sq = _gaussian_filter_2d(p * p, win_size, sigma) - mu_p_sq
        sigma_t_sq = _gaussian_filter_2d(t * t, win_size, sigma) - mu_t_sq
        sigma_pt = _gaussian_filter_2d(p * t, win_size, sigma) - mu_pt

        # Clamp tiny negatives from numerical error
        sigma_p_sq = np.maximum(sigma_p_sq, 0.0)
        sigma_t_sq = np.maximum(sigma_t_sq, 0.0)

        numerator = (2.0 * mu_pt + C1) * (2.0 * sigma_pt + C2)
        denominator = (mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2)
        ssim_map = numerator / denominator

        # Trim borders where the kernel is not fully valid
        pad = win_size // 2
        return float(ssim_map[pad:-pad, pad:-pad].mean())

    if pred_f.ndim == 2:
        return _ssim_channel(pred_f, target_f)

    # Multi-channel: average per-channel SSIM
    channels = pred_f.shape[2]
    return float(np.mean([_ssim_channel(pred_f[:, :, c], target_f[:, :, c]) for c in range(channels)]))


# ---------------------------------------------------------------------------
# CIE Delta E 2000
# ---------------------------------------------------------------------------

def _srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Inverse sRGB companding (sRGB → linear RGB)."""
    linear = np.where(
        srgb <= 0.04045,
        srgb / 12.92,
        ((srgb + 0.055) / 1.055) ** 2.4,
    )
    return linear


def _linear_rgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Linear RGB → CIE XYZ (D65 illuminant, sRGB primaries).

    *rgb* has shape ``(…, 3)`` in [0, 1].
    Returns array of same shape with XYZ values.
    """
    # sRGB to XYZ matrix (D65)
    M = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float64,
    )
    return rgb @ M.T


def _xyz_to_lab(xyz: np.ndarray) -> np.ndarray:
    """CIE XYZ → CIE L*a*b* (D65 illuminant).

    *xyz* has shape ``(…, 3)``.
    Returns array of same shape with L*, a*, b* values.
    """
    # D65 white point
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883

    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn

    delta = 6.0 / 29.0
    delta_sq = delta ** 2
    delta_cb = delta ** 3

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta_cb, np.cbrt(t), t / (3.0 * delta_sq) + 4.0 / 29.0)

    fx, fy, fz = f(x), f(y), f(z)

    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)

    return np.stack([L, a, b], axis=-1)


def _delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE Delta E 2000 per-pixel.

    *lab1*, *lab2* have shape ``(…, 3)`` (L*, a*, b*).
    Returns array of shape ``(…,)`` with ΔE₀₀ values.

    Reference: Sharma, Wu, Dalal – "The CIEDE2000 Color-Difference Formula:
    Implementation Notes, Supplementary Test Data, and Mathematical Observations"
    (Color Research and Application, 2005).
    """
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    # Step 1 ----------------------------------------------------------------
    C1_ab = np.sqrt(a1 ** 2 + b1 ** 2)
    C2_ab = np.sqrt(a2 ** 2 + b2 ** 2)
    C_ab_mean = (C1_ab + C2_ab) / 2.0

    C_ab_mean_pow7 = C_ab_mean ** 7
    G = 0.5 * (1.0 - np.sqrt(C_ab_mean_pow7 / (C_ab_mean_pow7 + 25.0 ** 7)))

    a1_prime = a1 * (1.0 + G)
    a2_prime = a2 * (1.0 + G)

    C1_prime = np.sqrt(a1_prime ** 2 + b1 ** 2)
    C2_prime = np.sqrt(a2_prime ** 2 + b2 ** 2)

    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360.0
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360.0

    # Step 2 ----------------------------------------------------------------
    dL_prime = L2 - L1
    dC_prime = C2_prime - C1_prime

    # dh_prime
    h_diff = h2_prime - h1_prime
    C_prod = C1_prime * C2_prime

    dh_prime = np.where(
        C_prod == 0.0,
        0.0,
        np.where(
            np.abs(h_diff) <= 180.0,
            h_diff,
            np.where(h_diff > 180.0, h_diff - 360.0, h_diff + 360.0),
        ),
    )

    dH_prime = 2.0 * np.sqrt(C_prod) * np.sin(np.radians(dh_prime / 2.0))

    # Step 3 ----------------------------------------------------------------
    L_prime_mean = (L1 + L2) / 2.0
    C_prime_mean = (C1_prime + C2_prime) / 2.0

    # h_prime_mean
    h_sum = h1_prime + h2_prime
    h_prime_mean = np.where(
        C_prod == 0.0,
        h_sum,
        np.where(
            np.abs(h_diff) <= 180.0,
            h_sum / 2.0,
            np.where(h_sum < 360.0, (h_sum + 360.0) / 2.0, (h_sum - 360.0) / 2.0),
        ),
    )

    T = (
        1.0
        - 0.17 * np.cos(np.radians(h_prime_mean - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_prime_mean))
        + 0.32 * np.cos(np.radians(3.0 * h_prime_mean + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_prime_mean - 63.0))
    )

    SL = 1.0 + 0.015 * (L_prime_mean - 50.0) ** 2 / np.sqrt(20.0 + (L_prime_mean - 50.0) ** 2)
    SC = 1.0 + 0.045 * C_prime_mean
    SH = 1.0 + 0.015 * C_prime_mean * T

    C_prime_mean_pow7 = C_prime_mean ** 7
    RC = 2.0 * np.sqrt(C_prime_mean_pow7 / (C_prime_mean_pow7 + 25.0 ** 7))

    d_theta = 30.0 * np.exp(-(((h_prime_mean - 275.0) / 25.0) ** 2))
    RT = -np.sin(np.radians(2.0 * d_theta)) * RC

    # Parametric weighting factors (all 1.0 for the standard formula)
    kL = kC = kH = 1.0

    dE = np.sqrt(
        (dL_prime / (kL * SL)) ** 2
        + (dC_prime / (kC * SC)) ** 2
        + (dH_prime / (kH * SH)) ** 2
        + RT * (dC_prime / (kC * SC)) * (dH_prime / (kH * SH))
    )

    return dE


def compute_delta_e(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute mean CIE Delta E 2000 colour difference.

    Pipeline:
        1. Inverse sRGB gamma → linear RGB
        2. Linear RGB → CIE XYZ (D65)
        3. XYZ → CIE L*a*b*
        4. ΔE₀₀ per pixel, then mean

    Args:
        pred: Predicted sRGB image ``(H, W, 3)`` in [0, 1].
        target: Ground-truth sRGB image ``(H, W, 3)`` in [0, 1].

    Returns:
        Mean Delta E 2000 across all pixels.
    """
    pred_f = _to_float64(pred)
    target_f = _to_float64(target)

    # Flatten spatial dims for batch processing
    shape = pred_f.shape
    pred_flat = pred_f.reshape(-1, 3)
    target_flat = target_f.reshape(-1, 3)

    lab1 = _xyz_to_lab(_linear_rgb_to_xyz(_srgb_to_linear(pred_flat)))
    lab2 = _xyz_to_lab(_linear_rgb_to_xyz(_srgb_to_linear(target_flat)))

    de = _delta_e_2000(lab1, lab2)
    return float(np.mean(de))


# ---------------------------------------------------------------------------
# LPIPS
# ---------------------------------------------------------------------------

def compute_lpips(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "alex",
) -> float:
    """Compute LPIPS perceptual distance.

    Tries to use the ``lpips`` package.  If unavailable, falls back to a
    lightweight VGG-based L2 feature distance (less accurate but functional).

    Args:
        pred: Predicted images ``(B, 3, H, W)`` in [0, 1].
        target: Ground-truth images ``(B, 3, H, W)`` in [0, 1].
        net: Backbone network – ``'alex'`` or ``'vgg'``.

    Returns:
        Mean LPIPS distance (scalar float).
    """
    # Normalise to [-1, 1] as expected by LPIPS
    pred_n = pred.float() * 2.0 - 1.0
    target_n = target.float() * 2.0 - 1.0

    try:
        import lpips as _lpips  # type: ignore[import-untyped]

        loss_fn = _lpips.LPIPS(net=net, verbose=False)
        loss_fn.eval()
        device = pred.device
        loss_fn = loss_fn.to(device)

        with torch.no_grad():
            dist = loss_fn(pred_n, target_n)
        return float(dist.mean().item())

    except ImportError:
        # Fallback: VGG feature extraction approximation
        return _lpips_vgg_fallback(pred_n, target_n)


_CACHED_VGG16 = None

def _get_vgg16_features():
    global _CACHED_VGG16
    if _CACHED_VGG16 is None:
        try:
            from torchvision.models import vgg16, VGG16_Weights
            weights = VGG16_Weights.IMAGENET1K_V1
        except Exception:
            from torchvision.models import vgg16
            weights = "IMAGENET1K_V1"
        _CACHED_VGG16 = vgg16(weights=weights).features.eval()
    return _CACHED_VGG16

def _lpips_vgg_fallback(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Simple VGG-based perceptual distance as LPIPS stand-in.

    Extracts features from selected VGG-16 layers and computes
    channel-normalised L2 distance, averaged over layers and pixels.
    """
    model = _get_vgg16_features()
    device = pred.device
    model = model.to(device)

    # Layer indices whose outputs we compare (relu1_2, relu2_2, relu3_3, relu4_3)
    target_layers = {3, 8, 15, 22}

    def _extract(x: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        for idx, layer in enumerate(model):
            x = layer(x)
            if idx in target_layers:
                feats.append(x)
        return feats

    with torch.no_grad():
        feats_pred = _extract(pred)
        feats_target = _extract(target)

    total = 0.0
    for fp, ft in zip(feats_pred, feats_target):
        # Channel-wise normalise
        fp_norm = fp / (fp.norm(dim=1, keepdim=True) + 1e-10)
        ft_norm = ft / (ft.norm(dim=1, keepdim=True) + 1e-10)
        total += float(((fp_norm - ft_norm) ** 2).mean().item())

    return total / len(feats_pred)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def evaluate_all(pred: np.ndarray, target: np.ndarray) -> dict[str, Optional[float]]:
    """Compute all quality metrics and return as a dictionary.

    Args:
        pred: Predicted image ``(H, W, 3)`` in [0, 1] float or [0, 255] uint8.
        target: Ground-truth image, same format as *pred*.

    Returns:
        ``{'psnr': float, 'ssim': float, 'delta_e': float, 'lpips': float | None}``
        LPIPS is ``None`` when neither ``lpips`` nor ``torchvision`` is available.
    """
    results: dict[str, Optional[float]] = {
        "psnr": compute_psnr(pred, target),
        "ssim": compute_ssim(pred, target),
        "delta_e": compute_delta_e(pred, target),
    }

    # LPIPS requires torch tensors
    try:
        pred_f = _to_float64(pred).astype(np.float32)
        target_f = _to_float64(target).astype(np.float32)

        # (H, W, C) → (1, C, H, W)
        pred_t = torch.from_numpy(pred_f).permute(2, 0, 1).unsqueeze(0)
        target_t = torch.from_numpy(target_f).permute(2, 0, 1).unsqueeze(0)

        results["lpips"] = compute_lpips(pred_t, target_t)
    except Exception:
        results["lpips"] = None

    return results
