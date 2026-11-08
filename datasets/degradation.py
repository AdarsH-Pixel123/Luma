"""Synthetic degradation engine for realistic sensor noise and artifacts.

Simulates the main impairments found in smartphone RAW captures:
  * Heteroscedastic (Poisson–Gaussian) sensor noise
  * White-balance colour-temperature errors
  * Exposure shifts (under- / over-exposure)
  * Conversion of clean RGB images to synthetic Bayer mosaics

These building blocks are combined in :func:`degrade_raw` to create a
one-call degradation pipeline suitable for on-the-fly data generation.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Noise presets  (shot_noise_range, read_noise_range)
# ---------------------------------------------------------------------------

_NOISE_PRESETS: dict[str, dict[str, tuple[float, float]]] = {
    "low": {
        "shot_noise": (0.0005, 0.005),
        "read_noise": (0.001, 0.005),
    },
    "medium": {
        "shot_noise": (0.005, 0.02),
        "read_noise": (0.005, 0.02),
    },
    "high": {
        "shot_noise": (0.02, 0.08),
        "read_noise": (0.02, 0.06),
    },
}


# ---------------------------------------------------------------------------
# Individual degradation functions
# ---------------------------------------------------------------------------


def add_poisson_gaussian_noise(
    image: np.ndarray,
    shot_noise: float = 0.01,
    read_noise: float = 0.005,
) -> np.ndarray:
    """Apply heteroscedastic Poisson–Gaussian sensor noise.

    The noise variance at each pixel is modelled as::

        var(x) = shot_noise * x  +  read_noise ** 2

    where *x* is the (clean) pixel value in [0, 1].  We approximate the
    Poisson component with a Gaussian whose variance equals the signal,
    scaled by *shot_noise*.

    Args:
        image: Input image in ``[0, 1]`` float range (any shape).
        shot_noise: Coefficient for the signal-dependent (shot) term.
        read_noise: Standard deviation of the signal-independent (read) term.

    Returns:
        Noisy image (same shape/dtype), clipped to ``[0, 1]``.
    """
    image = image.astype(np.float32)
    # Signal-dependent (Poisson-like) noise
    sigma_shot = np.sqrt(np.maximum(shot_noise * image, 0.0))
    noise_shot = np.random.randn(*image.shape).astype(np.float32) * sigma_shot
    # Signal-independent (Gaussian read) noise
    noise_read = np.random.randn(*image.shape).astype(np.float32) * read_noise

    noisy = image + noise_shot + noise_read
    return np.clip(noisy, 0.0, 1.0).astype(np.float32)


def apply_white_balance_error(
    image: np.ndarray,
    temp_shift: float | None = None,
) -> np.ndarray:
    """Simulate a white-balance colour-temperature error.

    Multiplies the **R** and **B** channels by random gains to mimic an
    incorrect AWB estimate.  The green channel is left unchanged.

    If the input has 4 channels (packed RGGB: ``[R, Gr, Gb, B]``), channels
    0 and 3 are scaled.  For 3-channel RGB, channels 0 and 2 are scaled.

    Args:
        image: Float image in ``[0, 1]``, shape ``(H, W, 3)`` or
               ``(H, W, 4)``.
        temp_shift: Explicit colour-temperature shift in the range
                    ``[-1, 1]``.  If ``None`` a random shift is sampled from
                    ``U(-0.3, 0.3)``.

    Returns:
        Colour-shifted image clipped to ``[0, 1]``.
    """
    image = image.astype(np.float32).copy()
    if temp_shift is None:
        temp_shift = np.random.uniform(-0.3, 0.3)

    # Convert temperature shift to per-channel gains
    r_gain = 1.0 + temp_shift * np.random.uniform(0.5, 1.5)
    b_gain = 1.0 - temp_shift * np.random.uniform(0.5, 1.5)

    n_ch = image.shape[-1] if image.ndim == 3 else 1
    if n_ch == 4:
        # Packed RGGB: [R, Gr, Gb, B]
        image[..., 0] *= r_gain
        image[..., 3] *= b_gain
    elif n_ch == 3:
        # Standard RGB
        image[..., 0] *= r_gain
        image[..., 2] *= b_gain
    else:
        # Single-channel — nothing meaningful to do
        return image

    return np.clip(image, 0.0, 1.0).astype(np.float32)


def apply_exposure_change(
    image: np.ndarray,
    ev_shift: float | None = None,
) -> np.ndarray:
    """Simulate under- or over-exposure by an EV shift.

    Pixel values are scaled by ``2 ** ev_shift``.  A positive *ev_shift*
    brightens the image (over-exposure) and a negative value darkens it
    (under-exposure).

    Args:
        image: Float image in ``[0, 1]`` (any shape).
        ev_shift: Exposure-value offset.  If ``None`` a random shift in
                  ``U(-1.5, 1.5)`` is sampled.

    Returns:
        Exposure-adjusted image clipped to ``[0, 1]``.
    """
    image = image.astype(np.float32)
    if ev_shift is None:
        ev_shift = np.random.uniform(-1.5, 1.5)

    scale = float(2.0 ** ev_shift)
    adjusted = image * scale
    return np.clip(adjusted, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Bayer mosaic synthesis
# ---------------------------------------------------------------------------

# Row/col offsets for each colour plane inside a 2×2 Bayer tile
_BAYER_OFFSETS: dict[str, dict[str, tuple[int, int]]] = {
    "RGGB": {"R": (0, 0), "Gr": (0, 1), "Gb": (1, 0), "B": (1, 1)},
    "BGGR": {"R": (1, 1), "Gr": (1, 0), "Gb": (0, 1), "B": (0, 0)},
    "GRBG": {"R": (0, 1), "Gr": (0, 0), "Gb": (1, 1), "B": (1, 0)},
    "GBRG": {"R": (1, 0), "Gr": (1, 1), "Gb": (0, 0), "B": (0, 1)},
}


def rgb_to_synthetic_bayer(
    rgb: np.ndarray,
    pattern: str = "RGGB",
) -> np.ndarray:
    """Convert an RGB image to a synthetic Bayer mosaic.

    Each pixel position keeps only the colour channel dictated by the CFA
    pattern.  This is useful for creating training pairs from clean RGB
    images when real RAW captures are unavailable.

    Args:
        rgb: ``(H, W, 3)`` image (float or uint8).  H and W must be even.
        pattern: CFA layout string.

    Returns:
        Single-channel Bayer mosaic ``(H, W)`` of the same dtype as *rgb*.

    Raises:
        ValueError: On unsupported pattern or odd dimensions.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) image, got shape {rgb.shape}")
    pattern = pattern.upper()
    if pattern not in _BAYER_OFFSETS:
        raise ValueError(
            f"Unsupported pattern '{pattern}'. "
            f"Choose from {list(_BAYER_OFFSETS.keys())}"
        )
    h, w = rgb.shape[:2]
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(
            f"Height and width must be even, got {h}×{w}"
        )

    offsets = _BAYER_OFFSETS[pattern]
    bayer = np.empty((h, w), dtype=rgb.dtype)

    # R positions ← red channel
    dr, dc = offsets["R"]
    bayer[dr::2, dc::2] = rgb[dr::2, dc::2, 0]

    # Gr positions ← green channel
    dr, dc = offsets["Gr"]
    bayer[dr::2, dc::2] = rgb[dr::2, dc::2, 1]

    # Gb positions ← green channel
    dr, dc = offsets["Gb"]
    bayer[dr::2, dc::2] = rgb[dr::2, dc::2, 1]

    # B positions ← blue channel
    dr, dc = offsets["B"]
    bayer[dr::2, dc::2] = rgb[dr::2, dc::2, 2]

    return bayer


# ---------------------------------------------------------------------------
# Combined degradation pipeline
# ---------------------------------------------------------------------------


def degrade_raw(
    image: np.ndarray,
    noise_level: str = "medium",
) -> tuple[np.ndarray, dict]:
    """Full degradation pipeline combining noise, WB error, and exposure shift.

    Steps executed in order:
      1. Random exposure change
      2. Random white-balance colour shift
      3. Poisson–Gaussian sensor noise

    The *noise_level* selects a preset that controls the range from which
    noise parameters are sampled (``'low'``, ``'medium'``, or ``'high'``).

    Args:
        image: Clean input image in ``[0, 1]`` float range.  May be
               ``(H, W)``, ``(H, W, 3)``, or ``(H, W, 4)``.
        noise_level: Severity preset name.

    Returns:
        Tuple of ``(degraded_image, params_dict)`` where *params_dict*
        records every random parameter that was applied, enabling
        reproducible augmentation or metadata-conditioned training.

    Raises:
        ValueError: If *noise_level* is not a known preset.
    """
    noise_level = noise_level.lower()
    if noise_level not in _NOISE_PRESETS:
        raise ValueError(
            f"Unknown noise_level '{noise_level}'. "
            f"Choose from {list(_NOISE_PRESETS.keys())}"
        )

    preset = _NOISE_PRESETS[noise_level]
    image = image.astype(np.float32)
    params: dict[str, float] = {}

    # --- 1. Exposure shift ------------------------------------------------
    ev_shift = float(np.random.uniform(-1.0, 1.0))
    image = apply_exposure_change(image, ev_shift=ev_shift)
    params["ev_shift"] = ev_shift

    # --- 2. White-balance error -------------------------------------------
    n_ch = image.shape[-1] if image.ndim == 3 else 1
    if n_ch >= 3:
        temp_shift = float(np.random.uniform(-0.25, 0.25))
        image = apply_white_balance_error(image, temp_shift=temp_shift)
        params["wb_temp_shift"] = temp_shift

    # --- 3. Sensor noise --------------------------------------------------
    shot_lo, shot_hi = preset["shot_noise"]
    read_lo, read_hi = preset["read_noise"]
    shot_noise = float(np.random.uniform(shot_lo, shot_hi))
    read_noise = float(np.random.uniform(read_lo, read_hi))
    image = add_poisson_gaussian_noise(
        image, shot_noise=shot_noise, read_noise=read_noise
    )
    params["shot_noise"] = shot_noise
    params["read_noise"] = read_noise
    params["noise_level"] = noise_level

    return image, params
