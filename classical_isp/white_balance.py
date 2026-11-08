"""White balance algorithms for Bayer and RGB domain processing."""

import numpy as np


def gray_world_wb(image: np.ndarray) -> np.ndarray:
    """Apply Gray World white balance to a 3-channel image.

    Assumes the average scene colour is achromatic (grey). Computes
    per-channel means and scales the R and B channels so their means
    match the G channel mean.

    Args:
        image: (H, W, 3) float image, typically in [0, 1].

    Returns:
        White-balanced image clipped to [0, 1], same shape as input.
    """
    image = image.astype(np.float32)
    mean_r = np.mean(image[:, :, 0])
    mean_g = np.mean(image[:, :, 1])
    mean_b = np.mean(image[:, :, 2])

    # Avoid division by zero
    eps = 1e-8
    r_gain = mean_g / (mean_r + eps)
    b_gain = mean_g / (mean_b + eps)

    result = image.copy()
    result[:, :, 0] *= r_gain
    result[:, :, 2] *= b_gain

    return np.clip(result, 0.0, 1.0)


def white_patch_wb(image: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Apply White Patch (Max-RGB) white balance.

    Uses a high percentile instead of the absolute maximum for robustness
    against hot pixels and specular highlights. Each channel is scaled so
    that its ``percentile``-th value maps to 1.0.

    Args:
        image: (H, W, 3) float image, typically in [0, 1].
        percentile: Percentile value (0–100) used as the reference white
            point per channel. Default 99.0.

    Returns:
        White-balanced image clipped to [0, 1], same shape as input.
    """
    image = image.astype(np.float32)
    result = image.copy()

    eps = 1e-8
    for c in range(3):
        channel_max = np.percentile(image[:, :, c], percentile)
        result[:, :, c] /= (channel_max + eps)

    return np.clip(result, 0.0, 1.0)


def apply_wb_gains(
    bayer: np.ndarray,
    r_gain: float,
    b_gain: float,
    pattern: str = "RGGB",
) -> np.ndarray:
    """Apply white balance gains directly in the Bayer domain.

    Multiplies R pixels by ``r_gain`` and B pixels by ``b_gain``.
    G pixels are left unchanged. This is the preferred method when
    WB should be applied before demosaicing.

    Args:
        bayer: (H, W) Bayer mosaic image, float.
        r_gain: Multiplicative gain for the red channel.
        b_gain: Multiplicative gain for the blue channel.
        pattern: Bayer pattern string, one of ``'RGGB'``, ``'BGGR'``,
            ``'GRBG'``, ``'GBRG'``.

    Returns:
        White-balanced Bayer image, same shape and dtype as input.
    """
    bayer = bayer.astype(np.float32).copy()
    pattern = pattern.upper()

    # Determine (row_offset, col_offset) for R and B pixels
    # Pattern string maps: index 0 → (0,0), 1 → (0,1), 2 → (1,0), 3 → (1,1)
    offsets = {
        "RGGB": {"R": (0, 0), "B": (1, 1)},
        "BGGR": {"R": (1, 1), "B": (0, 0)},
        "GRBG": {"R": (0, 1), "B": (1, 0)},
        "GBRG": {"R": (1, 0), "B": (0, 1)},
    }

    if pattern not in offsets:
        raise ValueError(
            f"Unknown Bayer pattern '{pattern}'. "
            f"Supported: {list(offsets.keys())}"
        )

    r_row, r_col = offsets[pattern]["R"]
    b_row, b_col = offsets[pattern]["B"]

    bayer[r_row::2, r_col::2] *= r_gain
    bayer[b_row::2, b_col::2] *= b_gain

    return bayer
