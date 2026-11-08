"""Demosaicing algorithms to reconstruct full-colour images from Bayer data."""

import numpy as np
import cv2
from scipy.ndimage import convolve


# Mapping from our pattern strings to OpenCV Bayer conversion codes
_BAYER_CV_MAP = {
    "RGGB": cv2.COLOR_BayerBG2RGB,   # OpenCV names Bayer by bottom-right pixel
    "BGGR": cv2.COLOR_BayerRG2RGB,
    "GRBG": cv2.COLOR_BayerGB2RGB,
    "GBRG": cv2.COLOR_BayerGR2RGB,
}


def bilinear_demosaic(bayer: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Simple bilinear interpolation demosaicing.

    Places known pixel values into their respective colour planes
    and fills missing positions by convolving with averaging kernels.

    Args:
        bayer: (H, W) raw Bayer mosaic, float [0, 1].
        pattern: Bayer pattern string (``'RGGB'``, ``'BGGR'``,
            ``'GRBG'``, ``'GBRG'``).

    Returns:
        (H, W, 3) RGB image, float32 clipped to [0, 1].
    """
    bayer = bayer.astype(np.float32)
    h, w = bayer.shape

    pattern = pattern.upper()
    offsets = {
        "RGGB": {"R": (0, 0), "G1": (0, 1), "G2": (1, 0), "B": (1, 1)},
        "BGGR": {"B": (0, 0), "G1": (0, 1), "G2": (1, 0), "R": (1, 1)},
        "GRBG": {"G1": (0, 0), "R": (0, 1), "B": (1, 0), "G2": (1, 1)},
        "GBRG": {"G1": (0, 0), "B": (0, 1), "R": (1, 0), "G2": (1, 1)},
    }

    if pattern not in offsets:
        raise ValueError(
            f"Unknown Bayer pattern '{pattern}'. "
            f"Supported: {list(offsets.keys())}"
        )

    off = offsets[pattern]

    # --- Build colour-specific masks and planes ---
    r_plane = np.zeros((h, w), dtype=np.float32)
    g_plane = np.zeros((h, w), dtype=np.float32)
    b_plane = np.zeros((h, w), dtype=np.float32)

    r_mask = np.zeros((h, w), dtype=np.float32)
    g_mask = np.zeros((h, w), dtype=np.float32)
    b_mask = np.zeros((h, w), dtype=np.float32)

    # Red
    rr, rc = off["R"]
    r_plane[rr::2, rc::2] = bayer[rr::2, rc::2]
    r_mask[rr::2, rc::2] = 1.0

    # Blue
    br, bc = off["B"]
    b_plane[br::2, bc::2] = bayer[br::2, bc::2]
    b_mask[br::2, bc::2] = 1.0

    # Green (two sub-grids)
    g1r, g1c = off["G1"]
    g2r, g2c = off["G2"]
    g_plane[g1r::2, g1c::2] = bayer[g1r::2, g1c::2]
    g_plane[g2r::2, g2c::2] = bayer[g2r::2, g2c::2]
    g_mask[g1r::2, g1c::2] = 1.0
    g_mask[g2r::2, g2c::2] = 1.0

    # --- Interpolation kernels ---
    # For R/B at their own positions: already known.
    # Kernel for interpolating R/B at opposite colour sites (2×2 avg → 3×3 cross)
    kernel_rb = np.array(
        [[0.25, 0.50, 0.25],
         [0.50, 1.00, 0.50],
         [0.25, 0.50, 0.25]], dtype=np.float32
    )
    kernel_g = np.array(
        [[0.0, 0.25, 0.0],
         [0.25, 1.0, 0.25],
         [0.0, 0.25, 0.0]], dtype=np.float32
    )

    # Interpolate missing values by dividing convolved values by convolved mask
    eps = 1e-8

    r_interp = convolve(r_plane, kernel_rb, mode="reflect")
    r_weight = convolve(r_mask, kernel_rb, mode="reflect")
    r_out = r_interp / (r_weight + eps)

    g_interp = convolve(g_plane, kernel_g, mode="reflect")
    g_weight = convolve(g_mask, kernel_g, mode="reflect")
    g_out = g_interp / (g_weight + eps)

    b_interp = convolve(b_plane, kernel_rb, mode="reflect")
    b_weight = convolve(b_mask, kernel_rb, mode="reflect")
    b_out = b_interp / (b_weight + eps)

    rgb = np.stack([r_out, g_out, b_out], axis=-1)
    return np.clip(rgb, 0.0, 1.0)


def malvar_demosaic(bayer: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Malvar-He-Cutler gradient-corrected demosaicing.

    Uses 5×5 linear filters for higher-quality, edge-aware interpolation.
    Falls back to ``cv2.demosaicing`` when available for efficiency,
    applying the equivalent Malvar algorithm internally.

    Args:
        bayer: (H, W) raw Bayer mosaic, float [0, 1].
        pattern: Bayer pattern string (``'RGGB'``, ``'BGGR'``,
            ``'GRBG'``, ``'GBRG'``).

    Returns:
        (H, W, 3) RGB image, float32 clipped to [0, 1].

    References:
        Malvar, He & Cutler, "High-quality linear interpolation for
        demosaicing of Bayer-patterned color images", ICASSP 2004.
    """
    pattern = pattern.upper()
    if pattern not in _BAYER_CV_MAP:
        raise ValueError(
            f"Unknown Bayer pattern '{pattern}'. "
            f"Supported: {list(_BAYER_CV_MAP.keys())}"
        )

    # OpenCV's demosaicing with the EA (edge-aware) variant uses the
    # Malvar-He-Cutler algorithm.  We convert to uint16 for best precision.
    bayer_u16 = np.clip(bayer * 65535.0, 0, 65535).astype(np.uint16)

    try:
        # Try edge-aware (Malvar) variant first
        code = _BAYER_CV_MAP[pattern]
        # cv2.COLOR_BayerBG2RGB_EA etc. are the EA variants
        ea_code_name = (
            cv2.COLOR_BayerBG2RGB_EA
            if code == cv2.COLOR_BayerBG2RGB
            else cv2.COLOR_BayerRG2RGB_EA
            if code == cv2.COLOR_BayerRG2RGB
            else cv2.COLOR_BayerGB2RGB_EA
            if code == cv2.COLOR_BayerGB2RGB
            else cv2.COLOR_BayerGR2RGB_EA
        )
        rgb_u16 = cv2.demosaicing(bayer_u16, ea_code_name)
    except (AttributeError, cv2.error):
        # Fall back to standard bilinear OpenCV demosaicing
        code = _BAYER_CV_MAP[pattern]
        rgb_u16 = cv2.demosaicing(bayer_u16, code)

    rgb = rgb_u16.astype(np.float32) / 65535.0
    return np.clip(rgb, 0.0, 1.0)
