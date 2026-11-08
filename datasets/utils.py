"""Utility functions for Bayer pattern manipulation and data augmentation.

Provides packing/unpacking of Bayer mosaics into 4-channel tensors,
random spatial cropping, and geometric augmentation for RAW+RGB pairs.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Bayer channel order look-up tables
# ---------------------------------------------------------------------------

# Maps each supported CFA pattern string to the (row, col) offsets for
# R, Gr, Gb, B within a 2×2 Bayer tile.  The canonical packed order is
# always [R, Gr, Gb, B] regardless of the physical pattern.
_PATTERN_OFFSETS: dict[str, tuple[tuple[int, int], ...]] = {
    "RGGB": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "BGGR": ((1, 1), (1, 0), (0, 1), (0, 0)),
    "GRBG": ((0, 1), (0, 0), (1, 1), (1, 0)),
    "GBRG": ((1, 0), (1, 1), (0, 0), (0, 1)),
}


def bayer_to_4ch(raw: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Convert a single-channel Bayer mosaic to a 4-channel packed tensor.

    The output channels are ordered **[R, Gr, Gb, B]** irrespective of the
    physical CFA layout specified by *pattern*.

    Args:
        raw: Bayer image of shape ``(H, W)`` with even H and W.
        pattern: One of ``'RGGB'``, ``'BGGR'``, ``'GRBG'``, ``'GBRG'``.

    Returns:
        Packed array of shape ``(H/2, W/2, 4)``.

    Raises:
        ValueError: If *pattern* is unsupported or dimensions are odd.
    """
    if raw.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {raw.shape}")
    if raw.shape[0] % 2 != 0 or raw.shape[1] % 2 != 0:
        raise ValueError(
            f"Height and width must be even, got {raw.shape[0]}×{raw.shape[1]}"
        )
    pattern = pattern.upper()
    if pattern not in _PATTERN_OFFSETS:
        raise ValueError(
            f"Unsupported Bayer pattern '{pattern}'. "
            f"Choose from {list(_PATTERN_OFFSETS.keys())}"
        )

    offsets = _PATTERN_OFFSETS[pattern]
    h, w = raw.shape
    out = np.empty((h // 2, w // 2, 4), dtype=raw.dtype)
    for ch_idx, (dr, dc) in enumerate(offsets):
        out[..., ch_idx] = raw[dr::2, dc::2]
    return out


def four_ch_to_bayer(packed: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
    """Unpack a 4-channel tensor back into a single-channel Bayer mosaic.

    Inverse of :func:`bayer_to_4ch`.  Packed channel order is assumed to be
    ``[R, Gr, Gb, B]``.

    Args:
        packed: Array of shape ``(H/2, W/2, 4)``.
        pattern: Target CFA layout.

    Returns:
        Bayer image of shape ``(H, W)``.
    """
    if packed.ndim != 3 or packed.shape[2] != 4:
        raise ValueError(f"Expected (H, W, 4) array, got shape {packed.shape}")
    pattern = pattern.upper()
    if pattern not in _PATTERN_OFFSETS:
        raise ValueError(
            f"Unsupported Bayer pattern '{pattern}'. "
            f"Choose from {list(_PATTERN_OFFSETS.keys())}"
        )

    offsets = _PATTERN_OFFSETS[pattern]
    ph, pw = packed.shape[:2]
    bayer = np.empty((ph * 2, pw * 2), dtype=packed.dtype)
    for ch_idx, (dr, dc) in enumerate(offsets):
        bayer[dr::2, dc::2] = packed[..., ch_idx]
    return bayer


# ---------------------------------------------------------------------------
# Cropping helpers
# ---------------------------------------------------------------------------


def random_crop(img: np.ndarray, crop_size: int) -> np.ndarray:
    """Extract a random square crop from *img*.

    Works with both ``(H, W)`` and ``(H, W, C)`` arrays.  The crop is always
    aligned to a 2-pixel grid so that Bayer structure is preserved.

    Args:
        img: Input image.
        crop_size: Side length of the square crop (must be even).

    Returns:
        Cropped view of shape ``(crop_size, crop_size[, C])``.

    Raises:
        ValueError: If image is smaller than *crop_size* in either dimension.
    """
    h, w = img.shape[:2]
    if h < crop_size or w < crop_size:
        raise ValueError(
            f"Image ({h}×{w}) is smaller than crop_size={crop_size}"
        )

    # Align to 2-pixel grid for Bayer safety
    max_y = (h - crop_size) // 2 * 2
    max_x = (w - crop_size) // 2 * 2
    top = np.random.randint(0, max_y + 1) if max_y > 0 else 0
    left = np.random.randint(0, max_x + 1) if max_x > 0 else 0
    # Ensure even alignment
    top = top - (top % 2)
    left = left - (left % 2)

    return img[top : top + crop_size, left : left + crop_size]


def random_crop_pair(
    raw: np.ndarray, rgb: np.ndarray, crop_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Synchronized random crop for a RAW + RGB pair.

    Both images must have the **same spatial dimensions** (H, W).  The same
    crop window is applied to both so that pixels stay aligned.

    Args:
        raw: RAW image ``(H, W)`` or ``(H, W, C)``.
        rgb: RGB image ``(H, W, 3)``.
        crop_size: Side length of the crop (must be even).

    Returns:
        Tuple ``(raw_crop, rgb_crop)``.

    Raises:
        ValueError: If spatial sizes differ or are too small.
    """
    h_raw, w_raw = raw.shape[:2]
    h_rgb, w_rgb = rgb.shape[:2]
    if h_raw != h_rgb or w_raw != w_rgb:
        raise ValueError(
            f"Spatial size mismatch: raw ({h_raw}×{w_raw}) vs "
            f"rgb ({h_rgb}×{w_rgb})"
        )
    h, w = h_raw, w_raw
    if h < crop_size or w < crop_size:
        raise ValueError(
            f"Images ({h}×{w}) are smaller than crop_size={crop_size}"
        )

    max_y = (h - crop_size) // 2 * 2
    max_x = (w - crop_size) // 2 * 2
    top = np.random.randint(0, max_y + 1) if max_y > 0 else 0
    left = np.random.randint(0, max_x + 1) if max_x > 0 else 0
    top = top - (top % 2)
    left = left - (left % 2)

    raw_crop = raw[top : top + crop_size, left : left + crop_size]
    rgb_crop = rgb[top : top + crop_size, left : left + crop_size]
    return raw_crop, rgb_crop


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def augment_pair(
    raw: np.ndarray, rgb: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply random horizontal and/or vertical flips to a RAW + RGB pair.

    Flips are applied **consistently** to both arrays so alignment is
    preserved.  Each flip is chosen independently with 50 % probability.

    Args:
        raw: RAW image (any shape whose first two dims are spatial).
        rgb: RGB image (same spatial size as *raw*).

    Returns:
        Tuple ``(raw_aug, rgb_aug)`` — augmented copies.
    """
    raw_aug = raw.copy()
    rgb_aug = rgb.copy()

    # Horizontal flip (left-right)
    if np.random.random() < 0.5:
        raw_aug = np.flip(raw_aug, axis=1).copy()
        rgb_aug = np.flip(rgb_aug, axis=1).copy()

    # Vertical flip (up-down)
    if np.random.random() < 0.5:
        raw_aug = np.flip(raw_aug, axis=0).copy()
        rgb_aug = np.flip(rgb_aug, axis=0).copy()

    return raw_aug, rgb_aug
