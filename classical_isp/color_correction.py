"""Colour correction matrix (CCM) application and estimation."""

import numpy as np


# Default CCM for a typical smartphone sensor (approximate).
# Designed to map linear sensor RGB to a more accurate colour space.
DEFAULT_CCM: np.ndarray = np.array(
    [
        [1.80, -0.65, -0.15],
        [-0.25, 1.55, -0.30],
        [0.05, -0.45, 1.40],
    ],
    dtype=np.float32,
)


def apply_ccm(
    image: np.ndarray,
    ccm: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a 3×3 Colour Correction Matrix to a linear RGB image.

    Reshapes the image to an (N, 3) matrix, multiplies by the CCM
    (right-multiply: ``out = pixels @ ccm.T``), and reshapes back.

    Args:
        image: (H, W, 3) linear RGB float image.
        ccm: (3, 3) colour correction matrix. If ``None``, uses
            :data:`DEFAULT_CCM`.

    Returns:
        Colour-corrected (H, W, 3) float32 image clipped to [0, 1].
    """
    if ccm is None:
        ccm = DEFAULT_CCM
    ccm = ccm.astype(np.float32)

    h, w, c = image.shape
    pixels = image.reshape(-1, 3).astype(np.float32)  # (N, 3)

    corrected = pixels @ ccm.T  # (N, 3) @ (3, 3)^T → (N, 3)
    corrected = corrected.reshape(h, w, c)

    return np.clip(corrected, 0.0, 1.0)


def compute_ccm_from_patches(
    source_colors: np.ndarray,
    target_colors: np.ndarray,
) -> np.ndarray:
    """Compute an optimal CCM using least-squares from colour patch pairs.

    Solves the overdetermined system ``source @ M^T ≈ target`` for M
    by minimising the sum of squared errors.

    Args:
        source_colors: (N, 3) measured sensor RGB values.
        target_colors: (N, 3) ground-truth sRGB values.

    Returns:
        (3, 3) optimal CCM as float32.
    """
    source = source_colors.astype(np.float64)
    target = target_colors.astype(np.float64)

    # We want M such that  source @ M^T ≈ target
    # Transpose:  M @ source^T ≈ target^T
    # Solve per-row:  M_row @ source^T = target_row^T
    # Or equivalently: source @ M_row^T = target_col
    # Use lstsq:  source @ x = target   →  x = M^T
    ccm_t, _residuals, _rank, _sv = np.linalg.lstsq(source, target, rcond=None)
    ccm = ccm_t.T  # (3, 3)

    return ccm.astype(np.float32)
