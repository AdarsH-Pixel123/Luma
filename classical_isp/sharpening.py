"""Sharpening filters for perceptual quality enhancement."""

import numpy as np
import cv2


def unsharp_mask(
    image: np.ndarray,
    sigma: float = 1.0,
    strength: float = 0.5,
) -> np.ndarray:
    """Apply unsharp masking for edge enhancement.

    Blurs the image with a Gaussian, subtracts it from the original
    to extract high-frequency detail, and adds the scaled detail back.

    ``sharpened = image + strength * (image - blurred)``

    Args:
        image: (H, W, 3) or (H, W) float image in [0, 1].
        sigma: Standard deviation of the Gaussian blur kernel.
        strength: Scaling factor for the high-frequency detail.
            0.0 = no sharpening, 1.0 = full contrast doubling of edges.

    Returns:
        Sharpened float32 image clipped to [0, 1].
    """
    image = image.astype(np.float32)

    # Kernel size derived from sigma, must be odd
    ksize = int(np.ceil(sigma * 6)) | 1  # ensure odd
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigma)

    detail = image - blurred
    sharpened = image + strength * detail

    return np.clip(sharpened, 0.0, 1.0)


def _box_filter(src: np.ndarray, radius: int) -> np.ndarray:
    """Mean (box) filter with reflect border handling.

    Args:
        src: 2-D float32 array.
        radius: Half-size of the box kernel.

    Returns:
        Filtered 2-D float32 array, same shape.
    """
    ksize = 2 * radius + 1
    return cv2.blur(src, (ksize, ksize), borderType=cv2.BORDER_REFLECT_101)


def guided_filter_sharpen(
    image: np.ndarray,
    radius: int = 4,
    eps: float = 0.01,
    strength: float = 0.5,
) -> np.ndarray:
    """Edge-aware sharpening using a guided filter.

    Uses the luminance channel as the guide. Computes the guided-filter
    output, extracts the high-frequency residual, and blends it back
    with a controllable ``strength``.

    The guided filter is implemented via the linear model:

        ``a_k = cov(I, p) / (var(I) + eps)``
        ``b_k = mean(p) - a_k * mean(I)``

    Args:
        image: (H, W, 3) float image in [0, 1].
        radius: Window radius for the guided filter box means.
        eps: Regularisation parameter. Smaller values produce more
            aggressive edge-preserving; larger values give smoother output.
        strength: Sharpening strength (0.0 = none, 1.0 = full).

    Returns:
        Sharpened (H, W, 3) float32 image clipped to [0, 1].
    """
    image = image.astype(np.float32)

    # Compute luminance guide (Rec. 709)
    guide = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]

    result = image.copy()

    for c in range(image.shape[2]):
        p = image[:, :, c]

        mean_I = _box_filter(guide, radius)
        mean_p = _box_filter(p, radius)
        mean_Ip = _box_filter(guide * p, radius)
        mean_II = _box_filter(guide * guide, radius)

        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I

        mean_a = _box_filter(a, radius)
        mean_b = _box_filter(b, radius)

        # Guided-filter output (smoothed version of p)
        q = mean_a * guide + mean_b

        # High-frequency detail = original − smoothed
        detail = p - q
        result[:, :, c] = p + strength * detail

    return np.clip(result, 0.0, 1.0)
