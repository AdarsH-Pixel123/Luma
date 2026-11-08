"""Denoising algorithms for RGB images."""

import numpy as np
import cv2


def bilateral_denoise(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75,
) -> np.ndarray:
    """Apply bilateral filter for edge-preserving denoising.

    The bilateral filter smooths an image while preserving edges by
    combining a spatial Gaussian with a range (intensity) Gaussian.

    Args:
        image: (H, W, 3) RGB float image in [0, 1].
        d: Diameter of each pixel neighbourhood. A value of 0 means
            it is computed from ``sigma_space``.
        sigma_color: Filter sigma in the colour/intensity space.
            Larger values mix colours from farther-apart intensities.
        sigma_space: Filter sigma in the coordinate space. Larger
            values include pixels that are spatially farther away.

    Returns:
        Denoised (H, W, 3) float32 image clipped to [0, 1].
    """
    # cv2.bilateralFilter works on uint8; scale accordingly
    img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    # Process in BGR for OpenCV convention
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    denoised_bgr = cv2.bilateralFilter(bgr, d, sigma_color, sigma_space)
    denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)

    result = denoised_rgb.astype(np.float32) / 255.0
    return np.clip(result, 0.0, 1.0)


def nlm_denoise(
    image: np.ndarray,
    h: float = 10,
    template_size: int = 7,
    search_size: int = 21,
) -> np.ndarray:
    """Apply Non-Local Means denoising.

    NLM compares patches in a search window to average similar regions,
    producing excellent denoising with good detail preservation.

    Args:
        image: (H, W, 3) RGB float image in [0, 1].
        h: Filter strength. Higher ``h`` removes more noise but may
            also remove fine detail.
        template_size: Size of the template patch (should be odd).
        search_size: Size of the area where the search is performed
            (should be odd).

    Returns:
        Denoised (H, W, 3) float32 image clipped to [0, 1].
    """
    img_u8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)

    # OpenCV NLM colored expects BGR input
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)

    # hForColorComponents is typically the same as h
    denoised_bgr = cv2.fastNlMeansDenoisingColored(
        bgr,
        None,
        h=h,
        hForColorComponents=h,
        templateWindowSize=template_size,
        searchWindowSize=search_size,
    )
    denoised_rgb = cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB)

    result = denoised_rgb.astype(np.float32) / 255.0
    return np.clip(result, 0.0, 1.0)
