"""Tone mapping and gamma correction operators."""

import numpy as np


def gamma_correction(image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
    """Apply simple gamma correction: ``output = input ** (1 / gamma)``.

    Args:
        image: Float image in [0, 1] (any shape).
        gamma: Gamma value. Standard displays use 2.2.

    Returns:
        Gamma-corrected float32 image clipped to [0, 1].
    """
    image = np.clip(image.astype(np.float32), 0.0, 1.0)
    return np.power(image, 1.0 / gamma)


def srgb_gamma(image: np.ndarray) -> np.ndarray:
    """Apply the standard sRGB piece-wise gamma curve.

    The sRGB transfer function is:
        - ``12.92 * x``  for ``x <= 0.0031308``
        - ``1.055 * x^(1/2.4) - 0.055``  otherwise

    Args:
        image: Linear float image in [0, 1] (any shape).

    Returns:
        sRGB-encoded float32 image clipped to [0, 1].
    """
    image = np.clip(image.astype(np.float32), 0.0, 1.0)

    linear_mask = image <= 0.0031308
    result = np.where(
        linear_mask,
        12.92 * image,
        1.055 * np.power(np.maximum(image, 0.0031308), 1.0 / 2.4) - 0.055,
    )
    return np.clip(result, 0.0, 1.0)


def reinhard_tone_map(
    image: np.ndarray,
    key: float = 0.18,
    white: float = 1.0,
) -> np.ndarray:
    """Reinhard global tone mapping operator.

    Computes the log-average luminance of the scene, scales the image
    by ``key / avg_luminance``, then compresses highlights using the
    extended Reinhard formula with a white point.

    Args:
        image: (H, W, 3) linear HDR float image (values may exceed 1).
        key: Middle-grey key value controlling overall brightness.
            0.18 is the standard photographic middle grey.
        white: The smallest luminance that maps to pure white. Higher
            values preserve more highlight detail.

    Returns:
        Tone-mapped (H, W, 3) float32 image clipped to [0, 1].
    """
    image = image.astype(np.float32)
    eps = 1e-8

    # Compute luminance (Rec. 709 coefficients)
    luminance = (
        0.2126 * image[:, :, 0]
        + 0.7152 * image[:, :, 1]
        + 0.0722 * image[:, :, 2]
    )

    # Log-average luminance
    log_avg = np.exp(np.mean(np.log(luminance + eps)))

    # Scale luminance
    l_scaled = (key / (log_avg + eps)) * luminance

    # Extended Reinhard: L_d = L * (1 + L / white^2) / (1 + L)
    white_sq = white * white
    l_mapped = l_scaled * (1.0 + l_scaled / (white_sq + eps)) / (1.0 + l_scaled)

    # Apply same scaling to colour channels (preserve chrominance)
    scale = l_mapped / (l_scaled + eps)
    result = image * (key / (log_avg + eps))
    result = result * scale[:, :, np.newaxis]

    return np.clip(result, 0.0, 1.0)


def filmic_tone_map(image: np.ndarray) -> np.ndarray:
    """Uncharted 2 / filmic tone mapping curve.

    Uses the classic Hable/Uncharted 2 parameterisation with constants
    A, B, C, D, E, F that produce a pleasing S-shaped contrast curve
    with shoulder compression and toe lift.

    Args:
        image: (H, W, 3) linear HDR float image (values may exceed 1).

    Returns:
        Tone-mapped (H, W, 3) float32 image clipped to [0, 1].
    """
    image = np.maximum(image.astype(np.float32), 0.0)

    # Uncharted 2 parameters
    A = 0.15  # Shoulder strength
    B = 0.50  # Linear strength
    C = 0.10  # Linear angle
    D = 0.20  # Toe strength
    E = 0.02  # Toe numerator
    F = 0.30  # Toe denominator

    def _uncharted2(x: np.ndarray) -> np.ndarray:
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F

    # Exposure bias (controls overall brightness)
    exposure_bias = 2.0
    curr = _uncharted2(image * exposure_bias)

    # White scale: normalize so that the white point maps to 1.0
    w = 11.2
    white_scale = 1.0 / _uncharted2(np.array([w], dtype=np.float32))
    result = curr * white_scale

    return np.clip(result, 0.0, 1.0)
