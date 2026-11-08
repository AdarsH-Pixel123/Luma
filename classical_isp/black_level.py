"""Black level subtraction and normalization for raw sensor data."""

import numpy as np


def subtract_black_level(
    raw: np.ndarray,
    black_level: int = 64,
    white_level: int = 1023,
) -> np.ndarray:
    """Subtract black level and normalize to [0, 1] float range.

    Removes the sensor's optical black offset and rescales pixel values so
    that ``black_level`` maps to 0.0 and ``white_level`` maps to 1.0.
    The result is clipped to [0, 1] to handle any out-of-range values.

    Args:
        raw: Input raw image of any shape. May be integer or float dtype.
        black_level: Sensor black level (pedestal value). Typical 10-bit
            sensors use 64.
        white_level: Sensor saturation / white level. Typical 10-bit sensors
            use 1023.

    Returns:
        Normalized float32 image with values in [0, 1].
    """
    raw = raw.astype(np.float32)
    # If data is already normalized to [0, 1] float range and integer black level is passed,
    # avoid subtracting integer pedestal from [0, 1] values
    if raw.max() <= 1.0 and black_level >= 1.0:
        return np.clip(raw, 0.0, 1.0)
    normalized = (raw - black_level) / (white_level - black_level)
    return np.clip(normalized, 0.0, 1.0)
