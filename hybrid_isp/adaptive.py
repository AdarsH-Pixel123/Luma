"""
Adaptive Hybrid ISP
====================
An intelligent wrapper around :class:`HybridISP` that analyses the raw sensor
data to automatically tune pipeline parameters for the current scene.

Features
--------
* **Noise estimation** – uses the Median Absolute Deviation (MAD) of
  high-frequency detail to robustly estimate sensor noise σ.
* **Illumination classification** – categorises the scene as ``'low_light'``,
  ``'normal'``, or ``'bright'`` from global brightness and histogram shape.
* **Adaptive configuration** – adjusts white-balance gains, tone-mapping curve,
  sharpening strength, and the neural denoising path accordingly.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hybrid_isp.pipeline import HybridISP


class AdaptiveHybridISP:
    """Adaptive version of :class:`HybridISP`.

    Estimates noise level and illumination from the raw input and adjusts
    pipeline parameters dynamically before handing off to ``HybridISP``.
    """

    # Thresholds for illumination classification (on [0, 1] normalised raw)
    _LOW_LIGHT_THRESH: float = 0.12
    _BRIGHT_THRESH: float = 0.55

    # Noise-level boundaries (estimated σ on [0, 1] data)
    _LOW_NOISE_THRESH: float = 0.01
    _HIGH_NOISE_THRESH: float = 0.04

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        """Initialise the adaptive ISP.

        Args:
            model_path: Optional checkpoint for the neural core.
            device:     ``'cpu'`` or ``'cuda'``.
        """
        self.model_path = model_path
        self.device = device
        # We create a fresh HybridISP per image (config changes each time),
        # but share the model weights to avoid re-loading.
        self._base_isp = HybridISP(model_path=model_path, device=device)

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_noise_level(raw: np.ndarray) -> float:
        """Estimate noise standard deviation from a raw Bayer frame.

        Uses the *Median Absolute Deviation* (MAD) estimator on the
        high-frequency Laplacian response, which is robust to image
        content (edges, textures).

        Args:
            raw: ``(H, W)`` raw Bayer frame (float or integer).

        Returns:
            Estimated noise σ (normalised to the data range of *raw*).
        """
        raw_f = raw.astype(np.float64)
        data_range = raw_f.max() - raw_f.min()
        if data_range < 1e-8:
            return 0.0

        # Simple 3×3 Laplacian (high-pass) kernel
        # Convolution via slicing to avoid OpenCV dependency inside this fn.
        h, w = raw_f.shape[:2]
        if h < 4 or w < 4:
            return 0.0

        centre = raw_f[1:-1, 1:-1]
        neighbours = (
            raw_f[:-2, 1:-1]
            + raw_f[2:, 1:-1]
            + raw_f[1:-1, :-2]
            + raw_f[1:-1, 2:]
        )
        laplacian = centre - neighbours / 4.0

        # MAD estimator: σ ≈ MAD / 0.6745
        median_val = np.median(np.abs(laplacian - np.median(laplacian)))
        sigma = median_val / 0.6745

        # Normalise to [0, 1] range
        return float(sigma / data_range)

    @staticmethod
    def estimate_illumination(raw: np.ndarray) -> str:
        """Classify illumination as ``'low_light'``, ``'normal'``, or ``'bright'``.

        Decision is based on the mean pixel value (normalised to [0, 1])
        and the fraction of very dark / very bright pixels.

        Args:
            raw: ``(H, W)`` raw Bayer frame.

        Returns:
            One of ``'low_light'``, ``'normal'``, ``'bright'``.
        """
        raw_f = raw.astype(np.float64)
        dmax = raw_f.max()
        if dmax < 1e-8:
            return "low_light"

        normalised = raw_f / dmax
        mean_val = float(np.mean(normalised))

        # Histogram-based refinement
        dark_frac = float(np.mean(normalised < 0.1))
        bright_frac = float(np.mean(normalised > 0.8))

        if mean_val < AdaptiveHybridISP._LOW_LIGHT_THRESH or dark_frac > 0.7:
            return "low_light"
        if mean_val > AdaptiveHybridISP._BRIGHT_THRESH or bright_frac > 0.5:
            return "bright"
        return "normal"

    def get_adaptive_config(self, raw: np.ndarray) -> dict:
        """Generate a pipeline configuration adapted to *raw*.

        Adjusts WB gains, tone-mapping choice, sharpening parameters,
        and black/white levels based on estimated noise and illumination.

        Args:
            raw: ``(H, W)`` raw Bayer frame.

        Returns:
            Configuration ``dict`` suitable for :class:`HybridISP`.
        """
        noise_sigma = self.estimate_noise_level(raw)
        illumination = self.estimate_illumination(raw)

        config: dict = {}

        # ---- Illumination-dependent settings ----------------------------
        if illumination == "low_light":
            # Boost brightness via higher WB gains and filmic tone-mapping
            config["r_gain"] = 2.0
            config["b_gain"] = 1.6
            config["tone_map"] = "filmic"
            config["black_level"] = 64
            config["white_level"] = 1023
        elif illumination == "bright":
            # Moderate gains, standard sRGB gamma
            config["r_gain"] = 1.3
            config["b_gain"] = 1.1
            config["tone_map"] = "srgb"
            config["black_level"] = 64
            config["white_level"] = 1023
        else:  # normal
            config["r_gain"] = 1.5
            config["b_gain"] = 1.3
            config["tone_map"] = "srgb"
            config["black_level"] = 64
            config["white_level"] = 1023

        # ---- Noise-dependent settings -----------------------------------
        if noise_sigma > self._HIGH_NOISE_THRESH:
            # High noise → softer sharpening to avoid amplifying grain
            config["sharpen_sigma"] = 1.5
            config["sharpen_strength"] = 0.15
        elif noise_sigma < self._LOW_NOISE_THRESH:
            # Low noise → more aggressive sharpening for detail
            config["sharpen_sigma"] = 0.8
            config["sharpen_strength"] = 0.5
        else:
            config["sharpen_sigma"] = 1.0
            config["sharpen_strength"] = 0.3

        return config

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self, raw: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
        """Adaptively process a raw Bayer image.

        1. Analyse the image to determine noise level and illumination.
        2. Build an adaptive config.
        3. Run :class:`HybridISP` with that config.

        Args:
            raw:     ``(H, W)`` Bayer mosaic.
            pattern: Bayer pattern string.

        Returns:
            ``(H, W, 3)`` ``uint8`` sRGB image.
        """
        config = self.get_adaptive_config(raw)

        # Re-use the already-loaded model but apply new config
        self._base_isp.config = config
        return self._base_isp.process(raw, pattern)

    def process_with_info(
        self, raw: np.ndarray, pattern: str = "RGGB"
    ) -> Tuple[np.ndarray, dict]:
        """Like :meth:`process` but also returns the analysis info.

        Returns:
            ``(output_image, info_dict)`` where *info_dict* contains
            ``noise_sigma``, ``illumination``, and the full ``config``.
        """
        noise_sigma = self.estimate_noise_level(raw)
        illumination = self.estimate_illumination(raw)
        config = self.get_adaptive_config(raw)

        self._base_isp.config = config
        output = self._base_isp.process(raw, pattern)

        info = {
            "noise_sigma": noise_sigma,
            "illumination": illumination,
            "config": config,
        }
        return output, info
