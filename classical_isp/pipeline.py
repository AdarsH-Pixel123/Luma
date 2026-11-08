"""Full classical ISP pipeline orchestrator."""

from __future__ import annotations

from typing import Any

import numpy as np

from .black_level import subtract_black_level
from .white_balance import apply_wb_gains
from .demosaic import malvar_demosaic, bilinear_demosaic
from .denoise import bilateral_denoise
from .color_correction import apply_ccm, DEFAULT_CCM
from .tone_mapping import srgb_gamma, filmic_tone_map, gamma_correction
from .sharpening import unsharp_mask


# Default pipeline configuration
_DEFAULT_CONFIG: dict[str, Any] = {
    # Black level
    "black_level": 64,
    "white_level": 1023,
    # White balance
    "wb_r_gain": 1.0,
    "wb_b_gain": 1.0,
    # Demosaicing
    "demosaic_method": "malvar",  # "malvar" or "bilinear"
    # Denoising
    "denoise_d": 5,
    "denoise_sigma_color": 25,
    "denoise_sigma_space": 25,
    "denoise_enabled": True,
    # Colour correction
    "ccm": None,  # None → use DEFAULT_CCM
    "ccm_enabled": False,
    # Tone mapping
    "tone_map_method": "srgb",  # "srgb", "filmic", "gamma"
    "gamma": 2.2,
    # Sharpening
    "sharpen_sigma": 1.0,
    "sharpen_strength": 0.3,
    "sharpen_enabled": True,
}


class ClassicalISP:
    """Complete classical ISP pipeline orchestrator.

    Pipeline order:
        1. Black level subtraction & normalisation
        2. White balance (in Bayer domain)
        3. Demosaicing (Malvar-He-Cutler or bilinear)
        4. Denoising (bilateral filter)
        5. Colour correction (CCM)
        6. Tone mapping (sRGB gamma, filmic, or simple gamma)
        7. Sharpening (unsharp mask)
        8. Clip to [0, 1] and convert to uint8

    Example::

        isp = ClassicalISP({"wb_r_gain": 2.0, "tone_map_method": "filmic"})
        rgb_u8 = isp.process(raw_bayer)

    Args:
        config: Optional dict overriding any default pipeline parameters.
            See :data:`_DEFAULT_CONFIG` for available keys.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config: dict[str, Any] = {**_DEFAULT_CONFIG}
        if config is not None:
            self.config.update(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, raw: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
        """Process a raw Bayer image through the full ISP pipeline.

        Args:
            raw: (H, W) raw Bayer mosaic (integer or float).
            pattern: Bayer colour filter array pattern.

        Returns:
            (H, W, 3) uint8 sRGB image.
        """
        steps = self.process_steps(raw, pattern)
        return steps["output"]

    def process_steps(
        self,
        raw: np.ndarray,
        pattern: str = "RGGB",
    ) -> dict[str, np.ndarray]:
        """Process and return intermediate results at each stage.

        Useful for visualisation, debugging, and quality inspection.

        Args:
            raw: (H, W) raw Bayer mosaic (integer or float).
            pattern: Bayer colour filter array pattern.

        Returns:
            Dict mapping stage names to intermediate images:
                ``"black_level"``, ``"white_balance"``, ``"demosaic"``,
                ``"denoise"``, ``"ccm"``, ``"tone_map"``, ``"sharpen"``,
                ``"output"`` (uint8).
        """
        cfg = self.config
        results: dict[str, np.ndarray] = {}

        # 1. Black level subtraction & normalisation
        img = subtract_black_level(
            raw,
            black_level=cfg["black_level"],
            white_level=cfg["white_level"],
        )
        results["black_level"] = img.copy()

        # 2. White balance in Bayer domain
        img = apply_wb_gains(
            img,
            r_gain=cfg["wb_r_gain"],
            b_gain=cfg["wb_b_gain"],
            pattern=pattern,
        )
        img = np.clip(img, 0.0, 1.0)
        results["white_balance"] = img.copy()

        # 3. Demosaicing
        if cfg["demosaic_method"] == "malvar":
            img = malvar_demosaic(img, pattern=pattern)
        else:
            img = bilinear_demosaic(img, pattern=pattern)
        results["demosaic"] = img.copy()

        # 4. Denoising
        if cfg["denoise_enabled"]:
            img = bilateral_denoise(
                img,
                d=cfg["denoise_d"],
                sigma_color=cfg["denoise_sigma_color"],
                sigma_space=cfg["denoise_sigma_space"],
            )
        results["denoise"] = img.copy()

        # 5. Colour correction
        if cfg["ccm_enabled"]:
            img = apply_ccm(img, ccm=cfg["ccm"])
        results["ccm"] = img.copy()

        # 6. Tone mapping
        method = cfg["tone_map_method"]
        if method == "filmic":
            img = filmic_tone_map(img)
        elif method == "gamma":
            img = gamma_correction(img, gamma=cfg["gamma"])
        else:  # "srgb" (default)
            img = srgb_gamma(img)
        results["tone_map"] = img.copy()

        # 7. Sharpening
        if cfg["sharpen_enabled"]:
            img = unsharp_mask(
                img,
                sigma=cfg["sharpen_sigma"],
                strength=cfg["sharpen_strength"],
            )
        results["sharpen"] = img.copy()

        # 8. Final clip and uint8 conversion
        output = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        results["output"] = output

        return results
