"""
Hybrid ISP Pipeline
====================
Combines classical signal-processing stages with a learned neural core
(MobileUNet) for joint demosaicking and denoising.

Processing stages:
    1. [Classical] Black level subtraction & normalization
    2. [Classical] White balance gains in Bayer domain
    3. [Classical] Pack Bayer to 4-channel (RGGB planes)
    4. [Neural]    Joint denoise + demosaic via MobileUNet (4ch → 3ch RGB)
    5. [Classical] Color Correction Matrix
    6. [Classical] Tone mapping (filmic or sRGB gamma)
    7. [Classical] Sharpening (unsharp mask)
    8. [Classical] Clip and convert to uint8
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is on the path so sibling packages resolve.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classical_isp.black_level import subtract_black_level
from classical_isp.white_balance import apply_wb_gains, gray_world_wb
from classical_isp.color_correction import apply_ccm
from classical_isp.tone_mapping import srgb_gamma, filmic_tone_map
from classical_isp.sharpening import unsharp_mask
from datasets.utils import bayer_to_4ch
from neural_isp.mobile_unet import MobileUNet


class HybridISP:
    """Hybrid ISP Pipeline: Classical Pre-processing → Neural Core → Classical Post-processing.

    Processing stages:
        1. [Classical] Black level subtraction & normalization
        2. [Classical] White balance gains in Bayer domain
        3. [Classical] Pack Bayer to 4-channel
        4. [Neural]    Joint denoise + demosaic via MobileUNet (4ch → 3ch RGB)
        5. [Classical] Color Correction Matrix
        6. [Classical] Tone mapping (filmic or sRGB gamma)
        7. [Classical] Sharpening (unsharp mask)
        8. [Classical] Clip and convert to uint8
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        config: Optional[dict] = None,
    ) -> None:
        """Initialise the hybrid ISP pipeline.

        Args:
            model_path: Path to a trained neural ISP checkpoint (``*.pth``).
                        If *None* or the file does not exist the model is used
                        with its random initialisation (useful for benchmarking).
            device:     ``'cpu'`` or ``'cuda'``.
            config:     Optional dict that overrides default pipeline parameters.
                        Recognised keys: ``black_level``, ``white_level``,
                        ``r_gain``, ``b_gain``, ``tone_map`` (``'srgb'`` |
                        ``'filmic'``), ``sharpen_sigma``, ``sharpen_strength``.
        """
        self.device = device
        self.config: dict = config or {}

        # ---- Neural model ---------------------------------------------------
        self.model = MobileUNet(in_channels=4, base_filters=32)
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def process(self, raw: np.ndarray, pattern: str = "RGGB") -> np.ndarray:
        """Process a raw Bayer image through the full hybrid pipeline.

        Args:
            raw:     ``(H, W)`` Bayer mosaic (integer or float, typically
                     10-bit sensor values in ``[0, 1023]``).
            pattern: Bayer pattern string, e.g. ``'RGGB'``.

        Returns:
            ``(H, W, 3)`` ``uint8`` sRGB image.
        """
        # Stage 1: Black level subtraction & normalisation ----------------
        normalized = subtract_black_level(
            raw,
            black_level=self.config.get("black_level", 64),
            white_level=self.config.get("white_level", 1023),
        )

        # Stage 2: White balance in Bayer domain --------------------------
        r_gain: float = self.config.get("r_gain", 1.0)
        b_gain: float = self.config.get("b_gain", 1.0)
        wb_raw = apply_wb_gains(normalized, r_gain, b_gain, pattern) if (r_gain != 1.0 or b_gain != 1.0) else normalized

        # Stage 3: Pack Bayer to 4-channel planes -------------------------
        packed = bayer_to_4ch(wb_raw, pattern)  # (H/2, W/2, 4)

        # Stage 4: Neural denoise + demosaic core -------------------------
        rgb_core = self._run_neural_core(packed)

        # Stage 5: Optional Color Correction Matrix -----------------------
        if self.config.get("apply_ccm", False):
            rgb_core = apply_ccm(rgb_core)

        # Stage 6: Optional Tone mapping ----------------------------------
        if self.config.get("apply_tone_map", False):
            rgb_core = self._tone_map(rgb_core)

        # Stage 7: Edge enhancement / sharpening --------------------------
        if self.config.get("sharpen_enabled", True):
            sigma = self.config.get("sharpen_sigma", 0.8)
            strength = self.config.get("sharpen_strength", 0.15)
            rgb_core = unsharp_mask(rgb_core, sigma=sigma, strength=strength)

        # Stage 8: Clip & convert to uint8 --------------------------------
        result = np.clip(rgb_core * 255.0, 0, 255).astype(np.uint8)
        return result

    @torch.no_grad()
    def process_steps(
        self, raw: np.ndarray, pattern: str = "RGGB"
    ) -> Dict[str, np.ndarray]:
        """Process and return intermediate results at every pipeline stage.

        This is the same computation as :meth:`process` but every intermediate
        image is stored in the returned dictionary so the caller can visualise
        the contribution of each stage.

        Args:
            raw:     ``(H, W)`` Bayer mosaic.
            pattern: Bayer pattern string.

        Returns:
            Dictionary mapping stage name → ``np.ndarray`` image.  All images
            are converted to ``uint8`` (3-channel where applicable) so they
            can be saved / displayed directly.
        """
        steps: Dict[str, np.ndarray] = {}

        # 0. Raw input (visualise as single-channel pseudo-colour) ---------
        raw_vis = np.clip(raw.astype(np.float32), 0, None)
        raw_norm = raw_vis / (raw_vis.max() + 1e-8)
        steps["0_raw_input"] = (np.stack([raw_norm] * 3, axis=-1) * 255).astype(
            np.uint8
        )

        # 1. Black level subtraction & normalisation -----------------------
        normalized = subtract_black_level(
            raw,
            black_level=self.config.get("black_level", 64),
            white_level=self.config.get("white_level", 1023),
        )
        bl_vis = np.clip(normalized, 0.0, 1.0)
        steps["1_black_level"] = (np.stack([bl_vis] * 3, axis=-1) * 255).astype(
            np.uint8
        )

        # 2. White balance in Bayer domain ---------------------------------
        r_gain: float = self.config.get("r_gain", 1.5)
        b_gain: float = self.config.get("b_gain", 1.3)
        wb_raw = apply_wb_gains(normalized, r_gain, b_gain, pattern)
        wb_vis = np.clip(wb_raw, 0.0, 1.0)
        steps["2_white_balance"] = (np.stack([wb_vis] * 3, axis=-1) * 255).astype(
            np.uint8
        )

        # 3. Pack Bayer to 4-channel planes --------------------------------
        packed = bayer_to_4ch(wb_raw, pattern)  # (H/2, W/2, 4)
        # Visualise as pseudo-RGB using the R, mean(G), B channels
        pack_vis = np.stack(
            [
                packed[:, :, 0],
                (packed[:, :, 1] + packed[:, :, 2]) / 2.0,
                packed[:, :, 3],
            ],
            axis=-1,
        )
        pack_vis = np.clip(pack_vis, 0.0, 1.0)
        steps["3_packed_4ch"] = (pack_vis * 255).astype(np.uint8)

        # 4. Neural denoise + demosaic ------------------------------------
        linear_rgb = self._run_neural_core(packed)
        steps["4_neural_demosaic"] = (np.clip(linear_rgb, 0.0, 1.0) * 255).astype(
            np.uint8
        )

        # 5. Color Correction Matrix --------------------------------------
        corrected = apply_ccm(linear_rgb)
        steps["5_color_correction"] = (np.clip(corrected, 0.0, 1.0) * 255).astype(
            np.uint8
        )

        # 6. Tone mapping -------------------------------------------------
        tone_mapped = self._tone_map(corrected)
        steps["6_tone_mapping"] = (np.clip(tone_mapped, 0.0, 1.0) * 255).astype(
            np.uint8
        )

        # 7. Sharpening ---------------------------------------------------
        sigma = self.config.get("sharpen_sigma", 1.0)
        strength = self.config.get("sharpen_strength", 0.3)
        sharpened = unsharp_mask(tone_mapped, sigma=sigma, strength=strength)
        steps["7_sharpening"] = (np.clip(sharpened, 0.0, 1.0) * 255).astype(np.uint8)

        # 8. Final output -------------------------------------------------
        result = np.clip(sharpened * 255.0, 0, 255).astype(np.uint8)
        steps["8_final_output"] = result

        return steps

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_neural_core(self, packed: np.ndarray) -> np.ndarray:
        """Run MobileUNet on packed 4-channel Bayer planes.

        Args:
            packed: ``(H/2, W/2, 4)`` float array.

        Returns:
            ``(H, W, 3)`` linear-RGB float array clipped to ``[0, 1]``.
        """
        tensor = (
            torch.from_numpy(packed.transpose(2, 0, 1))
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        # Pad spatial dims to multiples of 8 for the encoder.
        _, _, h, w = tensor.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h > 0 or pad_w > 0:
            tensor = torch.nn.functional.pad(
                tensor, (0, pad_w, 0, pad_h), mode="reflect"
            )

        output = self.model(tensor)

        # Remove padding (output is 2× spatial if UNet up-samples).
        out_h, out_w = output.shape[2], output.shape[3]
        target_h = min(h * 2, out_h)
        target_w = min(w * 2, out_w)
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :target_h, :target_w]

        linear_rgb: np.ndarray = (
            output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
        )
        return np.clip(linear_rgb, 0.0, 1.0)

    def _tone_map(self, linear_rgb: np.ndarray) -> np.ndarray:
        """Apply the configured tone-mapping curve.

        Args:
            linear_rgb: ``(H, W, 3)`` linear float image.

        Returns:
            Tone-mapped float image.
        """
        method = self.config.get("tone_map", "srgb")
        if method == "filmic":
            return filmic_tone_map(linear_rgb)
        return srgb_gamma(linear_rgb)
