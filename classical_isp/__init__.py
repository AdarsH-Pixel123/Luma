"""Classical ISP (Image Signal Processing) pipeline module.

Provides a complete traditional image processing pipeline for converting
raw Bayer sensor data to finished sRGB images, including:
- Black level subtraction & normalization
- White balance (Gray World, White Patch, Bayer-domain gains)
- Demosaicing (Bilinear, Malvar-He-Cutler)
- Denoising (Bilateral, Non-Local Means)
- Color correction (3x3 CCM)
- Tone mapping (Gamma, sRGB, Reinhard, Filmic)
- Sharpening (Unsharp Mask, Guided Filter)
"""

from .pipeline import ClassicalISP
from .black_level import subtract_black_level
from .white_balance import gray_world_wb, white_patch_wb, apply_wb_gains
from .demosaic import bilinear_demosaic, malvar_demosaic
from .denoise import bilateral_denoise, nlm_denoise
from .color_correction import apply_ccm, compute_ccm_from_patches, DEFAULT_CCM
from .tone_mapping import gamma_correction, srgb_gamma, reinhard_tone_map, filmic_tone_map
from .sharpening import unsharp_mask, guided_filter_sharpen

__all__ = [
    "ClassicalISP",
    "subtract_black_level",
    "gray_world_wb",
    "white_patch_wb",
    "apply_wb_gains",
    "bilinear_demosaic",
    "malvar_demosaic",
    "bilateral_denoise",
    "nlm_denoise",
    "apply_ccm",
    "compute_ccm_from_patches",
    "DEFAULT_CCM",
    "gamma_correction",
    "srgb_gamma",
    "reinhard_tone_map",
    "filmic_tone_map",
    "unsharp_mask",
    "guided_filter_sharpen",
]
