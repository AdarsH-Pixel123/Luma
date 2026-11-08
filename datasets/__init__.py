"""Datasets module for Hybrid Learned Smartphone ISP pipeline.

Provides utilities for Bayer pattern manipulation, synthetic degradation,
and PyTorch Dataset classes for training with both synthetic and real RAW data.
"""

from .utils import (
    bayer_to_4ch,
    four_ch_to_bayer,
    random_crop,
    random_crop_pair,
    augment_pair,
)
from .degradation import (
    add_poisson_gaussian_noise,
    apply_white_balance_error,
    apply_exposure_change,
    degrade_raw,
    rgb_to_synthetic_bayer,
)
from .raw_dataset import SyntheticRAWDataset, RAWDataset

__all__ = [
    # Utils
    "bayer_to_4ch",
    "four_ch_to_bayer",
    "random_crop",
    "random_crop_pair",
    "augment_pair",
    # Degradation
    "add_poisson_gaussian_noise",
    "apply_white_balance_error",
    "apply_exposure_change",
    "degrade_raw",
    "rgb_to_synthetic_bayer",
    # Datasets
    "SyntheticRAWDataset",
    "RAWDataset",
]
