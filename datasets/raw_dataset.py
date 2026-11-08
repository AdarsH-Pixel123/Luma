"""PyTorch Dataset classes for RAW-to-RGB ISP training.

Provides two datasets:

* :class:`SyntheticRAWDataset` — generates synthetic Bayer RAW data from
  clean RGB images with on-the-fly degradation.  Useful when real RAW
  captures are unavailable.
* :class:`RAWDataset` — loads pre-existing paired RAW + RGB data from disk
  (e.g. the Zurich RAW-to-RGB dataset stored as ``.npy`` or image files).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .degradation import degrade_raw, rgb_to_synthetic_bayer
from .utils import augment_pair, bayer_to_4ch, random_crop, random_crop_pair

# Supported image extensions (case-insensitive)
_IMG_EXTENSIONS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _collect_image_paths(directory: str) -> list[str]:
    """Recursively collect paths to image files under *directory*.

    Args:
        directory: Root folder to scan.

    Returns:
        Sorted list of absolute file paths.
    """
    paths: list[str] = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            if Path(fname).suffix.lower() in _IMG_EXTENSIONS:
                paths.append(os.path.join(root, fname))
    return sorted(paths)


def _collect_array_paths(directory: str) -> list[str]:
    """Collect ``.npy`` and image file paths under *directory*.

    Args:
        directory: Root folder to scan.

    Returns:
        Sorted list of absolute file paths.
    """
    valid_ext = _IMG_EXTENSIONS | {".npy"}
    paths: list[str] = []
    for entry in sorted(os.listdir(directory)):
        full = os.path.join(directory, entry)
        if os.path.isfile(full) and Path(full).suffix.lower() in valid_ext:
            paths.append(full)
    return paths


def _load_image_as_float32(path: str) -> np.ndarray:
    """Load an image file and return a float32 array in [0, 1].

    Uses OpenCV to read the file and converts BGR → RGB.

    Args:
        path: Absolute path to the image file.

    Returns:
        ``(H, W, 3)`` float32 array in ``[0, 1]``.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If OpenCV cannot decode the file.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def _load_array(path: str) -> np.ndarray:
    """Load a ``.npy`` file or an image and return a float32 array in [0, 1].

    Args:
        path: Path to a ``.npy`` or image file.

    Returns:
        float32 array normalised to ``[0, 1]``.
    """
    ext = Path(path).suffix.lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
        # If already in [0, 1] keep as is; otherwise normalise from uint range
        if arr.max() > 1.0:
            arr = arr / 255.0 if arr.max() <= 255.0 else arr / arr.max()
        return arr
    return _load_image_as_float32(path)


def _numpy_hwc_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """Convert an ``(H, W, C)`` numpy array to a ``(C, H, W)`` float32 tensor."""
    if arr.ndim == 2:
        return torch.from_numpy(arr).unsqueeze(0).float()
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()


# ============================================================================
# SyntheticRAWDataset
# ============================================================================


class SyntheticRAWDataset(Dataset):
    """Generate synthetic degraded Bayer RAW data from clean RGB images.

    For each sample the pipeline is:
      1. Load a high-quality RGB image.
      2. Random-crop to ``crop_size × crop_size`` (even alignment).
      3. Convert to a synthetic Bayer mosaic via CFA sub-sampling.
      4. Apply on-the-fly degradation (noise, WB error, exposure shift).
      5. Pack degraded Bayer into a 4-channel ``[R, Gr, Gb, B]`` tensor.
      6. Return ``(degraded_4ch, clean_rgb, metadata)``.

    All tensors are float32 in ``[0, 1]``, CHW format.

    Args:
        image_dir: Directory containing clean RGB images (JPG / PNG).
        crop_size: Spatial crop size (must be even).  Default ``256``.
        noise_level: Degradation severity (``'low'``, ``'medium'``,
                     ``'high'``).  Default ``'medium'``.
        pattern: Bayer CFA pattern string.  Default ``'RGGB'``.
        augment: Whether to apply random flips.  Default ``True``.
    """

    def __init__(
        self,
        image_dir: str,
        crop_size: int = 256,
        noise_level: str = "medium",
        pattern: str = "RGGB",
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.noise_level = noise_level
        self.pattern = pattern.upper()
        self.augment = augment

        self.image_paths = _collect_image_paths(image_dir)
        if len(self.image_paths) == 0:
            raise FileNotFoundError(
                f"No images found in '{image_dir}'. "
                "Supported formats: " + ", ".join(sorted(_IMG_EXTENSIONS))
            )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return ``(degraded_4ch, clean_rgb, metadata)``.

        Returns:
            degraded_4ch: ``(4, crop_size/2, crop_size/2)`` float32 tensor.
            clean_rgb:    ``(3, crop_size, crop_size)`` float32 tensor.
            metadata:     Dict with degradation params and source file name.
        """
        path = self.image_paths[index]
        rgb = _load_image_as_float32(path)  # (H, W, 3)

        # Ensure image is large enough; resize if needed
        h, w = rgb.shape[:2]
        if h < self.crop_size or w < self.crop_size:
            scale = max(self.crop_size / h, self.crop_size / w) + 0.01
            rgb = cv2.resize(
                rgb,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_LINEAR,
            )

        # Ensure even dimensions
        h, w = rgb.shape[:2]
        rgb = rgb[: h - h % 2, : w - w % 2]

        # Random crop (clean RGB)
        rgb_crop = random_crop(rgb, self.crop_size)  # (cs, cs, 3)

        # Synthetic Bayer from clean RGB
        bayer_clean = rgb_to_synthetic_bayer(rgb_crop, pattern=self.pattern)  # (cs, cs)

        # Degrade
        degraded_bayer, deg_params = degrade_raw(
            bayer_clean, noise_level=self.noise_level
        )  # (cs, cs)

        # Augment (flip both consistently)
        if self.augment:
            # Expand bayer dims for augment_pair which expects ≥2-D
            degraded_bayer_3d = degraded_bayer[..., np.newaxis]  # (cs, cs, 1)
            degraded_bayer_3d, rgb_crop = augment_pair(degraded_bayer_3d, rgb_crop)
            degraded_bayer = degraded_bayer_3d[..., 0]

        # Pack degraded Bayer → 4-channel
        packed = bayer_to_4ch(degraded_bayer, pattern=self.pattern)  # (cs/2, cs/2, 4)

        # Convert to CHW tensors
        degraded_tensor = _numpy_hwc_to_tensor(packed)     # (4, cs/2, cs/2)
        clean_tensor = _numpy_hwc_to_tensor(rgb_crop)      # (3, cs, cs)

        metadata: dict[str, Any] = {
            "source_file": os.path.basename(path),
            "pattern": self.pattern,
            **deg_params,
        }

        return degraded_tensor, clean_tensor, metadata

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_images={len(self)}, "
            f"crop={self.crop_size}, "
            f"noise='{self.noise_level}', "
            f"pattern='{self.pattern}')"
        )


# ============================================================================
# RAWDataset (real paired data)
# ============================================================================


class RAWDataset(Dataset):
    """Load real paired RAW + RGB data from disk.

    Expects two directories with matching filenames (ignoring extensions).
    RAW files may be ``.npy`` arrays (preferred) or images; RGB files are
    standard images.

    All tensors are float32 in ``[0, 1]``, CHW format.

    Args:
        raw_dir: Directory containing RAW files (``.npy`` or images).
        rgb_dir: Directory containing corresponding RGB target images.
        crop_size: Random crop size.  Default ``256``.
        pattern: Bayer CFA pattern (used when RAW is 2-D).  Default
                 ``'RGGB'``.
        augment: Apply random flips.  Default ``True``.
    """

    def __init__(
        self,
        raw_dir: str,
        rgb_dir: str,
        crop_size: int = 256,
        pattern: str = "RGGB",
        augment: bool = True,
    ) -> None:
        super().__init__()
        self.raw_dir = raw_dir
        self.rgb_dir = rgb_dir
        self.crop_size = crop_size
        self.pattern = pattern.upper()
        self.augment = augment

        self.raw_paths = _collect_array_paths(raw_dir)
        self.rgb_paths = _collect_array_paths(rgb_dir)

        if len(self.raw_paths) == 0:
            raise FileNotFoundError(f"No files found in RAW directory: {raw_dir}")
        if len(self.rgb_paths) == 0:
            raise FileNotFoundError(f"No files found in RGB directory: {rgb_dir}")
        if len(self.raw_paths) != len(self.rgb_paths):
            raise ValueError(
                f"Mismatch: {len(self.raw_paths)} RAW files vs "
                f"{len(self.rgb_paths)} RGB files"
            )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.raw_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Return ``(raw_4ch, clean_rgb, metadata)``.

        If the RAW file is a 2-D Bayer mosaic it is packed into 4 channels.
        If it is already ``(H, W, 4)`` it is used directly.

        Returns:
            raw_4ch:   ``(4, H', W')`` float32 tensor.
            clean_rgb: ``(3, H', W')`` float32 tensor.
            metadata:  Dict with source filenames and pattern.
        """
        raw_path = self.raw_paths[index]
        rgb_path = self.rgb_paths[index]

        raw = _load_array(raw_path)  # (H, W) or (H, W, 4)
        rgb = _load_array(rgb_path)  # (H, W, 3)

        # Ensure RGB is 3-channel
        if rgb.ndim == 2:
            rgb = np.stack([rgb] * 3, axis=-1)

        # Handle RAW dimensionality
        if raw.ndim == 2:
            # Single-channel Bayer — need to align spatial sizes for crop
            # RAW is full-res Bayer (H, W), RGB is (H, W, 3) — same spatial
            h = min(raw.shape[0], rgb.shape[0])
            w = min(raw.shape[1], rgb.shape[1])
            # Make even
            h = h - h % 2
            w = w - w % 2
            raw = raw[:h, :w]
            rgb = rgb[:h, :w]

            # Synchronized crop
            raw, rgb = random_crop_pair(raw, rgb, self.crop_size)

            # Augment
            if self.augment:
                raw_3d = raw[..., np.newaxis]
                raw_3d, rgb = augment_pair(raw_3d, rgb)
                raw = raw_3d[..., 0]

            # Pack Bayer → 4ch
            packed = bayer_to_4ch(raw, pattern=self.pattern)  # (cs/2, cs/2, 4)
            raw_tensor = _numpy_hwc_to_tensor(packed)
            rgb_tensor = _numpy_hwc_to_tensor(rgb)

        elif raw.ndim == 3 and raw.shape[2] == 4:
            # Already packed 4-channel RAW — RGB should be 2× spatial size
            # or same size depending on dataset conventions
            rh, rw = raw.shape[:2]
            gh, gw = rgb.shape[:2]

            # If RGB is 2× the RAW spatial size (common for packed Bayer)
            if gh == rh * 2 and gw == rw * 2:
                crop_raw = self.crop_size // 2
                crop_rgb = self.crop_size

                # Crop RAW
                raw = random_crop(raw, crop_raw)

                # Derive matching RGB crop position (we need to redo this
                # synchronously — use the same seed trick)
                # Instead, manually compute the position
                max_y = (rh - crop_raw) // 2 * 2
                max_x = (rw - crop_raw) // 2 * 2
                top = np.random.randint(0, max(max_y, 1))
                left = np.random.randint(0, max(max_x, 1))
                top = top - top % 2
                left = left - left % 2

                raw = raw  # already cropped above — redo properly:
                # Re-do with explicit positions
                raw_full = _load_array(raw_path)
                raw = raw_full[top : top + crop_raw, left : left + crop_raw]
                rgb_top, rgb_left = top * 2, left * 2
                rgb = rgb[rgb_top : rgb_top + crop_rgb, rgb_left : rgb_left + crop_rgb]
            else:
                # Same spatial size — crop both at same position
                min_h = min(rh, gh)
                min_w = min(rw, gw)
                min_h = min_h - min_h % 2
                min_w = min_w - min_w % 2
                raw = raw[:min_h, :min_w]
                rgb = rgb[:min_h, :min_w]

                cs = min(self.crop_size, min_h, min_w)
                raw, rgb = random_crop_pair(raw, rgb, cs)

            if self.augment:
                raw, rgb = augment_pair(raw, rgb)

            raw_tensor = _numpy_hwc_to_tensor(raw)
            rgb_tensor = _numpy_hwc_to_tensor(rgb)

        else:
            raise ValueError(
                f"Unexpected RAW shape {raw.shape} from {raw_path}. "
                "Expected (H, W) or (H, W, 4)."
            )

        metadata: dict[str, Any] = {
            "raw_file": os.path.basename(raw_path),
            "rgb_file": os.path.basename(rgb_path),
            "pattern": self.pattern,
        }

        return raw_tensor, rgb_tensor, metadata

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_pairs={len(self)}, "
            f"crop={self.crop_size}, "
            f"pattern='{self.pattern}')"
        )
