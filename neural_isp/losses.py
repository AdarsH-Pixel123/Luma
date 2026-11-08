"""Composite loss functions for neural ISP training.

Provides differentiable loss functions tailored for image-to-image
translation in the ISP domain:

- :class:`SSIMLoss` — structural similarity loss (1 - SSIM)
- :class:`ColorLoss` — approximate ΔE color loss via differentiable sRGB→Lab
- :class:`ISPLoss` — weighted combination of L1 + SSIM + Color losses
"""

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fspecial_gauss_1d(size: int, sigma: float) -> torch.Tensor:
    """Create a 1-D Gaussian kernel.

    Args:
        size: Kernel size (should be odd).
        sigma: Standard deviation of the Gaussian.

    Returns:
        Normalized 1-D Gaussian kernel of shape ``(1, 1, size)``.
    """
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return g.unsqueeze(0).unsqueeze(0)  # (1, 1, size)


class SSIMLoss(nn.Module):
    """Differentiable SSIM loss. Returns ``1 - SSIM``.

    Uses a Gaussian window for local statistics computation and operates
    per-channel before averaging.

    Args:
        window_size: Size of the Gaussian window.
        channel: Number of image channels.
        sigma: Standard deviation for the Gaussian window.
    """

    def __init__(
        self,
        window_size: int = 11,
        channel: int = 3,
        sigma: float = 1.5,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.channel = channel

        # Create 2-D Gaussian window from outer product of 1-D kernels
        _1d = _fspecial_gauss_1d(window_size, sigma)  # (1, 1, size)
        _2d = _1d.transpose(-1, -2) @ _1d  # (1, 1, size, size) via matmul-broadcast
        # Explicitly reshape for clarity
        _2d = _2d.view(1, 1, window_size, window_size)
        window = _2d.expand(channel, 1, window_size, window_size).contiguous()
        self.register_buffer("window", window)

        # SSIM constants
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss.

        Args:
            pred: Predicted image ``(B, C, H, W)`` in [0, 1].
            target: Ground truth image ``(B, C, H, W)`` in [0, 1].

        Returns:
            Scalar loss value ``1 - mean(SSIM)``.
        """
        channel = pred.shape[1]
        window = self.window
        if channel != self.channel:
            # Rebuild window for different channel count
            _1d = _fspecial_gauss_1d(self.window_size, 1.5).to(pred.device)
            _2d = (_1d.transpose(-1, -2) @ _1d).view(
                1, 1, self.window_size, self.window_size
            )
            window = _2d.expand(channel, 1, self.window_size, self.window_size).contiguous()

        window = window.to(pred.device, dtype=pred.dtype)
        pad = self.window_size // 2

        # Compute local statistics
        mu1 = F.conv2d(pred, window, padding=pad, groups=channel)
        mu2 = F.conv2d(target, window, padding=pad, groups=channel)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=channel) - mu2_sq
        sigma12 = F.conv2d(pred * target, window, padding=pad, groups=channel) - mu1_mu2

        # Clamp variances to avoid numerical issues
        sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0.0)

        # SSIM formula
        numerator = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        denominator = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        ssim_map = numerator / denominator

        return 1.0 - ssim_map.mean()


class ColorLoss(nn.Module):
    """Approximate Delta E color loss in sRGB space.

    Converts images to an approximate CIE Lab-like colour space using
    a differentiable sRGB → XYZ → Lab pipeline and computes the
    Euclidean distance (ΔE) between predicted and target colours.

    The sRGB → linear conversion uses a simplified gamma of 2.2 instead
    of the full piecewise sRGB transfer function, which is sufficient
    for loss computation and avoids discontinuous gradients.
    """

    def __init__(self) -> None:
        super().__init__()
        # sRGB to XYZ matrix (D65 illuminant)
        # Shape: (3, 3) — multiply with (B, 3, N) reshaped pixels
        rgb2xyz = torch.tensor(
            [
                [0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("rgb2xyz", rgb2xyz)

        # D65 white point
        self.register_buffer(
            "white_point",
            torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float32),
        )

    @staticmethod
    def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
        """Approximate sRGB gamma decoding (gamma 2.2).

        Args:
            x: sRGB values in [0, 1].

        Returns:
            Linear-light values.
        """
        return torch.pow(torch.clamp(x, min=1e-6), 2.2)

    def _linear_to_lab(self, xyz: torch.Tensor) -> torch.Tensor:
        """Convert XYZ to CIE Lab (differentiable approximation).

        Args:
            xyz: XYZ tensor ``(B, 3, H, W)``.

        Returns:
            Lab tensor ``(B, 3, H, W)``.
        """
        # Normalize by white point
        wp = self.white_point.view(1, 3, 1, 1)
        xyz = xyz / wp

        # f(t) with cube-root approximation (smooth)
        delta = 6.0 / 29.0
        delta_sq = delta ** 2
        delta_cb = delta ** 3

        mask = xyz > delta_cb
        f = torch.where(
            mask,
            torch.pow(torch.clamp(xyz, min=1e-10), 1.0 / 3.0),
            xyz / (3.0 * delta_sq) + 4.0 / 29.0,
        )

        fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)

        return torch.cat([L, a, b], dim=1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute approximate ΔE colour loss.

        Args:
            pred: Predicted sRGB image ``(B, C, H, W)`` in [0, 1].
            target: Ground truth sRGB image ``(B, C, H, W)`` in [0, 1].

        Returns:
            Mean ΔE across all pixels (scalar).
        """
        # sRGB -> linear
        pred_lin = self._srgb_to_linear(pred)
        target_lin = self._srgb_to_linear(target)

        # linear RGB -> XYZ  (einsum: 'ij, bjhw -> bihw')
        pred_xyz = torch.einsum("ij,bjhw->bihw", self.rgb2xyz, pred_lin)
        target_xyz = torch.einsum("ij,bjhw->bihw", self.rgb2xyz, target_lin)

        # XYZ -> Lab
        pred_lab = self._linear_to_lab(pred_xyz)
        target_lab = self._linear_to_lab(target_xyz)

        # ΔE = Euclidean distance in Lab space
        delta_e = torch.sqrt(
            torch.sum((pred_lab - target_lab) ** 2, dim=1, keepdim=True) + 1e-8
        )
        return delta_e.mean()


class ISPLoss(nn.Module):
    """Combined ISP training loss.

    Computes a weighted sum of multiple loss components:

    .. math::

        L_{\\text{total}} = \\alpha \\cdot L_1
                           + \\beta  \\cdot (1 - \\text{SSIM})
                           + \\gamma \\cdot \\Delta E

    Optionally includes LPIPS perceptual loss if the ``lpips`` package
    is installed and ``use_lpips=True``.

    Args:
        alpha: Weight for L1 loss.
        beta: Weight for SSIM loss (1 - SSIM).
        gamma: Weight for color (ΔE) loss.
        use_lpips: Whether to include LPIPS loss.
        lpips_weight: Weight for LPIPS loss (if enabled).
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 0.5,
        gamma: float = 0.1,
        use_lpips: bool = False,
        lpips_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lpips_weight = lpips_weight

        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss(window_size=11, channel=3)
        self.color_loss = ColorLoss()

        self.lpips_fn: Optional[nn.Module] = None
        if use_lpips:
            try:
                import lpips  # type: ignore[import-untyped]
                self.lpips_fn = lpips.LPIPS(net="vgg", verbose=False)
                # Freeze LPIPS parameters
                for p in self.lpips_fn.parameters():
                    p.requires_grad = False
            except ImportError:
                import warnings
                warnings.warn(
                    "lpips package not found. Install with `pip install lpips`. "
                    "Falling back to ISPLoss without LPIPS.",
                    stacklevel=2,
                )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute combined loss.

        Args:
            pred: Predicted image ``(B, 3, H, W)`` in [0, 1].
            target: Ground truth image ``(B, 3, H, W)`` in [0, 1].

        Returns:
            Dictionary with keys ``'total'``, ``'l1'``, ``'ssim'``,
            ``'color'``, and optionally ``'lpips'``.
        """
        l1 = self.l1_loss(pred, target)
        ssim = self.ssim_loss(pred, target)
        color = self.color_loss(pred, target)

        total = self.alpha * l1 + self.beta * ssim + self.gamma * color

        result: Dict[str, torch.Tensor] = {
            "l1": l1,
            "ssim": ssim,
            "color": color,
        }

        if self.lpips_fn is not None:
            # LPIPS expects images in [-1, 1]
            pred_scaled = pred * 2.0 - 1.0
            target_scaled = target * 2.0 - 1.0
            lpips_val = self.lpips_fn(pred_scaled, target_scaled).mean()
            total = total + self.lpips_weight * lpips_val
            result["lpips"] = lpips_val

        result["total"] = total
        return result


if __name__ == "__main__":
    # Quick smoke test
    pred = torch.rand(2, 3, 64, 64)
    target = torch.rand(2, 3, 64, 64)

    print("--- SSIMLoss ---")
    ssim_loss = SSIMLoss()
    print(f"  SSIM loss: {ssim_loss(pred, target).item():.4f}")

    print("--- ColorLoss ---")
    color_loss = ColorLoss()
    print(f"  Color loss: {color_loss(pred, target).item():.4f}")

    print("--- ISPLoss ---")
    isp_loss = ISPLoss(alpha=1.0, beta=0.5, gamma=0.1)
    losses = isp_loss(pred, target)
    for k, v in losses.items():
        print(f"  {k}: {v.item():.4f}")

    # Gradient check
    pred_g = torch.rand(1, 3, 32, 32, requires_grad=True)
    target_g = torch.rand(1, 3, 32, 32)
    loss = isp_loss(pred_g, target_g)["total"]
    loss.backward()
    assert pred_g.grad is not None, "No gradient computed!"
    print("\nGradient check passed!")
