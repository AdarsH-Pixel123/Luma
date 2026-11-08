"""Lightweight Mobile U-Net for RAW-to-RGB ISP processing.

A compact U-Net architecture designed for mobile ISP inference, using
depthwise-separable convolutions throughout for efficiency. Processes
4-channel packed Bayer input and produces 3-channel linear RGB output
via PixelShuffle upsampling.

Target parameter budget: ~500K-1.5M parameters.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise.

    Factorizes a standard convolution into a depthwise convolution
    (per-channel spatial filtering) followed by a pointwise 1x1
    convolution (cross-channel mixing). This reduces parameter count
    from ``in_ch * out_ch * k^2`` to ``in_ch * k^2 + in_ch * out_ch``.

    Args:
        in_ch: Number of input channels.
        out_ch: Number of output channels.
        kernel_size: Spatial size of the depthwise kernel.
        stride: Stride of the depthwise convolution.
        padding: Padding for the depthwise convolution.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape ``(B, in_ch, H, W)``.

        Returns:
            Output tensor of shape ``(B, out_ch, H', W')`` where spatial
            dimensions depend on stride/padding.
        """
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    """Residual block with two depthwise-separable convolutions.

    Applies two DSConv layers with a skip connection. Identity mapping
    when input and output channels match.

    Args:
        channels: Number of input and output channels (must be equal
            for the residual connection).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(channels, channels, 3, 1, 1)
        self.conv2_depthwise = nn.Conv2d(
            channels, channels, kernel_size=3, stride=1, padding=1,
            groups=channels, bias=False,
        )
        self.conv2_pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Output tensor of shape ``(B, C, H, W)``.
        """
        residual = x
        out = self.conv1(x)
        out = self.conv2_depthwise(out)
        out = self.conv2_pointwise(out)
        out = self.bn2(out)
        out = F.relu(out + residual, inplace=True)
        return out


class MobileUNet(nn.Module):
    """Lightweight U-Net for RAW-to-RGB processing.

    Architecture:
        - **Encoder**: 3 levels with DepthwiseSeparableConv + MaxPool

          - Level 0: 4  → 32 channels
          - Level 1: 32 → 64 channels
          - Level 2: 64 → 128 channels

        - **Bottleneck**: 128 → 128 channels with ResBlock

        - **Decoder**: 3 levels with skip connections + DepthwiseSeparableConv

          - Level 2: 128 + 128 → 64 channels
          - Level 1: 64 + 64 → 32 channels
          - Level 0: 32 + 32 → 32 channels

        - **Output head**: 32 → 12 channels (for PixelShuffle 2× → 3 RGB)

    The PixelShuffle at the end converts ``(B, 12, H/2, W/2)`` →
    ``(B, 3, H, W)`` which elegantly handles the resolution doubling
    from packed Bayer to full RGB.

    Args:
        in_channels: Number of input channels (4 for packed Bayer).
        base_filters: Number of filters in the first encoder level.
    """

    def __init__(self, in_channels: int = 4, base_filters: int = 32) -> None:
        super().__init__()
        bf = base_filters  # 32

        # --- Encoder ---
        # Level 0: in_channels -> bf
        self.enc0 = nn.Sequential(
            DepthwiseSeparableConv(in_channels, bf, 3, 1, 1),
            DepthwiseSeparableConv(bf, bf, 3, 1, 1),
        )
        self.pool0 = nn.MaxPool2d(2, 2)

        # Level 1: bf -> 2*bf
        self.enc1 = nn.Sequential(
            DepthwiseSeparableConv(bf, bf * 2, 3, 1, 1),
            DepthwiseSeparableConv(bf * 2, bf * 2, 3, 1, 1),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        # Level 2: 2*bf -> 4*bf
        self.enc2 = nn.Sequential(
            DepthwiseSeparableConv(bf * 2, bf * 4, 3, 1, 1),
            DepthwiseSeparableConv(bf * 4, bf * 4, 3, 1, 1),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        # --- Bottleneck ---
        self.bottleneck = nn.Sequential(
            ResBlock(bf * 4),
            ResBlock(bf * 4),
        )

        # --- Decoder ---
        # Up 2: (4*bf concat 4*bf) -> 2*bf
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = nn.Sequential(
            DepthwiseSeparableConv(bf * 4 + bf * 4, bf * 2, 3, 1, 1),
            DepthwiseSeparableConv(bf * 2, bf * 2, 3, 1, 1),
        )

        # Up 1: (2*bf concat 2*bf) -> bf
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            DepthwiseSeparableConv(bf * 2 + bf * 2, bf, 3, 1, 1),
            DepthwiseSeparableConv(bf, bf, 3, 1, 1),
        )

        # Up 0: (bf concat bf) -> bf
        self.up0 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec0 = nn.Sequential(
            DepthwiseSeparableConv(bf + bf, bf, 3, 1, 1),
            DepthwiseSeparableConv(bf, bf, 3, 1, 1),
        )

        # --- Output head ---
        # 32 -> 12 channels, then PixelShuffle(2) -> 3 channels at 2× resolution
        self.out_conv = nn.Conv2d(bf, 12, kernel_size=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Kaiming initialization to all convolutional layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """Pad spatial dimensions to be divisible by ``multiple``.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.
            multiple: The divisor for spatial dimensions.

        Returns:
            Padded tensor and tuple ``(pad_h, pad_w)`` for later cropping.
        """
        _, _, h, w = x.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (pad_h, pad_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: packed Bayer → linear RGB.

        Args:
            x: Packed Bayer tensor of shape ``(B, 4, H/2, W/2)``.

        Returns:
            Linear RGB tensor of shape ``(B, 3, H, W)``.
        """
        # Pad input so encoder pooling doesn't lose pixels
        x, (pad_h, pad_w) = self._pad_to_multiple(x, multiple=8)

        # Encoder
        s0 = self.enc0(x)           # (B, 32, H/2, W/2)
        s1 = self.enc1(self.pool0(s0))  # (B, 64, H/4, W/4)
        s2 = self.enc2(self.pool1(s1))  # (B, 128, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(self.pool2(s2))  # (B, 128, H/16, W/16)

        # Decoder with skip connections
        d2 = self.up2(b)  # (B, 128, H/8, W/8)
        d2 = self._match_and_concat(d2, s2)
        d2 = self.dec2(d2)  # (B, 64, H/8, W/8)

        d1 = self.up1(d2)  # (B, 64, H/4, W/4)
        d1 = self._match_and_concat(d1, s1)
        d1 = self.dec1(d1)  # (B, 32, H/4, W/4)

        d0 = self.up0(d1)  # (B, 32, H/2, W/2)
        d0 = self._match_and_concat(d0, s0)
        d0 = self.dec0(d0)  # (B, 32, H/2, W/2)

        # Output head
        out = self.out_conv(d0)       # (B, 12, H/2, W/2)
        out = self.pixel_shuffle(out)  # (B, 3, H, W)

        # Remove padding (in output space, padding is 2× the input padding)
        if pad_h > 0 or pad_w > 0:
            out_h = out.shape[2] - pad_h * 2
            out_w = out.shape[3] - pad_w * 2
            out = out[:, :, :out_h, :out_w]

        # Clamp to valid range
        out = torch.clamp(out, 0.0, 1.0)
        return out

    @staticmethod
    def _match_and_concat(
        upsampled: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        """Crop upsampled feature map to match skip connection size and concatenate.

        Args:
            upsampled: Decoder feature map (may be slightly larger).
            skip: Encoder skip connection feature map.

        Returns:
            Concatenated tensor along channel dimension.
        """
        _, _, h_s, w_s = skip.shape
        upsampled = upsampled[:, :, :h_s, :w_s]
        return torch.cat([upsampled, skip], dim=1)

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"{base}\n# Trainable parameters: {self.count_parameters():,}"


if __name__ == "__main__":
    model = MobileUNet(in_channels=4, base_filters=32)
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")

    # Test with a dummy input
    dummy = torch.randn(1, 4, 128, 128)
    out = model(dummy)
    print(f"Input shape:  {dummy.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (1, 3, 256, 256), f"Expected (1,3,256,256), got {out.shape}"

    # Test with non-power-of-2 dimensions
    dummy2 = torch.randn(1, 4, 100, 150)
    out2 = model(dummy2)
    print(f"Input shape:  {dummy2.shape}")
    print(f"Output shape: {out2.shape}")
    assert out2.shape == (1, 3, 200, 300), f"Expected (1,3,200,300), got {out2.shape}"
    print("All tests passed!")
