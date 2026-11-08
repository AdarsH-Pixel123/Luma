"""Simplified multi-scale PyNET variant for RAW-to-RGB ISP processing.

A lightweight version of the PyNET architecture that processes at multiple
resolutions (3 levels) for multi-scale supervision. Uses depthwise-separable
convolutions to keep the parameter budget around 1-2M.

Reference:
    Ignatov et al., "Replacing Mobile Camera ISP with a Single Deep Learning
    Model", CVPRW 2020.
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DSConvBlock(nn.Module):
    """Depthwise-separable conv block with BN and LeakyReLU.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        kernel_size: Spatial kernel size.
        stride: Convolution stride.
        padding: Convolution padding.
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
        self.block = nn.Sequential(
            # Depthwise
            nn.Conv2d(
                in_ch, in_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=in_ch,
                bias=False,
            ),
            # Pointwise
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _ResidualDSBlock(nn.Module):
    """Residual block using depthwise-separable convolutions.

    Args:
        channels: Number of channels (preserved through the block).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = _DSConvBlock(channels, channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                channels, channels,
                kernel_size=3, padding=1,
                groups=channels, bias=False,
            ),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        return self.act(out + x)


class _LevelBlock(nn.Module):
    """Processing block for a single PyNET level.

    Consists of multiple residual DS blocks followed by an optional
    RGB output head for multi-scale supervision.

    Args:
        in_ch: Input channels.
        mid_ch: Internal channel width.
        num_blocks: Number of residual blocks.
        has_output: Whether to include an RGB output head.
    """

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        num_blocks: int = 2,
        has_output: bool = True,
    ) -> None:
        super().__init__()
        # Input projection
        self.input_proj = _DSConvBlock(in_ch, mid_ch)

        # Residual blocks
        blocks = [_ResidualDSBlock(mid_ch) for _ in range(num_blocks)]
        self.body = nn.Sequential(*blocks)

        # RGB output head (for multi-scale loss)
        self.has_output = has_output
        if has_output:
            self.output_head = nn.Conv2d(mid_ch, 3, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            Tuple of (features, rgb_output_or_None).
        """
        feat = self.input_proj(x)
        feat = self.body(feat)
        rgb = self.output_head(feat) if self.has_output else None
        return feat, rgb


class PyNETLite(nn.Module):
    """Simplified PyNET: multi-scale processing with 3 levels.

    Each level processes at a different resolution:

    - **Level 3** (lowest res, 1/4): Global context, color correction
    - **Level 2** (mid res, 1/2): Detail enhancement
    - **Level 1** (full res): Fine detail reconstruction

    Input: 4-channel packed Bayer ``(B, 4, H/2, W/2)``
    Output: 3-channel RGB ``(B, 3, H, W)``

    The architecture is kept lightweight (~1-2M parameters) by using
    depthwise-separable convolutions. Each level produces its own RGB
    output for multi-scale supervision during training.

    Args:
        in_channels: Number of input channels (4 for packed Bayer).
    """

    def __init__(self, in_channels: int = 4) -> None:
        super().__init__()

        # Channel widths per level
        ch1 = 32   # Level 1 (full res)
        ch2 = 64   # Level 2 (half res)
        ch3 = 96   # Level 3 (quarter res)

        # Downsampling for multi-scale input
        self.down2 = nn.MaxPool2d(2, 2)
        self.down4 = nn.MaxPool2d(4, 4)

        # --- Level 3: lowest resolution (global context) ---
        self.level3 = _LevelBlock(
            in_ch=in_channels, mid_ch=ch3, num_blocks=3, has_output=True,
        )

        # --- Level 2: mid resolution (detail enhancement) ---
        # Input: original features at 1/2 res + upsampled level3 features
        self.up3_to_2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.level2 = _LevelBlock(
            in_ch=in_channels + ch3, mid_ch=ch2, num_blocks=3, has_output=True,
        )

        # --- Level 1: full resolution (fine detail) ---
        # Input: original features + upsampled level2 features
        self.up2_to_1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.level1 = _LevelBlock(
            in_ch=in_channels + ch2, mid_ch=ch1, num_blocks=3, has_output=False,
        )

        # --- Final output head ---
        # PixelShuffle to go from (B, ch1, H/2, W/2) -> (B, 3, H, W)
        self.final_conv = nn.Conv2d(ch1, 12, kernel_size=1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Apply Kaiming initialization to all convolutional layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, multiple: int = 4) -> tuple:
        """Pad spatial dimensions to be divisible by ``multiple``.

        Args:
            x: Input tensor ``(B, C, H, W)``.
            multiple: Required divisor.

        Returns:
            Padded tensor and ``(pad_h, pad_w)`` for later cropping.
        """
        _, _, h, w = x.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return x, (pad_h, pad_w)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with multi-scale outputs.

        Args:
            x: Packed Bayer tensor ``(B, 4, H/2, W/2)``.

        Returns:
            Dictionary with keys:

            - ``'output'``: Final RGB at full resolution ``(B, 3, H, W)``.
            - ``'level2'``: RGB at half resolution (for training loss).
            - ``'level3'``: RGB at quarter resolution (for training loss).
        """
        x, (pad_h, pad_w) = self._pad_to_multiple(x, multiple=4)
        orig_h, orig_w = x.shape[2], x.shape[3]

        # Multi-scale inputs
        x_half = self.down2(x)    # (B, 4, H/4, W/4)
        x_quarter = self.down4(x)  # (B, 4, H/8, W/8)

        # Level 3: lowest resolution
        feat3, rgb3 = self.level3(x_quarter)  # feat3: (B, ch3, H/8, W/8)

        # Level 2: mid resolution
        feat3_up = self.up3_to_2(feat3)
        # Match spatial dims
        feat3_up = feat3_up[:, :, :x_half.shape[2], :x_half.shape[3]]
        level2_input = torch.cat([x_half, feat3_up], dim=1)
        feat2, rgb2 = self.level2(level2_input)  # feat2: (B, ch2, H/4, W/4)

        # Level 1: full resolution
        feat2_up = self.up2_to_1(feat2)
        feat2_up = feat2_up[:, :, :x.shape[2], :x.shape[3]]
        level1_input = torch.cat([x, feat2_up], dim=1)
        feat1, _ = self.level1(level1_input)  # feat1: (B, ch1, H/2, W/2)

        # Final output via PixelShuffle
        out = self.final_conv(feat1)       # (B, 12, H/2, W/2)
        out = self.pixel_shuffle(out)      # (B, 3, H, W)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            out_h = orig_h * 2 - pad_h * 2
            out_w = orig_w * 2 - pad_w * 2
            out = out[:, :, :out_h, :out_w]

        out = torch.clamp(out, 0.0, 1.0)

        result: Dict[str, torch.Tensor] = {"output": out}
        if rgb3 is not None:
            result["level3"] = torch.clamp(rgb3, 0.0, 1.0)
        if rgb2 is not None:
            result["level2"] = torch.clamp(rgb2, 0.0, 1.0)

        return result

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
    model = PyNETLite(in_channels=4)
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")

    # Test with a dummy input
    dummy = torch.randn(1, 4, 128, 128)
    result = model(dummy)
    print(f"Input shape:   {dummy.shape}")
    print(f"Output shape:  {result['output'].shape}")
    print(f"Level2 shape:  {result['level2'].shape}")
    print(f"Level3 shape:  {result['level3'].shape}")
    assert result["output"].shape == (1, 3, 256, 256)

    # Test with non-power-of-2 dimensions
    dummy2 = torch.randn(1, 4, 100, 150)
    result2 = model(dummy2)
    print(f"\nInput shape:   {dummy2.shape}")
    print(f"Output shape:  {result2['output'].shape}")
    assert result2["output"].shape == (1, 3, 200, 300)
    print("All tests passed!")
