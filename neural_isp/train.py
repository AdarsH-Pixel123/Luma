"""Training script for Neural ISP models.

Supports training both MobileUNet and PyNETLite on synthetic RAW data.
Uses the SyntheticRAWDataset from the datasets module to generate
(degraded_bayer, clean_rgb) pairs on the fly from a directory of RGB images.

Features:
    - Mixed precision training with ``torch.cuda.amp`` (if GPU available)
    - Cosine annealing LR scheduler with linear warmup
    - Periodic validation with PSNR / SSIM metrics
    - Checkpoint saving (best model + periodic every 10 epochs)
    - JSON training log
    - Support for both MobileUNet and PyNETLite architectures

Usage::

    python -m neural_isp.train \\
        --model mobile_unet \\
        --data_dir /path/to/train_images \\
        --val_dir /path/to/val_images \\
        --epochs 100 --batch_size 16 --lr 1e-3
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Allow running as ``python -m neural_isp.train`` or ``python neural_isp/train.py``
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_isp.mobile_unet import MobileUNet
from neural_isp.pynet_lite import PyNETLite
from neural_isp.losses import ISPLoss, SSIMLoss
from datasets import SyntheticRAWDataset


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR (dB) between predicted and target images.

    Args:
        pred: ``(B, C, H, W)`` tensor in [0, 1].
        target: ``(B, C, H, W)`` tensor in [0, 1].

    Returns:
        Mean PSNR across the batch.
    """
    mse = F.mse_loss(pred, target, reduction="mean").item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


@torch.no_grad()
def compute_ssim_metric(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute mean SSIM between predicted and target images.

    Args:
        pred: ``(B, C, H, W)`` tensor in [0, 1].
        target: ``(B, C, H, W)`` tensor in [0, 1].

    Returns:
        Mean SSIM (higher is better).
    """
    ssim_loss = SSIMLoss(window_size=11, channel=pred.shape[1])
    # SSIMLoss returns 1 - SSIM, so SSIM = 1 - loss
    device = pred.device
    ssim_loss = ssim_loss.to(device)
    return 1.0 - ssim_loss(pred, target).item()


# ---------------------------------------------------------------------------
# LR scheduler with warmup
# ---------------------------------------------------------------------------

class CosineAnnealingWithWarmup:
    """Cosine annealing scheduler with linear warmup.

    Args:
        optimizer: The optimizer to schedule.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total number of training epochs.
        min_lr: Minimum learning rate at the end of cosine decay.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_epoch = 0

    def step(self) -> None:
        """Advance the scheduler by one epoch."""
        self.current_epoch += 1
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if self.current_epoch <= self.warmup_epochs:
                # Linear warmup
                lr = base_lr * (self.current_epoch / max(1, self.warmup_epochs))
            else:
                # Cosine annealing
                progress = (self.current_epoch - self.warmup_epochs) / max(
                    1, self.total_epochs - self.warmup_epochs
                )
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1.0 + math.cos(math.pi * progress)
                )
            pg["lr"] = lr

    def get_lr(self) -> float:
        """Return current learning rate of the first param group."""
        return self.optimizer.param_groups[0]["lr"]


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: ISPLoss,
    device: torch.device,
    is_pynet: bool = False,
) -> Dict[str, float]:
    """Run validation and compute metrics.

    Args:
        model: The neural ISP model.
        val_loader: Validation data loader.
        criterion: Loss function.
        device: Compute device.
        is_pynet: Whether the model is PyNETLite (dict output).

    Returns:
        Dictionary with ``'loss'``, ``'psnr'``, ``'ssim'`` keys.
    """
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for batch in val_loader:
        bayer, rgb = batch[0], batch[1]
        bayer = bayer.to(device)
        rgb = rgb.to(device)

        output = model(bayer)
        if is_pynet:
            pred = output["output"]
        else:
            pred = output

        losses = criterion(pred, rgb)
        batch_size = bayer.shape[0]
        total_loss += losses["total"].item() * batch_size
        total_psnr += compute_psnr(pred, rgb) * batch_size
        total_ssim += compute_ssim_metric(pred, rgb) * batch_size
        count += batch_size

    model.train()
    return {
        "loss": total_loss / max(count, 1),
        "psnr": total_psnr / max(count, 1),
        "ssim": total_ssim / max(count, 1),
    }


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Main training function.

    Args:
        args: Parsed command-line arguments.
    """
    # ---- Device setup ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"[INFO] Using device: {device} | AMP: {use_amp}")

    # ---- Save directory ----
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- Dataset & DataLoader ----
    noise_map = {"low": (0.001, 0.005), "medium": (0.005, 0.02), "high": (0.02, 0.06)}
    noise_range = noise_map.get(args.noise_level, (0.005, 0.02))

    train_dataset = SyntheticRAWDataset(
        image_dir=args.data_dir,
        crop_size=args.crop_size,
        noise_level=args.noise_level,
        augment=True,
    )
    if getattr(args, "max_samples", None) and args.max_samples > 0:
        train_dataset.image_paths = train_dataset.image_paths[:args.max_samples]
    num_workers = 0 if os.name == "nt" else min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    print(f"[INFO] Training samples: {len(train_dataset)}")

    val_loader: Optional[DataLoader] = None
    if args.val_dir is not None:
        val_dataset = SyntheticRAWDataset(
            image_dir=args.val_dir,
            crop_size=args.crop_size,
            noise_level=args.noise_level,
            augment=False,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
        )
        print(f"[INFO] Validation samples: {len(val_dataset)}")

    # ---- Model ----
    is_pynet = args.model == "pynet_lite"
    if args.model == "mobile_unet":
        model = MobileUNet(in_channels=4, base_filters=32)
    elif args.model == "pynet_lite":
        model = PyNETLite(in_channels=4)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model: {args.model} | Parameters: {num_params:,}")

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4,
    )

    # ---- Scheduler ----
    warmup_epochs = max(1, args.epochs // 20)
    scheduler = CosineAnnealingWithWarmup(
        optimizer, warmup_epochs=warmup_epochs, total_epochs=args.epochs,
    )

    # ---- Loss ----
    criterion = ISPLoss(alpha=1.0, beta=0.5, gamma=0.1).to(device)

    # ---- Mixed precision scaler ----
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ---- Resume from checkpoint ----
    start_epoch = 0
    best_val_psnr = 0.0
    training_log: list = []

    if args.resume is not None and os.path.isfile(args.resume):
        print(f"[INFO] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_psnr = ckpt.get("best_val_psnr", 0.0)
        if "scaler_state_dict" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        # Advance scheduler to the correct epoch
        for _ in range(start_epoch):
            scheduler.step()
        print(f"[INFO] Resumed at epoch {start_epoch}, best PSNR: {best_val_psnr:.2f}")

    # ---- Training loop ----
    print(f"[INFO] Starting training for {args.epochs} epochs "
          f"(warmup: {warmup_epochs})")
    print("-" * 72)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        running_loss = 0.0
        running_l1 = 0.0
        running_ssim = 0.0
        running_color = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            bayer, rgb = batch[0], batch[1]
            bayer = bayer.to(device, non_blocking=True)
            rgb = rgb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(bayer)
                if is_pynet:
                    pred = output["output"]
                else:
                    pred = output

                losses = criterion(pred, rgb)
                loss = losses["total"]

                # Multi-scale loss for PyNET
                if is_pynet and "level2" in output and "level3" in output:
                    # Downsample target for intermediate supervision
                    _, _, h2, w2 = output["level2"].shape
                    rgb_l2 = F.interpolate(
                        rgb, size=(h2, w2), mode="bilinear", align_corners=False,
                    )
                    _, _, h3, w3 = output["level3"].shape
                    rgb_l3 = F.interpolate(
                        rgb, size=(h3, w3), mode="bilinear", align_corners=False,
                    )
                    loss_l2 = F.l1_loss(output["level2"], rgb_l2)
                    loss_l3 = F.l1_loss(output["level3"], rgb_l3)
                    loss = loss + 0.5 * loss_l2 + 0.25 * loss_l3

            scaler.scale(loss).backward()
            # Gradient clipping
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += losses["total"].item()
            running_l1 += losses["l1"].item()
            running_ssim += losses["ssim"].item()
            running_color += losses["color"].item()
            num_batches += 1

        # Step scheduler
        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / max(num_batches, 1)
        avg_l1 = running_l1 / max(num_batches, 1)
        avg_ssim = running_ssim / max(num_batches, 1)
        avg_color = running_color / max(num_batches, 1)

        log_entry: Dict[str, Any] = {
            "epoch": epoch + 1,
            "lr": scheduler.get_lr(),
            "train_loss": avg_loss,
            "train_l1": avg_l1,
            "train_ssim": avg_ssim,
            "train_color": avg_color,
            "time_s": round(epoch_time, 1),
        }

        log_msg = (
            f"Epoch [{epoch + 1:3d}/{args.epochs}]  "
            f"Loss: {avg_loss:.4f}  L1: {avg_l1:.4f}  "
            f"SSIM: {avg_ssim:.4f}  Color: {avg_color:.4f}  "
            f"LR: {scheduler.get_lr():.2e}  "
            f"Time: {epoch_time:.1f}s"
        )

        # Validation
        if val_loader is not None and (epoch + 1) % 5 == 0:
            val_metrics = validate(model, val_loader, criterion, device, is_pynet)
            log_entry.update({
                "val_loss": val_metrics["loss"],
                "val_psnr": val_metrics["psnr"],
                "val_ssim": val_metrics["ssim"],
            })
            log_msg += (
                f"  | Val Loss: {val_metrics['loss']:.4f}  "
                f"PSNR: {val_metrics['psnr']:.2f}  "
                f"SSIM: {val_metrics['ssim']:.4f}"
            )

            # Save best model
            if val_metrics["psnr"] > best_val_psnr:
                best_val_psnr = val_metrics["psnr"]
                _save_checkpoint(
                    model, optimizer, scaler, epoch + 1,
                    best_val_psnr, save_dir / f"{args.model}_best.pth",
                    args,
                )
                log_msg += " ★"

        print(log_msg)
        training_log.append(log_entry)

        # Periodic checkpoint (every 10 epochs)
        if (epoch + 1) % 10 == 0:
            _save_checkpoint(
                model, optimizer, scaler, epoch + 1,
                best_val_psnr, save_dir / f"{args.model}_epoch{epoch + 1}.pth",
                args,
            )

    # Save final model
    _save_checkpoint(
        model, optimizer, scaler, args.epochs,
        best_val_psnr, save_dir / f"{args.model}_final.pth",
        args,
    )

    # Save training log
    log_path = save_dir / f"{args.model}_training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"\n[INFO] Training complete. Best validation PSNR: {best_val_psnr:.2f} dB")
    print(f"[INFO] Training log saved to {log_path}")


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val_psnr: float,
    path: Path,
    args: argparse.Namespace,
) -> None:
    """Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer state.
        scaler: AMP gradient scaler state.
        epoch: Current epoch number.
        best_val_psnr: Best validation PSNR so far.
        path: File path for the checkpoint.
        args: Training arguments for reproducibility.
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val_psnr": best_val_psnr,
            "args": vars(args),
        },
        path,
    )
    print(f"[CKPT] Saved checkpoint: {path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Neural ISP (MobileUNet or PyNETLite)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default="mobile_unet",
        choices=["mobile_unet", "pynet_lite"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Directory containing training RGB images.",
    )
    parser.add_argument(
        "--val_dir", type=str, default=None,
        help="Directory containing validation RGB images.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Training batch size.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--crop_size", type=int, default=256,
        help="Random crop size for training patches.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Optional maximum number of training images to use (useful for fast CPU training).",
    )
    parser.add_argument(
        "--noise_level", type=str, default="medium",
        choices=["low", "medium", "high"],
        help="Synthetic noise level for degradation.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="../checkpoints",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume training from.",
    )
    args = parser.parse_args()
    train(args)
