"""FlowClean training script.

Implements the full training pipeline from the paper:
  1. Compute complex STFTs of noisy/clean pairs -> 2-channel tensors
  2. Sample Gaussian base noise Z
  3. Construct linear interpolation path z_t = (1-t)*Z + t*Y
  4. Predict velocity v_theta(z_t, t | X)
  5. Flow matching loss: ||v_theta - (Y - Z)||^2
  6. Auxiliary multi-resolution STFT loss in waveform space

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --data.root /path/to/data
"""

import argparse
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml

from flowclean.models import FlowCleanUNet
from flowclean.data import VoiceBankDEMAND
from flowclean.losses import MultiResolutionSTFTLoss
from flowclean.utils.stft import stft, istft


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def override_config(cfg: dict, overrides: list[str]) -> dict:
    """Apply dot-separated overrides like --data.root /path."""
    for i in range(0, len(overrides), 2):
        keys = overrides[i].lstrip("-").split(".")
        val = overrides[i + 1]
        d = cfg
        for k in keys[:-1]:
            d = d[k]
        # Auto-cast
        old = d.get(keys[-1])
        if isinstance(old, int):
            val = int(val)
        elif isinstance(old, float):
            val = float(val)
        elif isinstance(old, bool):
            val = val.lower() in ("true", "1", "yes")
        d[keys[-1]] = val
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_scheduler(optimizer, cfg_sched: dict, epochs: int, steps_per_epoch: int):
    """Build LR scheduler with optional warmup."""
    warmup_steps = cfg_sched.get("warmup_epochs", 0) * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    min_lr = cfg_sched.get("min_lr", 1e-6)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        import math
        return max(min_lr / optimizer.defaults["lr"], 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    mr_stft_loss: MultiResolutionSTFTLoss,
    stft_cfg: dict,
    loss_cfg: dict,
    device: torch.device,
    epoch: int,
    log_every: int,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for step, batch in enumerate(dataloader):
        clean = batch["clean"].to(device)   # (B, T)
        noisy = batch["noisy"].to(device)   # (B, T)
        B = clean.shape[0]

        # Step 1: Compute complex STFTs -> (B, 2, F, T')
        Y = stft(clean, **stft_cfg)
        X = stft(noisy, **stft_cfg)

        # Step 2: Sample base noise Z ~ N(0, I)
        Z = torch.randn_like(Y)

        # Step 3: Sample time t ~ U(0, 1) and construct path
        t = torch.rand(B, device=device)
        # z_t = (1 - t) * Z + t * Y   shape broadcast: t is (B,) -> (B,1,1,1)
        t_expanded = t[:, None, None, None]
        z_t = (1 - t_expanded) * Z + t_expanded * Y

        # Step 4: Predict velocity v_theta(z_t, t | X)
        v_pred = model(z_t, t, X)

        # Step 5: Flow matching loss: ||v_theta - (Y - Z)||^2
        target_velocity = Y - Z
        loss_fm = F.mse_loss(v_pred, target_velocity)

        # Step 6: Auxiliary multi-resolution STFT loss in waveform space
        loss_mr = torch.tensor(0.0, device=device)
        lam = loss_cfg.get("lambda_mr_stft", 0.0)
        if lam > 0:
            # Reconstruct waveforms from predicted spectrograms
            # Approximate: use z_t + v_pred as one-step estimate of Y_hat
            Y_hat = z_t + (1 - t_expanded) * v_pred
            y_hat = istft(Y_hat, **stft_cfg, length=clean.shape[-1])
            loss_mr = mr_stft_loss(y_hat, clean)

        loss = loss_fm + lam * loss_mr

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        if (step + 1) % log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [Epoch {epoch+1} | Step {step+1}/{len(dataloader)}] "
                f"loss={loss.item():.4f} (fm={loss_fm.item():.4f}, mr={loss_mr.item():.4f}) "
                f"lr={lr:.2e}"
            )

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="FlowClean Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args, overrides = parser.parse_known_args()

    cfg = load_config(args.config)
    if overrides:
        cfg = override_config(cfg, overrides)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    train_ds = VoiceBankDEMAND(
        root=cfg["data"]["root"],
        split="train",
        segment_length=cfg["data"]["segment_length"],
        sample_rate=cfg["data"]["sample_rate"],
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    print(f"Training samples: {len(train_ds)}")

    # Model
    model = FlowCleanUNet(
        base_channels=cfg["model"]["base_channels"],
        num_levels=cfg["model"]["num_levels"],
        time_dim=cfg["model"]["time_dim"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Loss
    mr_stft_loss = MultiResolutionSTFTLoss(
        fft_sizes=cfg["loss"]["mr_stft"]["fft_sizes"],
        hop_sizes=cfg["loss"]["mr_stft"]["hop_sizes"],
        win_sizes=cfg["loss"]["mr_stft"]["win_sizes"],
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    # Scheduler
    scheduler = get_scheduler(
        optimizer,
        cfg["training"]["scheduler"],
        cfg["training"]["epochs"],
        len(train_loader),
    )

    # Checkpoint directory
    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # STFT config dict
    stft_cfg = {
        "n_fft": cfg["stft"]["n_fft"],
        "hop_length": cfg["stft"]["hop_length"],
        "win_length": cfg["stft"]["win_length"],
    }

    # Training loop
    best_loss = float("inf")
    for epoch in range(cfg["training"]["epochs"]):
        t0 = time.time()
        avg_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            mr_stft_loss=mr_stft_loss,
            stft_cfg=stft_cfg,
            loss_cfg=cfg["loss"],
            device=device,
            epoch=epoch,
            log_every=cfg["training"]["log_every"],
            grad_clip=cfg["training"]["grad_clip"],
        )
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} — avg_loss={avg_loss:.4f} — {elapsed:.1f}s")

        # Save checkpoint
        if (epoch + 1) % cfg["training"]["save_every"] == 0 or avg_loss < best_loss:
            ckpt_path = ckpt_dir / f"flowclean_epoch{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": cfg,
                },
                ckpt_path,
            )
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_path = ckpt_dir / "flowclean_best.pt"
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "config": cfg,
                    },
                    best_path,
                )
                print(f"  -> New best model saved (loss={best_loss:.4f})")

    print("Training complete!")


if __name__ == "__main__":
    main()
