"""FlowClean training script.

  1. Compute complex STFTs of noisy/clean pairs -> 2-channel tensors
  2. Sample Gaussian base noise Z
  3. Construct linear interpolation path z_t = (1-t)*Z + t*Y
  4. Predict velocity v_theta(z_t, t | X)
  5. Flow matching loss: ||v_theta - (Y - Z)||^2
  6. Auxiliary multi-resolution STFT loss in waveform space

Usage:
    python train.py --config configs/default.yaml
"""

import argparse
import math
import random
import time
from pathlib import Path


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb

from flowclean.config import FlowCleanConfig
from flowclean.models import FlowCleanUNet
from flowclean.data import VoiceBankDEMAND
from flowclean.losses import MultiResolutionSTFTLoss
from flowclean.utils.stft import stft, istft


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_scheduler(optimizer, cfg: FlowCleanConfig, steps_per_epoch: int):
    """Build LR scheduler with optional warmup."""
    sched = cfg.training.scheduler
    warmup_steps = sched.warmup_epochs * steps_per_epoch
    total_steps = cfg.training.epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(sched.min_lr / optimizer.defaults["lr"], 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    mr_stft_loss: MultiResolutionSTFTLoss,
    cfg: FlowCleanConfig,
    device: torch.device,
    epoch: int,
    global_step: int,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    n_batches = 0
    stft_kwargs = cfg.stft.to_dict()

    for step, batch in enumerate(dataloader):
        clean = batch["clean"].to(device)   # (B, T)
        noisy = batch["noisy"].to(device)   # (B, T)
        B = clean.shape[0]

        # Step 1: Compute complex STFTs -> (B, 2, F, T')
        Y = stft(clean, **stft_kwargs)
        X = stft(noisy, **stft_kwargs)

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
        lam = cfg.loss.lambda_mr_stft
        if lam > 0:
            # Reconstruct waveforms from predicted spectrograms
            # Approximate: use z_t + v_pred as one-step estimate of Y_hat
            Y_hat = z_t + (1 - t_expanded) * v_pred
            y_hat = istft(Y_hat, **stft_kwargs, length=clean.shape[-1])
            loss_mr = mr_stft_loss(y_hat, clean)

        loss = loss_fm + lam * loss_mr

        optimizer.zero_grad()
        loss.backward()
        if cfg.training.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        global_step += 1

        if (step + 1) % cfg.training.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [Epoch {epoch+1} | Step {step+1}/{len(dataloader)}] "
                f"loss={loss.item():.4f} (fm={loss_fm.item():.4f}, mr={loss_mr.item():.4f}) "
                f"lr={lr:.2e}"
            )
            if cfg.wandb.use_wandb:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/loss_fm": loss_fm.item(),
                    "train/loss_mr": loss_mr.item(),
                    "train/lr": lr,
                }, step=global_step)

    return total_loss / max(n_batches, 1), global_step


def main():
    parser = argparse.ArgumentParser(description="FlowClean Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    cfg = FlowCleanConfig.from_yaml(args.config)

    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Wandb (optional)
    if cfg.wandb.use_wandb:
        wandb.init(
            project=cfg.wandb.project,
            config={
                "data": cfg.data.__dict__,
                "stft": cfg.stft.__dict__,
                "model": cfg.model.__dict__,
                "loss": {"lambda_mr_stft": cfg.loss.lambda_mr_stft},
                "training": {k: v for k, v in cfg.training.__dict__.items() if k != "scheduler"},
            },
        )

    # Dataset (loads from HuggingFace: JacobLinCool/VoiceBank-DEMAND-16k)
    train_ds = VoiceBankDEMAND(
        split="train",
        segment_length=cfg.data.segment_length,
        sample_rate=cfg.data.sample_rate,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Training samples: {len(train_ds)}")

    # Model
    model = FlowCleanUNet(
        base_channels=cfg.model.base_channels,
        num_levels=cfg.model.num_levels,
        time_dim=cfg.model.time_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # Loss
    mr_stft_loss = MultiResolutionSTFTLoss(
        fft_sizes=cfg.loss.mr_stft.fft_sizes,
        hop_sizes=cfg.loss.mr_stft.hop_sizes,
        win_sizes=cfg.loss.mr_stft.win_sizes,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )

    # Scheduler
    scheduler = get_scheduler(optimizer, cfg, len(train_loader))

    # Checkpoint directory
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_loss = float("inf")
    global_step = 0
    for epoch in range(cfg.training.epochs):
        t0 = time.time()
        avg_loss, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            mr_stft_loss=mr_stft_loss,
            cfg=cfg,
            device=device,
            epoch=epoch,
            global_step=global_step,
        )
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{cfg.training.epochs} — avg_loss={avg_loss:.4f} — {elapsed:.1f}s")

        if cfg.wandb.use_wandb:
            wandb.log({"epoch/avg_loss": avg_loss, "epoch": epoch + 1}, step=global_step)

        # Save checkpoint
        if (epoch + 1) % cfg.training.save_every == 0 or avg_loss < best_loss:
            ckpt_path = ckpt_dir / f"flowclean_epoch{epoch+1}.pt"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "config": args.config,
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
                        "config": args.config,
                    },
                    best_path,
                )
                print(f"  -> New best model saved (loss={best_loss:.4f})")

    if cfg.wandb.use_wandb:
        wandb.finish()

    print("Training complete!")


if __name__ == "__main__":
    main()
