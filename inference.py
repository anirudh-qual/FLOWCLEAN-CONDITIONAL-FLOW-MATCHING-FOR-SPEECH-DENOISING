"""FlowClean inference / evaluation script.

Generates enhanced speech by solving the conditional probability flow ODE:
  dz_t/dt = v_theta(z_t, t | X)
using Euler or Heun solvers with K steps.

Optionally computes PESQ, STOI, and DNSMOS metrics on the test set.

Usage:
    python inference.py --checkpoint checkpoints/flowclean_best.pt \
                        --output_dir ./enhanced \
                        --ode_steps 10 \
                        --solver euler
"""

import argparse
from pathlib import Path

import torch
import torchaudio

from flowclean.config import FlowCleanConfig
from flowclean.models import FlowCleanUNet
from flowclean.data import VoiceBankDEMAND
from flowclean.utils.stft import stft, istft


@torch.no_grad()
def euler_solve(
    model: FlowCleanUNet,
    x_noisy_spec: torch.Tensor,
    z0: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """Euler ODE solver: K uniform steps from t=0 to t=1.

    Args:
        model: velocity network.
        x_noisy_spec: (B, 2, F, T) noisy spectrogram condition.
        z0: (B, 2, F, T) initial Gaussian noise.
        K: number of steps.

    Returns:
        z1: (B, 2, F, T) enhanced spectrogram estimate.
    """
    dt = 1.0 / K
    z = z0
    for k in range(K):
        t_val = k * dt
        t = torch.full((z.shape[0],), t_val, device=z.device)
        v = model(z, t, x_noisy_spec)
        z = z + dt * v
    return z


@torch.no_grad()
def heun_solve(
    model: FlowCleanUNet,
    x_noisy_spec: torch.Tensor,
    z0: torch.Tensor,
    K: int,
) -> torch.Tensor:
    """Heun (improved Euler) ODE solver.

    Each step uses two function evaluations for better accuracy.
    """
    dt = 1.0 / K
    z = z0
    for k in range(K):
        t_val = k * dt
        t = torch.full((z.shape[0],), t_val, device=z.device)
        v1 = model(z, t, x_noisy_spec)

        z_euler = z + dt * v1
        t_next = torch.full((z.shape[0],), min(t_val + dt, 1.0), device=z.device)
        v2 = model(z_euler, t_next, x_noisy_spec)

        z = z + dt * 0.5 * (v1 + v2)
    return z


def enhance_waveform(
    model: FlowCleanUNet,
    noisy_wav: torch.Tensor,
    cfg: FlowCleanConfig,
    K: int = 10,
    solver: str = "euler",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Enhance a single noisy waveform.

    Args:
        model: trained FlowClean model.
        noisy_wav: (T,) mono waveform.
        cfg: FlowCleanConfig with STFT parameters.
        K: number of ODE steps.
        solver: "euler" or "heun".

    Returns:
        enhanced: (T,) waveform.
    """
    model.eval()
    orig_len = noisy_wav.shape[-1]
    noisy_wav = noisy_wav.unsqueeze(0).to(device)  # (1, T)
    stft_kwargs = cfg.stft.to_dict()

    # Step 1: Compute noisy spectrogram
    X = stft(noisy_wav, **stft_kwargs)  # (1, 2, F, T')

    # Step 2: Initialize from Gaussian
    z0 = torch.randn_like(X)

    # Step 3: ODE integration
    solve_fn = euler_solve if solver == "euler" else heun_solve
    Y_hat = solve_fn(model, X, z0, K)

    # Step 4: Inverse STFT
    enhanced = istft(Y_hat, **stft_kwargs, length=orig_len)
    return enhanced.squeeze(0)  # (T,)


def evaluate_metrics(enhanced_dir: str, test_ds: VoiceBankDEMAND, sample_rate: int, cfg: FlowCleanConfig):
    """Compute PESQ and STOI on enhanced files using clean refs from HF dataset."""
    try:
        from pesq import pesq
        from pystoi import stoi
    except ImportError:
        print("Install pesq and pystoi for metrics: pip install pesq pystoi")
        return

    enhanced_dir = Path(enhanced_dir)

    pesq_scores = []
    stoi_scores = []

    for i in range(len(test_ds)):
        fname = test_ds.filenames[i]
        enh_path = enhanced_dir / fname
        if not enh_path.exists():
            continue

        enh, sr = torchaudio.load(str(enh_path))
        enh = enh[0].numpy()

        cln = test_ds[i]["clean"].numpy()

        min_len = min(len(enh), len(cln))
        enh = enh[:min_len]
        cln = cln[:min_len]

        pesq_scores.append(pesq(sample_rate, cln, enh, "wb"))
        stoi_scores.append(stoi(cln, enh, sample_rate, extended=False))

    n = len(pesq_scores)
    if n > 0:
        avg_pesq = sum(pesq_scores) / n
        avg_stoi = sum(stoi_scores) / n
        print(f"\nMetrics over {n} files:")
        print(f"  PESQ:  {avg_pesq:.3f}")
        print(f"  STOI:  {avg_stoi:.4f}")

        if cfg.wandb.use_wandb:
            import wandb
            wandb.log({"eval/pesq": avg_pesq, "eval/stoi": avg_stoi})
    else:
        print("No files to evaluate.")


def main():
    parser = argparse.ArgumentParser(description="FlowClean Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default="./enhanced")
    parser.add_argument("--ode_steps", type=int, default=10, help="Number of ODE steps K")
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "heun"])
    parser.add_argument("--eval_metrics", action="store_true", help="Compute PESQ/STOI")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = FlowCleanConfig.from_yaml(args.config)

    # Wandb (optional)
    if cfg.wandb.use_wandb:
        import wandb
        if cfg.wandb.wandb_token:
            wandb.login(key=cfg.wandb.wandb_token)
        wandb.init(
            project=cfg.wandb.project,
            job_type="inference",
        )

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Build model
    model = FlowCleanUNet(
        base_channels=cfg.model.base_channels,
        num_levels=cfg.model.num_levels,
        time_dim=cfg.model.time_dim,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    # Load test set (from HuggingFace: JacobLinCool/VoiceBank-DEMAND-16k)
    test_ds = VoiceBankDEMAND(
        split="test",
        segment_length=None,  # full utterance
        sample_rate=cfg.data.sample_rate,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Enhancing {len(test_ds)} utterances with K={args.ode_steps} ({args.solver})...")
    for i in range(len(test_ds)):
        sample = test_ds[i]
        noisy = sample["noisy"]
        enhanced = enhance_waveform(
            model, noisy, cfg,
            K=args.ode_steps,
            solver=args.solver,
            device=device,
        )
        fname = test_ds.filenames[i]
        out_path = output_dir / fname
        torchaudio.save(str(out_path), enhanced.unsqueeze(0).cpu(), cfg.data.sample_rate)

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(test_ds)}")

    print(f"Enhanced files saved to {output_dir}")

    # Evaluate metrics
    if args.eval_metrics:
        evaluate_metrics(str(output_dir), test_ds, cfg.data.sample_rate, cfg)

    if cfg.wandb.use_wandb:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
