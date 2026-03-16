"""FlowClean inference / evaluation script.

Generates enhanced speech by solving the conditional probability flow ODE:
  dz_t/dt = v_theta(z_t, t | X)
using Euler or Heun solvers with K steps.

Optionally computes PESQ, STOI, and DNSMOS metrics on the test set.

Usage:
    python inference.py --checkpoint checkpoints/flowclean_best.pt \
                        --data_root ./data/voicebank_demand \
                        --output_dir ./enhanced \
                        --ode_steps 10 \
                        --solver euler
"""

import argparse
import os
from pathlib import Path

import torch
import torchaudio
import yaml

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
    stft_cfg: dict,
    K: int = 10,
    solver: str = "euler",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Enhance a single noisy waveform.

    Args:
        model: trained FlowClean model.
        noisy_wav: (T,) mono waveform.
        stft_cfg: STFT parameters dict.
        K: number of ODE steps.
        solver: "euler" or "heun".

    Returns:
        enhanced: (T,) waveform.
    """
    model.eval()
    orig_len = noisy_wav.shape[-1]
    noisy_wav = noisy_wav.unsqueeze(0).to(device)  # (1, T)

    # Step 1: Compute noisy spectrogram
    X = stft(noisy_wav, **stft_cfg)  # (1, 2, F, T')

    # Step 2: Initialize from Gaussian
    z0 = torch.randn_like(X)

    # Step 3: ODE integration
    solve_fn = euler_solve if solver == "euler" else heun_solve
    Y_hat = solve_fn(model, X, z0, K)

    # Step 4: Inverse STFT
    enhanced = istft(Y_hat, **stft_cfg, length=orig_len)
    return enhanced.squeeze(0)  # (T,)


def evaluate_metrics(enhanced_dir: str, clean_dir: str):
    """Compute PESQ and STOI on enhanced files. Requires pesq and pystoi packages."""
    try:
        from pesq import pesq
        from pystoi import stoi
    except ImportError:
        print("Install pesq and pystoi for metrics: pip install pesq pystoi")
        return

    enhanced_dir = Path(enhanced_dir)
    clean_dir = Path(clean_dir)

    pesq_scores = []
    stoi_scores = []

    for f in sorted(enhanced_dir.glob("*.wav")):
        clean_path = clean_dir / f.name
        if not clean_path.exists():
            continue

        enh, sr = torchaudio.load(str(f))
        cln, _ = torchaudio.load(str(clean_path))
        enh = enh[0].numpy()
        cln = cln[0].numpy()

        min_len = min(len(enh), len(cln))
        enh = enh[:min_len]
        cln = cln[:min_len]

        pesq_scores.append(pesq(sr, cln, enh, "wb"))
        stoi_scores.append(stoi(cln, enh, sr, extended=False))

    n = len(pesq_scores)
    if n > 0:
        print(f"\nMetrics over {n} files:")
        print(f"  PESQ:  {sum(pesq_scores)/n:.3f}")
        print(f"  STOI:  {sum(stoi_scores)/n:.4f}")
    else:
        print("No files to evaluate.")


def main():
    parser = argparse.ArgumentParser(description="FlowClean Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--data_root", type=str, default="./data/voicebank_demand")
    parser.add_argument("--output_dir", type=str, default="./enhanced")
    parser.add_argument("--ode_steps", type=int, default=10, help="Number of ODE steps K")
    parser.add_argument("--solver", type=str, default="euler", choices=["euler", "heun"])
    parser.add_argument("--eval_metrics", action="store_true", help="Compute PESQ/STOI")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]

    # Build model
    model = FlowCleanUNet(
        base_channels=cfg["model"]["base_channels"],
        num_levels=cfg["model"]["num_levels"],
        time_dim=cfg["model"]["time_dim"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    stft_cfg = {
        "n_fft": cfg["stft"]["n_fft"],
        "hop_length": cfg["stft"]["hop_length"],
        "win_length": cfg["stft"]["win_length"],
    }

    # Load test set
    test_ds = VoiceBankDEMAND(
        root=args.data_root,
        split="test",
        segment_length=None,  # full utterance
        sample_rate=cfg["data"]["sample_rate"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Enhancing {len(test_ds)} utterances with K={args.ode_steps} ({args.solver})...")
    for i in range(len(test_ds)):
        sample = test_ds[i]
        noisy = sample["noisy"]
        enhanced = enhance_waveform(
            model, noisy, stft_cfg,
            K=args.ode_steps,
            solver=args.solver,
            device=device,
        )
        fname = test_ds.filenames[i]
        out_path = output_dir / fname
        torchaudio.save(str(out_path), enhanced.unsqueeze(0).cpu(), cfg["data"]["sample_rate"])

        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(test_ds)}")

    print(f"Enhanced files saved to {output_dir}")

    # Evaluate metrics
    if args.eval_metrics:
        clean_test_dir = Path(args.data_root) / "clean_testset_wav"
        evaluate_metrics(str(output_dir), str(clean_test_dir))


if __name__ == "__main__":
    main()
