"""STFT utilities for FlowClean.

Converts waveforms to 2-channel (real, imag) complex STFT tensors
and back, matching the paper's R^{2 x F x T} representation.

Optionally applies SGMSE+-style power-law compression on the complex
spectrogram so the network operates on a tighter dynamic range:

    c(z) = beta * |z|^alpha * exp(j * angle(z))

with the standard inverse

    c^{-1}(z') = (|z'| / beta)^{1/alpha} * exp(j * angle(z'))
"""

import torch


def _compress_complex(spec: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Apply c(z) = beta * |z|^alpha * exp(j*angle(z)) to a complex tensor."""
    if alpha == 1.0 and beta == 1.0:
        return spec
    mag = spec.abs().clamp_min(1e-8)
    return beta * mag.pow(alpha) * torch.exp(1j * spec.angle())


def _decompress_complex(spec: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    """Inverse of _compress_complex."""
    if alpha == 1.0 and beta == 1.0:
        return spec
    mag = spec.abs().clamp_min(1e-8)
    return (mag / beta).pow(1.0 / alpha) * torch.exp(1j * spec.angle())


def stft(
    waveform: torch.Tensor,
    n_fft: int = 510,
    hop_length: int = 128,
    win_length: int = 510,
    center: bool = True,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Compute complex STFT and return as 2-channel real tensor.

    Args:
        waveform: (B, T) or (T,)
        n_fft, hop_length, win_length: STFT parameters.
        center: whether to pad signal.
        alpha, beta: power-law compression parameters (1.0, 1.0 disables it).

    Returns:
        Tensor of shape (B, 2, F, T') where channel 0 = real, 1 = imag.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    window = torch.hann_window(win_length, device=waveform.device)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True,
    )
    spec = _compress_complex(spec, alpha, beta)
    return torch.stack([spec.real, spec.imag], dim=1)


def istft(
    spec_tensor: torch.Tensor,
    n_fft: int = 510,
    hop_length: int = 128,
    win_length: int = 510,
    center: bool = True,
    length: int | None = None,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> torch.Tensor:
    """Inverse STFT from 2-channel real tensor.

    Decompresses with c^{-1} before applying torch.istft.
    """
    real = spec_tensor[:, 0]
    imag = spec_tensor[:, 1]
    spec = torch.complex(real, imag)
    spec = _decompress_complex(spec, alpha, beta)

    window = torch.hann_window(win_length, device=spec_tensor.device)
    return torch.istft(
        spec,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        length=length,
    )


def spec_to_tensor(spec_complex: torch.Tensor) -> torch.Tensor:
    """Convert complex spectrogram to 2-channel real tensor."""
    return torch.stack([spec_complex.real, spec_complex.imag], dim=1)


def tensor_to_spec(tensor: torch.Tensor) -> torch.Tensor:
    """Convert 2-channel real tensor back to complex spectrogram."""
    return torch.complex(tensor[:, 0], tensor[:, 1])
