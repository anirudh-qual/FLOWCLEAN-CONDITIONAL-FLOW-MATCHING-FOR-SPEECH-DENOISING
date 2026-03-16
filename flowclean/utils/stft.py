"""STFT utilities for FlowClean.

Converts waveforms to 2-channel (real, imag) complex STFT tensors
and back, matching the paper's R^{2 x F x T} representation.
"""

import torch
import torch.nn.functional as F


def stft(
    waveform: torch.Tensor,
    n_fft: int = 510,
    hop_length: int = 128,
    win_length: int = 510,
    center: bool = True,
) -> torch.Tensor:
    """Compute complex STFT and return as 2-channel real tensor.

    Args:
        waveform: (B, T) or (T,)
        n_fft, hop_length, win_length: STFT parameters.
        center: whether to pad signal.

    Returns:
        Tensor of shape (B, 2, F, T') where channel 0 = real, 1 = imag.
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    window = torch.hann_window(win_length, device=waveform.device)
    # shape: (B, F, T') complex
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        return_complex=True,
    )
    # Stack real and imaginary -> (B, 2, F, T')
    return torch.stack([spec.real, spec.imag], dim=1)


def istft(
    spec_tensor: torch.Tensor,
    n_fft: int = 510,
    hop_length: int = 128,
    win_length: int = 510,
    center: bool = True,
    length: int | None = None,
) -> torch.Tensor:
    """Inverse STFT from 2-channel real tensor.

    Args:
        spec_tensor: (B, 2, F, T') tensor.
        length: desired output length.

    Returns:
        waveform: (B, T)
    """
    real = spec_tensor[:, 0]
    imag = spec_tensor[:, 1]
    spec = torch.complex(real, imag)

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
