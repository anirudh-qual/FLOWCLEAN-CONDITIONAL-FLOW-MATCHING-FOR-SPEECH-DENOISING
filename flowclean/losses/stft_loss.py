"""Multi-Resolution STFT Loss for FlowClean.

Auxiliary perceptual loss computed in waveform space at multiple
STFT resolutions to improve magnitude and phase quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleResolutionSTFTLoss(nn.Module):
    """Spectral convergence + log magnitude loss at a single resolution."""

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_hat: predicted waveform (B, T)
            y: target waveform (B, T)
        Returns:
            scalar loss
        """
        window = torch.hann_window(self.win_length, device=y.device)
        # Complex STFT
        Y_hat = torch.stft(
            y_hat, self.n_fft, self.hop_length, self.win_length,
            window=window, return_complex=True,
        )
        Y = torch.stft(
            y, self.n_fft, self.hop_length, self.win_length,
            window=window, return_complex=True,
        )

        mag_hat = Y_hat.abs()
        mag = Y.abs()

        # Spectral convergence loss
        sc_loss = torch.norm(mag - mag_hat, p="fro") / (torch.norm(mag, p="fro") + 1e-8)

        # Log magnitude loss
        log_loss = F.l1_loss(torch.log(mag + 1e-8), torch.log(mag_hat + 1e-8))

        return sc_loss + log_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss (L_MR-STFT from the paper).

    Averages single-resolution losses across multiple STFT configurations.
    """

    def __init__(
        self,
        fft_sizes: list[int] = [512, 1024, 2048],
        hop_sizes: list[int] = [120, 240, 480],
        win_sizes: list[int] = [512, 1024, 2048],
    ):
        super().__init__()
        self.losses = nn.ModuleList([
            SingleResolutionSTFTLoss(n, h, w)
            for n, h, w in zip(fft_sizes, hop_sizes, win_sizes)
        ])

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for loss_fn in self.losses:
            total = total + loss_fn(y_hat, y)
        return total / len(self.losses)
