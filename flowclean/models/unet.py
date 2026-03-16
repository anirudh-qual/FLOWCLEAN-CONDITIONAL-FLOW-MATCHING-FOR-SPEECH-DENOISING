"""FlowClean Conditional U-Net for velocity prediction.

Architecture:
  - Input: concatenation of z_t (2ch) and noisy spectrogram X (2ch) -> 4 channels
  - Time embedding via sinusoidal positional encoding + MLP
  - Encoder-decoder with skip connections
  - Output: predicted velocity v_theta (2 channels: real, imag)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding (same as used in diffusion models)."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) tensor of timesteps in [0, 1].
        Returns:
            (B, dim) embedding.
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=device) / half
        )
        args = t[:, None] * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class TimeMLPBlock(nn.Module):
    """Project time embedding to channel dimension for FiLM conditioning."""

    def __init__(self, time_dim: int, channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels * 2),
        )

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (scale, shift) each of shape (B, C)."""
        out = self.mlp(t_emb)
        scale, shift = out.chunk(2, dim=-1)
        return scale, shift


class ResBlock(nn.Module):
    """Residual block with time conditioning via FiLM."""

    def __init__(self, channels: int, time_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.time_mlp = TimeMLPBlock(time_dim, channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # FiLM conditioning
        scale, shift = self.time_mlp(t_emb)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class DownBlock(nn.Module):
    """Encoder block: ResBlock + downsample."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.res = ResBlock(in_ch, time_dim)
        self.down = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.res(x, t_emb)
        return self.down(h), h  # downsampled, skip


class UpBlock(nn.Module):
    """Decoder block: upsample + concat skip + ResBlock."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, time_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.res = ResBlock(out_ch + skip_ch, time_dim)
        self.proj = nn.Conv2d(out_ch + skip_ch, out_ch, 1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        # Handle size mismatch from non-power-of-2 dimensions
        if h.shape != skip.shape:
            h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)
        h = torch.cat([h, skip], dim=1)
        h = self.res(h, t_emb)
        h = self.proj(h)
        return h


class FlowCleanUNet(nn.Module):
    """Conditional U-Net for FlowClean velocity prediction.

    Input channels: 4 (z_t: 2ch real/imag + X: 2ch real/imag)
    Output channels: 2 (predicted velocity: real/imag)

    Args:
        base_channels: base channel width (doubled at each level).
        num_levels: number of encoder/decoder levels.
        time_dim: dimension of time embedding.
    """

    def __init__(
        self,
        base_channels: int = 64,
        num_levels: int = 4,
        time_dim: int = 256,
    ):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input projection: 4 channels (z_t + X) -> base_channels
        self.input_conv = nn.Conv2d(4, base_channels, 3, padding=1)

        # Build encoder — track skip channel sizes
        self.encoders = nn.ModuleList()
        ch = base_channels
        enc_channels = []  # channel size of each skip connection
        for i in range(num_levels):
            out_ch = ch * 2
            self.encoders.append(DownBlock(ch, out_ch, time_dim))
            enc_channels.append(ch)  # skip comes from ResBlock *before* downsample
            ch = out_ch

        # Bottleneck
        self.bottleneck = ResBlock(ch, time_dim)

        # Build decoder — match skip channels from encoder (reversed)
        self.decoders = nn.ModuleList()
        for i in range(num_levels):
            skip_ch = enc_channels[num_levels - 1 - i]
            out_ch = ch // 2
            self.decoders.append(UpBlock(ch, skip_ch, out_ch, time_dim))
            ch = out_ch

        # Output projection -> 2 channels (real, imag)
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, 2, 1),
        )

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        x_noisy: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t: (B, 2, F, T) current state on the flow path.
            t: (B,) timesteps in [0, 1].
            x_noisy: (B, 2, F, T) noisy spectrogram conditioning.

        Returns:
            v_theta: (B, 2, F, T) predicted velocity.
        """
        # Concatenate along channel dim -> (B, 4, F, T)
        h = torch.cat([z_t, x_noisy], dim=1)
        h = self.input_conv(h)

        t_emb = self.time_embed(t)

        # Encoder path
        skips = []
        for enc in self.encoders:
            h, skip = enc(h, t_emb)
            skips.append(skip)

        # Bottleneck
        h = self.bottleneck(h, t_emb)

        # Decoder path
        for dec, skip in zip(self.decoders, reversed(skips)):
            h = dec(h, skip, t_emb)

        return self.output_conv(h)
