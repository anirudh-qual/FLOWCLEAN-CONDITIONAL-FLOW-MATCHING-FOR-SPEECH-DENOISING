"""VoiceBank-DEMAND dataset for FlowClean.

Expects the standard directory layout:
    data_root/
        clean_trainset_28spk_wav/   (or clean_trainset_wav/)
        noisy_trainset_28spk_wav/   (or noisy_trainset_wav/)
        clean_testset_wav/
        noisy_testset_wav/

All files are expected at 16 kHz mono.
"""

import os
import random
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset


class VoiceBankDEMAND(Dataset):
    """Paired noisy/clean speech dataset.

    Args:
        root: path to VoiceBank-DEMAND root.
        split: "train" or "test".
        segment_length: fixed waveform length in samples (default ~2s at 16kHz).
            None = return full utterance (for evaluation).
        sample_rate: expected sample rate.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        segment_length: int | None = 32000,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.segment_length = segment_length
        self.sample_rate = sample_rate

        if split == "train":
            clean_dir_candidates = [
                "clean_trainset_28spk_wav",
                "clean_trainset_wav",
            ]
            noisy_dir_candidates = [
                "noisy_trainset_28spk_wav",
                "noisy_trainset_wav",
            ]
        else:
            clean_dir_candidates = ["clean_testset_wav"]
            noisy_dir_candidates = ["noisy_testset_wav"]

        self.clean_dir = self._find_dir(clean_dir_candidates)
        self.noisy_dir = self._find_dir(noisy_dir_candidates)

        # Collect paired filenames
        clean_files = set(os.listdir(self.clean_dir))
        noisy_files = set(os.listdir(self.noisy_dir))
        common = sorted(clean_files & noisy_files)
        self.filenames = [f for f in common if f.endswith(".wav")]

        if len(self.filenames) == 0:
            raise RuntimeError(
                f"No paired .wav files found in {self.clean_dir} and {self.noisy_dir}"
            )

    def _find_dir(self, candidates: list[str]) -> Path:
        for name in candidates:
            d = self.root / name
            if d.is_dir():
                return d
        raise FileNotFoundError(
            f"None of {candidates} found under {self.root}"
        )

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fname = self.filenames[idx]
        clean, sr = torchaudio.load(str(self.clean_dir / fname))
        noisy, _ = torchaudio.load(str(self.noisy_dir / fname))

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            clean = resampler(clean)
            noisy = resampler(noisy)

        # Mono
        clean = clean[0]  # (T,)
        noisy = noisy[0]

        # Random crop or pad
        if self.segment_length is not None:
            clean, noisy = self._fix_length(clean, noisy, self.segment_length)

        return {"clean": clean, "noisy": noisy}

    @staticmethod
    def _fix_length(
        clean: torch.Tensor, noisy: torch.Tensor, length: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Randomly crop or zero-pad to fixed length."""
        t = clean.shape[0]
        if t >= length:
            start = random.randint(0, t - length)
            clean = clean[start : start + length]
            noisy = noisy[start : start + length]
        else:
            pad = length - t
            clean = torch.nn.functional.pad(clean, (0, pad))
            noisy = torch.nn.functional.pad(noisy, (0, pad))
        return clean, noisy
