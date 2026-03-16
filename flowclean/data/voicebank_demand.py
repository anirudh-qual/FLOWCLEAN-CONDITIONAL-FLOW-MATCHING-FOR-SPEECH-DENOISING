"""VoiceBank-DEMAND dataset for FlowClean.

Loads JacobLinCool/VoiceBank-DEMAND-16k from HuggingFace.
"""

import random

import torch
import torchaudio
from torch.utils.data import Dataset
from datasets import load_dataset


class VoiceBankDEMAND(Dataset):
    """Paired noisy/clean speech dataset from HuggingFace.

    Args:
        split: "train" or "test".
        segment_length: fixed waveform length in samples (default ~2s at 16kHz).
            None = return full utterance (for evaluation).
        sample_rate: expected sample rate.
    """

    def __init__(
        self,
        split: str = "train",
        segment_length: int | None = 32000,
        sample_rate: int = 16000,
    ):
        super().__init__()
        self.split = split
        self.segment_length = segment_length
        self.sample_rate = sample_rate

        self.hf_dataset = load_dataset(
            "JacobLinCool/VoiceBank-DEMAND-16k",
            split=split,
        )
        # Build filenames list from the 'id' column for compatibility
        # Use column access (fast) instead of row iteration (decodes audio)
        self.filenames = [uid + ".wav" for uid in self.hf_dataset["id"]]

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.hf_dataset[idx]

        # HF audio columns decode to {"array": np.ndarray, "sampling_rate": int}
        clean_audio = row["clean"]
        noisy_audio = row["noisy"]

        clean = torch.from_numpy(clean_audio["array"]).float()
        noisy = torch.from_numpy(noisy_audio["array"]).float()

        # Resample if needed (dataset is already 16kHz, but just in case)
        sr = clean_audio["sampling_rate"]
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            clean = resampler(clean)
            noisy = resampler(noisy)

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
