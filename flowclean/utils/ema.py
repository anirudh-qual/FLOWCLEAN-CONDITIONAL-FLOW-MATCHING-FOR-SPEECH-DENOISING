"""Lightweight EMA wrapper for model parameters.

Maintains shadow copies of every trainable parameter and updates them
with `decay * shadow + (1 - decay) * param` after each optimizer step.
Use `apply_to(model)` to swap shadow weights into the live model for
inference, and `restore(model)` to put the original weights back.
"""

from contextlib import contextmanager
from copy import deepcopy

import torch
import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": {k: v.cpu() for k, v in self.shadow.items()}}

    def load_state_dict(self, state: dict, device=None) -> None:
        self.decay = state["decay"]
        self.shadow = {k: (v.to(device) if device is not None else v) for k, v in state["shadow"].items()}

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        """Swap the live model parameters with the EMA shadow weights."""
        self._backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                self._backup[name] = param.detach().clone()
                param.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Restore the original parameters saved by apply_to."""
        for name, param in model.named_parameters():
            if name in self._backup:
                param.copy_(self._backup[name])
        self._backup = {}

    @contextmanager
    def average_parameters(self, model: nn.Module):
        self.apply_to(model)
        try:
            yield
        finally:
            self.restore(model)

    def shadow_state_dict_for_model(self, model: nn.Module) -> dict:
        """Return a state_dict-shaped dict where parameter entries are EMA values
        and buffers come from the live model."""
        out = deepcopy(model.state_dict())
        for name, val in self.shadow.items():
            if name in out:
                out[name] = val.detach().clone()
        return out
