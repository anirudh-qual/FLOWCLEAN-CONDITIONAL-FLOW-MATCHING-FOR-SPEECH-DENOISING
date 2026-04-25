"""FlowClean unified configuration."""

from dataclasses import dataclass, field

import yaml


@dataclass
class STFTConfig:
    n_fft: int = 510
    hop_length: int = 128
    win_length: int = 510
    compress_alpha: float = 0.5
    compress_beta: float = 0.15

    def to_dict(self) -> dict:
        return {"n_fft": self.n_fft, "hop_length": self.hop_length, "win_length": self.win_length}

    def compress_kwargs(self) -> dict:
        return {"alpha": self.compress_alpha, "beta": self.compress_beta}


@dataclass
class ModelConfig:
    base_channels: int = 64
    num_levels: int = 4
    time_dim: int = 256


@dataclass
class MRSTFTConfig:
    fft_sizes: list[int] = field(default_factory=lambda: [512, 1024, 2048])
    hop_sizes: list[int] = field(default_factory=lambda: [120, 240, 480])
    win_sizes: list[int] = field(default_factory=lambda: [512, 1024, 2048])


@dataclass
class LossConfig:
    lambda_mr_stft: float = 0.1
    mr_stft: MRSTFTConfig = field(default_factory=MRSTFTConfig)


@dataclass
class SchedulerConfig:
    type: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6


@dataclass
class TrainingConfig:
    epochs: int = 200
    batch_size: int = 8
    grad_accum_steps: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-6
    grad_clip: float = 1.0
    num_workers: int = 4
    seed: int = 42
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    save_every: int = 10
    log_every: int = 100
    checkpoint_dir: str = "./checkpoints"
    use_ema: bool = True
    ema_decay: float = 0.999
    val_fraction: float = 0.05
    val_split_seed: int = 1234


@dataclass
class DataConfig:
    sample_rate: int = 16000
    segment_length: int = 32000


@dataclass
class InferenceConfig:
    ode_steps: int = 10
    solver: str = "euler"


@dataclass
class WandbConfig:
    use_wandb: bool = False
    project: str = "flowclean"
    wandb_token: str | None = None


@dataclass
class FlowCleanConfig:
    data: DataConfig = field(default_factory=DataConfig)
    stft: STFTConfig = field(default_factory=STFTConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    @staticmethod
    def from_yaml(path: str) -> "FlowCleanConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        return FlowCleanConfig._from_dict(raw)

    @staticmethod
    def _from_dict(raw: dict) -> "FlowCleanConfig":
        cfg = FlowCleanConfig()
        if "data" in raw:
            cfg.data = DataConfig(**raw["data"])
        if "stft" in raw:
            cfg.stft = STFTConfig(**raw["stft"])
        if "model" in raw:
            cfg.model = ModelConfig(**raw["model"])
        if "loss" in raw:
            loss_raw = dict(raw["loss"])
            if "mr_stft" in loss_raw:
                loss_raw["mr_stft"] = MRSTFTConfig(**loss_raw["mr_stft"])
            cfg.loss = LossConfig(**loss_raw)
        if "training" in raw:
            train_raw = dict(raw["training"])
            if "scheduler" in train_raw:
                train_raw["scheduler"] = SchedulerConfig(**train_raw["scheduler"])
            cfg.training = TrainingConfig(**train_raw)
        if "inference" in raw:
            cfg.inference = InferenceConfig(**raw["inference"])
        if "wandb" in raw:
            cfg.wandb = WandbConfig(**raw["wandb"])
        return cfg
