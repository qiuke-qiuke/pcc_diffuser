"""Small YAML-backed configuration object used by the training CLI."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TrainingConfig:
    split: str

    model_dim: int
    dim_mults: tuple[int, ...]
    context_dim: int
    dropout: float
    graph_hidden_dim: int
    graph_layers: int
    train_timesteps: int

    batch_size: int
    learning_rate: float
    weight_decay: float
    training_steps: int
    gradient_accumulation: int
    gradient_clip: float
    ema_decay: float
    ema_update_every: int
    num_workers: int
    save_every: int
    log_every: int
    seed: int
    device: str

    def __post_init__(self) -> None:
        if self.split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainingConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError("training config must be a YAML mapping")
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ValueError(f"unknown training config fields: {', '.join(unknown)}")
        missing = sorted(allowed - set(raw))
        if missing:
            raise ValueError(f"missing training config fields: {', '.join(missing)}")
        for name in ("dim_mults",):
            if name in raw:
                raw[name] = tuple(raw[name])
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["dim_mults"] = list(self.dim_mults)
        return result
