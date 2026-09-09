"""Deterministic finite-step training and checkpointing utilities.

The optimiser updates the denoiser owned by :class:`GaussianPathDiffusion`.
EMA is a separate, gradient-free copy of that denoiser; its decay is an
averaging coefficient and is entirely independent of the optimiser learning
rate.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from .diffusion import DiffusionConfig, GaussianPathDiffusion
from .model import ConditionalTemporalUNet, ModelConfig
from .normalisation import PccNormaliser


CHECKPOINT_VERSION = 2
TRAINING_LOG_FORMAT_VERSION = 2


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration whose ``num_steps`` counts completed optimiser updates."""

    num_steps: int
    learning_rate: float
    weight_decay: float
    gradient_accumulation_steps: int
    max_gradient_norm: float | None
    ema_decay: float
    ema_start_step: int
    ema_update_every: int
    checkpoint_every: int
    log_every: int
    output_dir: str
    device: str
    seed: int
    deterministic: bool

    def __post_init__(self) -> None:
        if self.num_steps < 0:
            raise ValueError("num_steps must be non-negative")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.max_gradient_norm is not None and (
            not math.isfinite(self.max_gradient_norm) or self.max_gradient_norm <= 0
        ):
            raise ValueError("max_gradient_norm must be finite and positive or None")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must lie in [0,1)")
        if self.ema_start_step < 0 or self.ema_update_every < 1:
            raise ValueError("EMA start/update steps must be non-negative/positive")
        if self.checkpoint_every < 0 or self.log_every < 0:
            raise ValueError("checkpoint_every and log_every must be non-negative")


def seed_everything(seed: int, deterministic: bool = True) -> torch.Generator:
    """Seed Python, Torch, CUDA, and return a seeded CPU DataLoader generator."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        # Required by deterministic CUDA matrix multiplications.  setdefault
        # respects a deliberate process-wide choice made by the caller.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def batch_to_device(
    value: Any,
    device: torch.device | str,
    *,
    non_blocking: bool = False,
) -> Any:
    """Recursively move tensors in nested mappings/sequences to ``device``."""
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, Data):
        return value.to(device=device, non_blocking=non_blocking)
    if isinstance(value, Mapping):
        return {
            key: batch_to_device(item, device, non_blocking=non_blocking)
            for key, item in value.items()
        }
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)(
            *(batch_to_device(item, device, non_blocking=non_blocking) for item in value)
        )
    if isinstance(value, tuple):
        return tuple(
            batch_to_device(item, device, non_blocking=non_blocking) for item in value
        )
    if isinstance(value, list):
        return [
            batch_to_device(item, device, non_blocking=non_blocking) for item in value
        ]
    return value


class EMA:
    """A gradient-free exponential moving average copy of one denoiser."""

    def __init__(
        self,
        model: nn.Module,
        *,
        decay: float,
        start_step: int = 0,
        update_every: int = 1,
    ) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must lie in [0,1)")
        if start_step < 0 or update_every < 1:
            raise ValueError("EMA start/update steps must be non-negative/positive")
        self.decay = float(decay)
        self.start_step = int(start_step)
        self.update_every = int(update_every)
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def copy_from(self, model: nn.Module) -> None:
        self.model.load_state_dict(model.state_dict())

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        # Before averaging starts, track the raw model exactly so EMA does not
        # retain its initialisation while the optimiser moves far away.
        if step < self.start_step:
            self.copy_from(model)
            return
        if (step - self.start_step) % self.update_every:
            return
        source_parameters = dict(model.named_parameters())
        for name, averaged in self.model.named_parameters():
            averaged.lerp_(source_parameters[name].detach(), 1.0 - self.decay)
        source_buffers = dict(model.named_buffers())
        for name, averaged in self.model.named_buffers():
            averaged.copy_(source_buffers[name].detach())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.model.state_dict()

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state)


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
    return device


def _configure_loader_seed(loader: DataLoader[Any], generator: torch.Generator) -> None:
    """Attach the seeded generator before the first DataLoader iterator exists."""
    loader.generator = generator
    sampler = getattr(loader, "sampler", None)
    if sampler is not None and hasattr(sampler, "generator"):
        sampler.generator = generator


class Trainer:
    """Train a conditional PCC denoiser for an exact number of optimiser steps."""

    def __init__(
        self,
        diffusion: GaussianPathDiffusion,
        dataloader: DataLoader[Any],
        normaliser: PccNormaliser,
        config: TrainingConfig,
        *,
        section_lengths: tuple[float, float, float],
        run_config: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(dataloader, DataLoader):
            raise TypeError("dataloader must be a torch.utils.data.DataLoader")
        if not hasattr(normaliser, "to_dict"):
            raise TypeError("normaliser must provide to_dict()")
        if len(section_lengths) != 3 or any(
            not math.isfinite(float(length)) or float(length) <= 0
            for length in section_lengths
        ):
            raise ValueError("section_lengths must contain three finite positive values")

        self.config = config
        self.device = _resolve_device(config.device)
        generator = seed_everything(config.seed, config.deterministic)
        _configure_loader_seed(dataloader, generator)
        self.dataloader = dataloader
        self._iterator = iter(dataloader)
        self.normaliser = normaliser
        self.section_lengths = tuple(float(length) for length in section_lengths)
        self.run_config = dict(run_config or {})
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.diffusion = diffusion.to(self.device)
        self.model = self.diffusion.model
        self.optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.ema = EMA(
            self.model,
            decay=config.ema_decay,
            start_step=config.ema_start_step,
            update_every=config.ema_update_every,
        )
        self.step = 0
        self._training_log: dict[str, list[float | int]] = {
            "step": [],
            "elapsed_seconds": [],
            "diffusion_loss": [],
            "gradient_norm": [],
            "mean_timestep": [],
            "learning_rate": [],
        }

    def _show_progress(self, message: str, elapsed: float) -> None:
        """Print one permanent, flushed training-status row."""
        print(f"[{elapsed:9.1f} s] {message}", flush=True)

    def _save_training_log(
        self,
        complete: bool,
        total_elapsed_seconds: float | None = None,
    ) -> None:
        """Atomically save plot-ready training history and its description."""
        if complete and total_elapsed_seconds is None:
            raise ValueError("completed training log requires total elapsed time")
        array_path = self.output_dir / "log.npz"
        temporary_array = array_path.with_name(array_path.name + ".tmp")
        arrays = {
            "step": np.asarray(self._training_log["step"], dtype=np.int64),
            **{
                name: np.asarray(values, dtype=np.float64)
                for name, values in self._training_log.items()
                if name != "step"
            },
        }
        with temporary_array.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary_array.replace(array_path)

        metadata = {
            "format": "pcc_diffuser_training_log",
            "version": TRAINING_LOG_FORMAT_VERSION,
            "complete": bool(complete),
            "total_elapsed_seconds": (
                float(total_elapsed_seconds) if complete else None
            ),
            "recorded_points": len(self._training_log["step"]),
            "last_recorded_step": (
                int(self._training_log["step"][-1])
                if self._training_log["step"]
                else None
            ),
            "total_training_steps": self.config.num_steps,
            "logging_interval_steps": self.config.log_every,
            "random_seed": self.config.seed,
            "resolved_device": str(self.device),
            "training_config": asdict(self.config),
            "model_config": self.model.config_dict(),
            "diffusion_config": self.diffusion.config_dict(),
            "section_lengths": list(self.section_lengths),
            "run_config": self.run_config,
            "metrics": {
                "step": "Completed optimiser updates.",
                "elapsed_seconds": "Wall-clock training time since this run started.",
                "diffusion_loss": (
                    "Mean squared error between predicted and sampled diffusion noise "
                    "on unconditioned path points."
                ),
                "gradient_norm": (
                    "Total parameter-gradient L2 norm before gradient clipping."
                ),
                "mean_timestep": (
                    "Mean randomly sampled diffusion timestep in the logged update."
                ),
                "learning_rate": "Optimiser learning rate after the logged update.",
            },
        }
        metadata_path = self.output_dir / "log_meta.json"
        temporary_metadata = metadata_path.with_name(metadata_path.name + ".tmp")
        temporary_metadata.write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        temporary_metadata.replace(metadata_path)

    def _next_batch(self) -> Any:
        try:
            return next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.dataloader)
            try:
                return next(self._iterator)
            except StopIteration as error:
                raise ValueError("cannot train from an empty DataLoader") from error

    def _metadata(self) -> dict[str, Any]:
        if not hasattr(self.model, "config_dict") or not hasattr(
            self.diffusion, "config_dict"
        ):
            raise TypeError("model and diffusion must expose config_dict()")
        return {
            "model_config": self.model.config_dict(),
            "diffusion_config": self.diffusion.config_dict(),
            "normaliser": self.normaliser.to_dict(),
            "training_config": asdict(self.config),
            "section_lengths": list(self.section_lengths),
            "run_config": self.run_config,
        }

    def checkpoint(self) -> dict[str, Any]:
        """Build a checkpoint; model/EMA states are denoisers, not wrappers."""
        return {
            "checkpoint_version": CHECKPOINT_VERSION,
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimiser.state_dict(),
            **self._metadata(),
        }

    def save_checkpoint(self, filename: str | None = None) -> Path:
        filename = filename or f"step_{self.step:08d}.pt"
        path = self.output_dir / filename
        temporary = path.with_name(path.name + ".tmp")
        torch.save(self.checkpoint(), temporary)
        temporary.replace(path)
        return path

    def restore_checkpoint(
        self,
        path: str | Path,
        *,
        load_optimiser: bool = True,
    ) -> None:
        state = load_checkpoint(path, map_location=self.device)
        self.model.load_state_dict(state["model"])
        self.ema.load_state_dict(state["ema"])
        if load_optimiser:
            self.optimiser.load_state_dict(state["optimizer"])
        self.step = int(state["step"])

    def train(self) -> Path:
        """Run until ``config.num_steps`` and always write ``final.pt``."""
        self.diffusion.train()
        started = time.perf_counter()
        self._save_training_log(complete=False)
        while self.step < self.config.num_steps:
            self.optimiser.zero_grad(set_to_none=True)
            metric_sums: dict[str, float] = {}
            loss_sum = 0.0

            for _ in range(self.config.gradient_accumulation_steps):
                batch = batch_to_device(
                    self._next_batch(),
                    self.device,
                    non_blocking=self.device.type == "cuda",
                )
                if not isinstance(batch, Mapping):
                    raise TypeError("GaussianPathDiffusion expects mapping batches")
                loss, metrics = self.diffusion.training_loss(dict(batch))
                if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"non-finite scalar training loss at step {self.step}: {loss}"
                    )
                (loss / self.config.gradient_accumulation_steps).backward()
                loss_sum += float(loss.detach())
                for name, value in metrics.items():
                    scalar = (
                        float(value.detach())
                        if torch.is_tensor(value)
                        else float(value)
                    )
                    metric_sums[name] = metric_sums.get(name, 0.0) + scalar

            parameters = [
                parameter
                for parameter in self.model.parameters()
                if parameter.grad is not None
            ]
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                max_norm=(
                    self.config.max_gradient_norm
                    if self.config.max_gradient_norm is not None
                    else float("inf")
                ),
                error_if_nonfinite=True,
            )
            self.optimiser.step()
            self.step += 1
            self.ema.update(self.model, self.step)

            if self.config.log_every and (
                self.step % self.config.log_every == 0
                or self.step == self.config.num_steps
            ):
                divisor = self.config.gradient_accumulation_steps
                diffusion_loss = metric_sums.get("diffusion_loss", loss_sum) / divisor
                mean_timestep = metric_sums.get("mean_t", float("nan")) / divisor
                elapsed = time.perf_counter() - started
                self._training_log["step"].append(self.step)
                self._training_log["elapsed_seconds"].append(elapsed)
                self._training_log["diffusion_loss"].append(diffusion_loss)
                self._training_log["gradient_norm"].append(float(gradient_norm))
                self._training_log["mean_timestep"].append(mean_timestep)
                self._training_log["learning_rate"].append(
                    float(self.optimiser.param_groups[0]["lr"])
                )
                self._save_training_log(complete=False)
                self._show_progress(
                    f"step {self.step}/{self.config.num_steps} | "
                    f"diffusion loss {diffusion_loss:.6f} | "
                    f"gradient norm {float(gradient_norm):.6f} | "
                    f"mean diffusion timestep {mean_timestep:.1f} | "
                    f"{100.0 * self.step / self.config.num_steps:.1f}%",
                    elapsed,
                )

            if (
                self.config.checkpoint_every
                and self.step % self.config.checkpoint_every == 0
            ):
                self.save_checkpoint()

        final_checkpoint = self.save_checkpoint("final.pt")
        total_elapsed_seconds = time.perf_counter() - started
        self._save_training_log(
            complete=True, total_elapsed_seconds=total_elapsed_seconds
        )
        return final_checkpoint


def load_checkpoint(
    path: str | Path,
    *,
    map_location: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load and minimally validate a versioned PCC diffuser checkpoint."""
    state = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("checkpoint root must be a dictionary")
    if int(state.get("checkpoint_version", -1)) != CHECKPOINT_VERSION:
        raise ValueError(
            f"unsupported checkpoint version {state.get('checkpoint_version')!r}"
        )
    required = {
        "step",
        "model",
        "ema",
        "optimizer",
        "model_config",
        "diffusion_config",
        "normaliser",
        "training_config",
        "section_lengths",
    }
    missing = sorted(required.difference(state))
    if missing:
        raise ValueError(f"checkpoint is missing fields: {missing}")
    return state


def reconstruct_diffusion(
    path: str | Path,
    *,
    weights: Literal["ema", "model"] = "ema",
    device: torch.device | str = "cpu",
) -> tuple[GaussianPathDiffusion, PccNormaliser, dict[str, Any]]:
    """Reconstruct an inference diffusion model and normaliser from checkpoint."""
    if weights not in ("ema", "model"):
        raise ValueError("weights must be 'ema' or 'model'")
    # Reconstruct on CPU so an inference-only load does not also materialise
    # the saved AdamW moments on accelerator memory.
    state = load_checkpoint(path, map_location="cpu")
    model_config = dict(state["model_config"])
    model_config["dim_mults"] = tuple(model_config["dim_mults"])
    model = ConditionalTemporalUNet(ModelConfig(**model_config))
    diffusion = GaussianPathDiffusion(
        model, DiffusionConfig(**dict(state["diffusion_config"]))
    )
    model.load_state_dict(state[weights])
    diffusion.to(device).eval()
    normaliser_state = state.get("normaliser")
    if normaliser_state is None:
        raise KeyError("checkpoint is missing normaliser state")
    normaliser = PccNormaliser.from_dict(normaliser_state)
    return diffusion, normaliser, state


__all__ = [
    "CHECKPOINT_VERSION",
    "TRAINING_LOG_FORMAT_VERSION",
    "EMA",
    "Trainer",
    "TrainingConfig",
    "batch_to_device",
    "load_checkpoint",
    "reconstruct_diffusion",
    "seed_everything",
]
