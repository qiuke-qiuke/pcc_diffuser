"""Command-line interface for training the unified PCC path diffuser."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys

# Permit running directly from a source checkout without installation.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader

from pcc_diffuser.config import TrainingConfig
from pcc_diffuser.data import PccDataset, collate_pcc_batch
from pcc_diffuser.diffusion import DiffusionConfig, GaussianPathDiffusion
from pcc_diffuser.model import ConditionalTemporalUNet, ModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train p(configuration-space path | q_start, target_tip, obstacle_set)."
            "Only path frame zero is hard-inpainted; the terminal configuration remains latent."
        )
    )
    parser.add_argument("--config", required=True, help="YAML config")
    parser.add_argument(
        "--dataset",
        required=True,
        help="directory containing dataset.npz and dataset_meta.json",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="root directory for the automatically named training run",
    )
    parser.add_argument("--steps", type=int, help="override number of optimiser steps")
    parser.add_argument("--device", help="override torch device, e.g. cpu or cuda:0")
    parser.add_argument("--seed", type=int, help="override random seed")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace a nonempty run directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainingConfig.from_yaml(args.config)

    overrides = {}
    if args.steps is not None:
        overrides["training_steps"] = args.steps
    if args.device is not None:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = args.seed
    config = replace(config, **overrides)
    dataset_directory = Path(args.dataset).expanduser().resolve()
    missing_dataset_files = [
        name
        for name in ("dataset.npz", "dataset_meta.json")
        if not (dataset_directory / name).is_file()
    ]
    if missing_dataset_files:
        raise SystemExit(
            f"dataset directory is missing {', '.join(missing_dataset_files)}: "
            f"{dataset_directory}"
        )
    run_name = dataset_directory.name
    runs_directory = Path(args.output_dir).expanduser().resolve()
    output_directory = runs_directory / run_name
    if output_directory.parent != runs_directory:
        raise RuntimeError("resolved training output escaped the runs directory")
    training_outputs = [
        output_directory / "final.pt",
        output_directory / "log.npz",
        output_directory / "log_meta.json",
    ]
    if output_directory.is_dir():
        training_outputs.extend(output_directory.glob("step_*.pt"))
    existing_outputs = [path for path in training_outputs if path.exists()]
    if existing_outputs:
        if not args.overwrite:
            names = ", ".join(path.name for path in existing_outputs)
            raise SystemExit(
                f"training output already exists in {output_directory}: {names}; "
                "pass --overwrite"
            )
        for path in existing_outputs:
            path.unlink()
    print(f"Training output: {output_directory}")

    dataset = PccDataset.load(dataset_directory, split=config.split)
    zero_start_fraction = float(
        np.mean(np.max(np.abs(dataset.paths[:, 0]), axis=1) < 1e-6)
    )
    print(
        f"Loaded {len(dataset)} paths (H={dataset.horizon}), "
        f"zero start {100.0 * zero_start_fraction:.2f}%, "
        f"random start {100.0 * (1.0 - zero_start_fraction):.2f}%"
    )

    model_config = ModelConfig(
        horizon=dataset.horizon,
        transition_dim=dataset.paths.shape[-1],
        model_dim=config.model_dim,
        dim_mults=config.dim_mults,
        context_dim=config.context_dim,
        dropout=config.dropout,
        graph_hidden_dim=config.graph_hidden_dim,
        graph_layers=config.graph_layers,
    )
    model = ConditionalTemporalUNet(model_config)
    diffusion = GaussianPathDiffusion(
        model, DiffusionConfig(train_timesteps=config.train_timesteps)
    )

    # Import here so --help and dataset diagnostics remain usable independently.
    from pcc_diffuser.training import Trainer, TrainingConfig as LoopConfig

    loader_generator = torch.Generator().manual_seed(config.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda") and torch.cuda.is_available(),
        generator=loader_generator,
        collate_fn=collate_pcc_batch,
    )
    loop_config = LoopConfig(
        num_steps=config.training_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=config.gradient_accumulation,
        max_gradient_norm=config.gradient_clip,
        ema_decay=config.ema_decay,
        ema_start_step=0,
        ema_update_every=config.ema_update_every,
        checkpoint_every=config.save_every,
        log_every=config.log_every,
        output_dir=str(output_directory),
        device=config.device,
        seed=config.seed,
        deterministic=True,
    )

    run_config = config.to_dict()
    run_config["dataset"] = str(dataset_directory)
    trainer = Trainer(
        diffusion,
        dataloader,
        dataset.normaliser,
        loop_config,
        section_lengths=dataset.section_lengths,
        run_config=run_config,
    )
    checkpoint = trainer.train()
    print(f"Training complete: {checkpoint}")


if __name__ == "__main__":
    main()
