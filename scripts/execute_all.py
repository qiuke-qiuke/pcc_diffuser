"""Generate a dataset, train a model, and evaluate its final checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PCC dataset generation, model training, and evaluation in order."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--output-dir",
        required=True,
        help="root directory for the automatically named training run",
    )
    parser.add_argument("--target-samples", type=int, default=10_000)
    parser.add_argument("--target-batch-size", type=int, default=256)
    parser.add_argument("--terminal-count", type=int, default=10)
    parser.add_argument("--sample-batch-size", type=int, default=32)
    parser.add_argument("--path-batch-size", type=int, default=64)
    parser.add_argument("--start-count", type=int, default=5)
    parser.add_argument("--min-path-distance", type=float, default=0.6)
    parser.add_argument("--obstacle-layout-radius", type=float, default=1.2)
    parser.add_argument(
        "--obstacle-counts",
        type=int,
        nargs="+",
        choices=(0, 1, 2, 4),
        default=(0, 1, 2, 4),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolved(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_DIR / candidate
    return candidate.resolve()


def _run(command: list[str], stage: str) -> None:
    print(f"\nStarting {stage}\n  {' '.join(command)}\n", flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def _preflight_outputs(dataset: Path, run_dir: Path, overwrite: bool) -> None:
    target_files = [
        dataset / "dataset.npz",
        dataset / "dataset_meta.json",
        run_dir / "final.pt",
        run_dir / "log.npz",
        run_dir / "log_meta.json",
        run_dir / "eval.npz",
        run_dir / "eval_meta.json",
    ]
    if run_dir.is_dir():
        target_files.extend(run_dir.glob("step_*.pt"))
    existing = [path for path in target_files if path.exists()]
    if existing and not overwrite:
        descriptions = ", ".join(str(path) for path in existing)
        raise SystemExit(
            f"output files already exist: {descriptions}; pass --overwrite"
        )


def main() -> None:
    args = parse_args()
    dataset = _resolved(args.dataset)
    config_path = _resolved(args.config)
    runs_dir = _resolved(args.output_dir)
    run_dir = runs_dir / dataset.name
    checkpoint = run_dir / "final.pt"
    _preflight_outputs(dataset, run_dir, args.overwrite)

    generate_command = [
        sys.executable,
        "-u",
        str(PROJECT_DIR / "scripts" / "generate_dataset.py"),
        "--output-dir",
        str(dataset),
        "--target-samples",
        str(args.target_samples),
        "--target-batch-size",
        str(args.target_batch_size),
        "--terminal-count",
        str(args.terminal_count),
        "--sample-batch-size",
        str(args.sample_batch_size),
        "--path-batch-size",
        str(args.path_batch_size),
        "--start-count",
        str(args.start_count),
        "--min-path-distance",
        str(args.min_path_distance),
        "--obstacle-layout-radius",
        str(args.obstacle_layout_radius),
        "--obstacle-counts",
        *(str(count) for count in args.obstacle_counts),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    train_command = [
        sys.executable,
        "-u",
        str(PROJECT_DIR / "scripts" / "train_model.py"),
        "--config",
        str(config_path),
        "--dataset",
        str(dataset),
        "--output-dir",
        str(runs_dir),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    evaluate_command = [
        sys.executable,
        "-u",
        str(PROJECT_DIR / "scripts" / "evaluate_model.py"),
        "--output-dir",
        str(run_dir),
        "--dataset",
        str(dataset),
        "--checkpoint",
        str(checkpoint),
        "--device",
        args.device,
    ]
    if args.overwrite:
        generate_command.append("--overwrite")
        train_command.append("--overwrite")
        evaluate_command.append("--overwrite")

    _run(generate_command, "dataset generation")
    _run(train_command, "model training")
    _run(evaluate_command, "model evaluation")
    print(f"\nAll stages complete\n  output: {run_dir}\n", flush=True)


if __name__ == "__main__":
    main()
