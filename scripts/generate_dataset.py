"""Command-line interface for PCC path demonstration generation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pcc_diffuser.generation import PccGenerationConfig, generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PCC paths for selected static-sphere scene counts. Obstacle "
            "paths use straight-workspace IK followed by batched collision repair; "
            "obstacle-free scenes retain only successful straight paths."
        )
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory in which dataset.npz and dataset_meta.json are written",
    )
    parser.add_argument(
        "--target-samples",
        type=int,
        default=10_000,
        help="number of distinct Cartesian targets to generate (default: 10000)",
    )
    parser.add_argument(
        "--start-count",
        type=int,
        default=5,
        help=(
            "exactly N starts: zero plus N-1 random when zero is clear, "
            "otherwise N random starts "
            "(default: 5)"
        ),
    )
    parser.add_argument(
        "--terminal-count",
        type=int,
        default=10,
        help="exact separated terminal IK modes requested per target (default: 10)",
    )
    parser.add_argument(
        "--obstacle-counts",
        type=int,
        nargs="+",
        choices=(0, 1, 2, 4),
        default=(0, 1, 2, 4),
        help=(
            "sphere counts cycled equally across target scenes; use "
            "--obstacle-counts 0 for a purely obstacle-free dataset; "
            "other combinations are supported, such as --obstacle-counts 1 2 4 "
            "(default: 0 1 2 4)"
        ),
    )
    parser.add_argument(
        "--target-batch-size",
        type=int,
        default=256,
        help="targets sharing one obstacle layout per generation batch (default: 256)",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=32,
        help="configuration candidates in the single terminal/start sample batch (default: 32)",
    )
    parser.add_argument(
        "--path-batch-size",
        type=int,
        default=64,
        help="start-terminal paths processed together (default: 64)",
    )
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-bending", type=float, default=math.pi / 2.0)
    parser.add_argument("--radial-min", type=float, default=0.1)
    parser.add_argument("--radial-max", type=float, default=2.5)
    parser.add_argument("--z-min", type=float, default=1.5)
    parser.add_argument("--z-max", type=float, default=3.0)
    parser.add_argument("--ik-iterations", type=int, default=200)
    parser.add_argument(
        "--ik-convergence-interval",
        type=int,
        default=10,
        help="iterations between synchronising IK convergence checks (default: 10)",
    )
    parser.add_argument("--ik-damping", type=float, default=1e-4)
    parser.add_argument("--ik-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--ik-continuity-gain",
        type=float,
        default=0.1,
        help="null-space configuration-continuity correction gain (default: 0.1)",
    )
    parser.add_argument(
        "--separation",
        type=float,
        default=0.5,
        help="minimum separation between sampled configurations (default: 0.5)",
    )
    parser.add_argument(
        "--max-adjacent-step",
        type=float,
        default=0.4,
        help=(
            "maximum adjacent configuration change in a retained path "
            "(default: 0.4)"
        ),
    )
    parser.add_argument(
        "--min-path-distance",
        type=float,
        default=0.6,
        help="minimum Cartesian distance from a random-start tip to the target (default: 0.6)",
    )
    parser.add_argument("--sphere-radius", type=float, default=0.4)
    parser.add_argument(
        "--obstacle-layout-radius",
        type=float,
        default=1.2,
        help="radial placement of prescribed 2/4-obstacle layouts (default: 1.2)",
    )
    parser.add_argument("--obstacle-z", type=float, default=1.5)
    parser.add_argument("--clearance-margin", type=float, default=0.2)
    parser.add_argument(
        "--repulsion-clearance-factor",
        type=float,
        default=1.5,
        help="clearance multiplier defining the collision-repulsion region (default: 1.5)",
    )
    parser.add_argument(
        "--max-collision-repairs",
        type=int,
        default=10,
        help="maximum collision-repulsion and path-IK rounds (default: 10)",
    )
    parser.add_argument(
        "--spline-dp-tolerance",
        type=float,
        default=0.1,
        help=(
            "Douglas-Peucker workspace tolerance for selecting clamped "
            "B-spline controls (not dynamic programming; default: 0.1)"
        ),
    )
    parser.add_argument("--split-seed", type=int, default=999)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="generate one tiny scene for each obstacle count; not training data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PccGenerationConfig(
        output_dir=args.output_dir,
        seed=args.seed,
        target_samples=10 * len(args.obstacle_counts) if args.smoke else args.target_samples,
        target_batch_size=args.target_batch_size,
        start_count=1 if args.smoke else args.start_count,
        terminal_count=1 if args.smoke else args.terminal_count,
        sample_batch_size=args.sample_batch_size,
        path_batch_size=args.path_batch_size,
        obstacle_counts=tuple(args.obstacle_counts),
        horizon=args.horizon,
        section_lengths=(1.0, 1.0, 1.0),
        max_bending=args.max_bending,
        radial_min=args.radial_min,
        radial_max=args.radial_max,
        z_min=args.z_min,
        z_max=args.z_max,
        ik_iterations=args.ik_iterations,
        ik_convergence_interval=args.ik_convergence_interval,
        ik_damping=args.ik_damping,
        ik_tolerance=args.ik_tolerance,
        ik_continuity_gain=args.ik_continuity_gain,
        separation=args.separation,
        max_adjacent_step=args.max_adjacent_step,
        min_path_distance=args.min_path_distance,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        sphere_radius=args.sphere_radius,
        obstacle_layout_radius=args.obstacle_layout_radius,
        obstacle_z=args.obstacle_z,
        clearance_margin=args.clearance_margin,
        repulsion_clearance_factor=args.repulsion_clearance_factor,
        max_collision_repairs=args.max_collision_repairs,
        spline_dp_tolerance=args.spline_dp_tolerance,
        device=args.device,
        overwrite=args.overwrite,
    )

    output = Path(config.output_dir).resolve()
    started = time.monotonic()

    def show_progress(message: str) -> None:
        elapsed = time.monotonic() - started
        print(f"[{elapsed:9.1f} s] {message}", flush=True)

    print(
        "Starting PCC dataset generation\n"
        f"  output: {output}\n"
        f"  requested targets: {config.target_samples}\n"
        f"  requested candidate paths: {config.maximum_path_count}\n"
        f"  requested terminals per target: {config.terminal_count}\n"
        f"  requested starts per target: {config.start_count}\n"
        f"  targets per batch: {config.target_batch_size}\n"
        f"  sample batch size: {config.sample_batch_size}\n"
        f"  path batch size: {config.path_batch_size}\n"
        f"  interpolation: straight workspace IK\n"
        f"  IK convergence interval: {config.ik_convergence_interval}\n"
        f"  IK continuity gain: {config.ik_continuity_gain}\n"
        f"  obstacle counts: {', '.join(map(str, config.obstacle_counts))}\n"
        f"  obstacle layout radius: {config.obstacle_layout_radius}\n"
        f"  obstacle z: {config.obstacle_z}\n"
        f"  sphere radius: {config.sphere_radius}\n"
        f"  clearance margin: {config.clearance_margin}\n"
        f"  repulsion clearance factor: {config.repulsion_clearance_factor}\n"
        f"  maximum collision repairs: {config.max_collision_repairs}\n"
        f"  spline Douglas-Peucker tolerance: {config.spline_dp_tolerance}\n"
        f"  horizon: {config.horizon}\n"
        f"  device: {config.device}",
        flush=True,
    )
    summary = generate_dataset(config, progress=show_progress)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "requested_targets": config.target_samples,
                "actual_targets": summary.target_count,
                "requested_paths": config.maximum_path_count,
                "actual_paths": summary.path_count,
                "path_failure_counts": summary.path_failure_counts,
                "total_elapsed_seconds": summary.elapsed_seconds,
                "archive": str(output / "dataset.npz"),
                "metadata": str(output / "dataset_meta.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
