"""Command-line interface for independent PCC planning baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from benchmarks.evaluation import (
    ALGORITHMS,
    BenchmarkEvaluationConfig,
    evaluate_benchmarks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more independent planning baselines."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--algorithm",
        required=True,
        nargs="+",
        choices=ALGORITHMS,
        help="one or more benchmark algorithms",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tip-success-tolerance", type=float, default=0.1)
    parser.add_argument("--samples-per-case", type=int, default=10)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--goal-bias", type=float, default=0.1)
    parser.add_argument("--step-size", type=float, default=0.2)
    parser.add_argument("--rewire-radius", type=float, default=0.3)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="recompute selected methods already present in the combined archive",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.algorithm)) != len(args.algorithm):
        raise ValueError("algorithm choices must not contain duplicates")
    metadata = evaluate_benchmarks(
        BenchmarkEvaluationConfig(
            output_dir=args.output_dir,
            dataset=args.dataset,
            algorithms=tuple(args.algorithm),
            device=args.device,
            split=args.split,
            seed=args.seed,
            overwrite=args.overwrite,
            tip_success_tolerance=args.tip_success_tolerance,
            samples_per_case=args.samples_per_case,
            max_iterations=args.max_iterations,
            goal_bias=args.goal_bias,
            step_size=args.step_size,
            rewire_radius=args.rewire_radius,
        ),
        progress=lambda message: print(message, flush=True),
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
