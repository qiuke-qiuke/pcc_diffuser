"""Command-line interface for evaluating a trained PCC path diffuser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pcc_diffuser.evaluation import PccEvaluationConfig, evaluate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the test split of a PCC dataset."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="directory in which eval.npz and eval_meta.json are written",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="directory containing dataset.npz and dataset_meta.json",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="trained .pt checkpoint"
    )
    parser.add_argument("--device", default="auto", help="torch device, e.g. cpu or cuda:0")
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples-per-case", type=int, default=10)
    parser.add_argument("--case-batch-size", type=int, default=1)
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tip-success-tolerance", type=float, default=0.1)
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument("--post-correction-step-size", type=float, default=0.1)
    parser.add_argument(
        "--post-correction-repulsion-step-size", type=float, default=5
    )
    parser.add_argument("--post-correction-fraction", type=float, default=0.4)
    parser.add_argument("--guidance-weight", type=float, default=50.0)
    parser.add_argument("--guidance-repulsion-weight", type=float, default=50.0)
    parser.add_argument("--guidance-fraction", type=float, default=0.4)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing eval.npz and eval_meta.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PccEvaluationConfig(
        output_dir=args.output_dir,
        dataset=args.dataset,
        checkpoint=args.checkpoint,
        split=args.split,
        device=args.device,
        samples_per_case=args.samples_per_case,
        case_batch_size=args.case_batch_size,
        ddim_steps=args.ddim_steps,
        eta=args.eta,
        seed=args.seed,
        tip_success_tolerance=args.tip_success_tolerance,
        use_ema_weights=not args.raw_weights,
        post_correction_step_size=args.post_correction_step_size,
        post_correction_repulsion_step_size=(
            args.post_correction_repulsion_step_size
        ),
        post_correction_fraction=args.post_correction_fraction,
        guidance_weight=args.guidance_weight,
        guidance_repulsion_weight=args.guidance_repulsion_weight,
        guidance_fraction=args.guidance_fraction,
        overwrite=args.overwrite,
    )
    metadata = evaluate_model(config, progress=lambda message: print(message, flush=True))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
