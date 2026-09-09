"""Plot diffusion training curves from one or more training runs.

Edit ``RUN_DIRECTORIES`` below, then run ``python tests/eval_training.py``
from ``src/pcc_diffuser``.  One run is plotted as a single curve.  Multiple
runs are aligned by optimiser step and shown as mean +/- one standard
deviation across random seeds.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pcc_diffuser.training import TRAINING_LOG_FORMAT_VERSION

# Add one directory per independently trained random seed.  Every directory
# must contain log.npz and log_meta.json written by pcc_diffuser.training.
RUN_DIRECTORIES = [
    # PROJECT_DIR / "runs" / "set_v1",
    PROJECT_DIR / "runs" / "set_v2",
]

SMOOTHING_WINDOW = 1  # Number of logged points; 1 disables smoothing.
SHOW_INDIVIDUAL_RUNS = True
FIGURE_SIZE = (5.2, 3.4)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Apply a centred moving average without changing array length."""
    values = np.asarray(values, dtype=np.float64)
    if window < 1:
        raise ValueError("SMOOTHING_WINDOW must be positive")
    if window == 1 or len(values) < 2:
        return values.copy()
    window = min(int(window), len(values))
    left = (window - 1) // 2
    right = window // 2
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def load_run(directory: str | Path) -> dict[str, Any]:
    """Load and validate one numeric training history and its metadata."""
    directory = Path(directory).expanduser().resolve()
    log_path = directory / "log.npz"
    metadata_path = directory / "log_meta.json"
    if not log_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"log.npz/log_meta.json not found in {directory}")
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if (
        metadata.get("format") != "pcc_diffuser_training_log"
        or metadata.get("version") != TRAINING_LOG_FORMAT_VERSION
    ):
        raise ValueError(f"unsupported training log format in {metadata_path}")
    with np.load(log_path, allow_pickle=False) as archive:
        required = {"step", "diffusion_loss"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{log_path} is missing arrays: {missing}")
        steps = np.asarray(archive["step"], dtype=np.int64)
        losses = np.asarray(archive["diffusion_loss"], dtype=np.float64)
    if (
        steps.ndim != 1
        or losses.shape != steps.shape
        or len(steps) == 0
        or np.any(np.diff(steps) <= 0)
        or not np.isfinite(losses).all()
    ):
        raise ValueError(f"invalid step/loss arrays in {log_path}")
    return {
        "directory": directory,
        "steps": steps,
        "losses": moving_average(losses, SMOOTHING_WINDOW),
        "seed": metadata.get("random_seed", metadata.get("training_config", {}).get("seed")),
        "complete": bool(metadata.get("complete", False)),
    }


def align_runs(runs: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate runs onto common logged steps over their shared range."""
    if not runs:
        raise ValueError("RUN_DIRECTORIES must contain at least one run")
    lower = max(int(run["steps"][0]) for run in runs)
    upper = min(int(run["steps"][-1]) for run in runs)
    if lower > upper:
        raise ValueError("training runs do not share an overlapping step range")
    common_steps = np.unique(
        np.concatenate(
            [run["steps"][(run["steps"] >= lower) & (run["steps"] <= upper)] for run in runs]
        )
    )
    if len(common_steps) == 0:
        raise ValueError("training runs have no common plotting steps")
    aligned = np.stack(
        [np.interp(common_steps, run["steps"], run["losses"]) for run in runs]
    )
    return common_steps, aligned


def plot_training_curves(run_directories: Sequence[str | Path]) -> None:
    """Display training-loss curves in an interactive Matplotlib window."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        from matplotlib.ticker import MaxNLocator
    except ImportError as error:
        raise SystemExit("Matplotlib is required: python -m pip install matplotlib") from error

    runs = [load_run(directory) for directory in run_directories]
    steps, losses = align_runs(runs)
    mean = losses.mean(axis=0)
    deviation = losses.std(axis=0, ddof=1) if len(runs) > 1 else None

    plt.rcParams.update(
        {
            "font.size": 10,
            "font.weight": "normal",
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, facecolor="white")
    figure.canvas.manager.set_window_title("PCC Diffuser Training Curve")
    line_colour = colormaps["Blues"](0.85)
    individual_colour = colormaps["Blues"](0.60)
    shading_colour = colormaps["Blues"](0.35)
    if SHOW_INDIVIDUAL_RUNS and len(runs) > 1:
        for run in runs:
            label = "Individual Runs" if run is runs[0] else None
            axis.plot(
                run["steps"],
                run["losses"],
                color=individual_colour,
                alpha=0.28,
                linewidth=0.8,
                label=label,
            )
    axis.plot(
        steps,
        mean,
        color=line_colour,
        linewidth=1.5,
        label="Mean" if len(runs) > 1 else f"Seed {runs[0]['seed']}",
    )
    if deviation is not None:
        axis.fill_between(
            steps,
            mean - deviation,
            mean + deviation,
            color=shading_colour,
            alpha=0.28,
            linewidth=0.0,
            label="Mean +/- One Standard Deviation",
        )
    axis.set_xlabel("Training Step")
    axis.set_ylabel("Diffusion Loss")
    axis.set_facecolor("none")
    axis.grid(False)
    axis.xaxis.set_major_locator(MaxNLocator(6, integer=True))
    axis.yaxis.set_major_locator(MaxNLocator(5))
    axis.minorticks_off()
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#000000")
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(
        colors="#000000", width=0.7, length=3, direction="out"
    )
    axis.legend(loc="upper right", frameon=False)
    figure.tight_layout()

    print(f"Loaded {len(runs)} training run(s)")
    for run in runs:
        status = "complete" if run["complete"] else "incomplete"
        print(
            f"  seed {run['seed']}: {len(run['steps'])} logged points, "
            f"step {run['steps'][0]} to {run['steps'][-1]} ({status})"
        )
    plt.show()


if __name__ == "__main__":
    plot_training_curves(RUN_DIRECTORIES)
