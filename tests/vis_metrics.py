"""Plot paired success, terminal-tip error, smoothness, and runtime metrics."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pcc_diffuser.evaluation import EVALUATION_FORMAT_VERSION, METHODS


EVALUATION_DIR = PROJECT_DIR / "runs" / "set_v1"
FIGURE_SIZE = (9.8, 3.5)


def load_evaluation(
    directory: str | Path,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load a current paired evaluation from NPZ and JSON files."""
    directory = Path(directory).expanduser().resolve()
    archive_path = directory / "eval.npz"
    metadata_path = directory / "eval_meta.json"
    if not archive_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"eval.npz/eval_meta.json not found in {directory}")
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if (
        metadata.get("format") != "pcc_diffuser_evaluation"
        or metadata.get("version") != EVALUATION_FORMAT_VERSION
    ):
        raise ValueError(f"unsupported evaluation format in {metadata_path}")
    if metadata.get("split") != "test":
        raise ValueError("vis_metrics.py requires an evaluation of the test split")
    with np.load(archive_path, allow_pickle=False) as archive:
        methods = [str(value) for value in archive["methods"].tolist()]
        errors = np.asarray(archive["terminal_tip_errors"], dtype=np.float64)
        collision_failures = np.asarray(archive["collision_failures"], dtype=bool)
        smoothness = np.asarray(archive["max_configuration_step"], dtype=np.float64)
        method_seconds = np.asarray(archive["method_elapsed_seconds"], dtype=np.float64)
        batch_seconds = np.asarray(
            archive["method_batch_elapsed_seconds"], dtype=np.float64
        )
        batch_path_counts = np.asarray(archive["batch_path_counts"], dtype=np.float64)
    if errors.ndim != 3 or errors.shape[0] != len(methods):
        raise ValueError("terminal_tip_errors must have shape [methods,cases,samples]")
    if not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("terminal tip errors must be finite and non-negative")
    if collision_failures.shape != errors.shape:
        raise ValueError("collision_failures must have shape [methods,cases,samples]")
    if smoothness.shape != errors.shape:
        raise ValueError("max_configuration_step must have shape [methods,cases,samples]")
    if not np.isfinite(smoothness).all() or np.any(smoothness < 0):
        raise ValueError("path smoothness must be finite and non-negative")
    if methods != metadata.get("methods"):
        raise ValueError("method order differs between eval.npz and eval_meta.json")
    if methods != list(METHODS):
        raise ValueError("evaluation uses obsolete method names; rerun evaluate_model.py")
    if method_seconds.shape != (len(methods),) or np.any(method_seconds < 0):
        raise ValueError("method_elapsed_seconds must contain one value per method")
    if batch_seconds.ndim != 2 or batch_seconds.shape[0] != len(methods):
        raise ValueError("method_batch_elapsed_seconds must have shape [methods,batches]")
    if batch_path_counts.shape != (batch_seconds.shape[1],):
        raise ValueError("batch_path_counts must contain one value per batch")
    if np.any(batch_seconds < 0) or np.any(batch_path_counts <= 0):
        raise ValueError("runtime values must be non-negative with positive batch sizes")
    batch_milliseconds_per_path = 1000.0 * batch_seconds / batch_path_counts[None]
    path_milliseconds = np.repeat(
        batch_milliseconds_per_path,
        batch_path_counts.astype(np.int64),
        axis=1,
    )
    expected_path_count = errors.shape[1] * errors.shape[2]
    if path_milliseconds.shape != (len(methods), expected_path_count):
        raise ValueError("batch timing does not cover every evaluated path")
    return (
        methods,
        errors,
        collision_failures,
        smoothness,
        path_milliseconds,
        metadata,
    )


def plot_metrics(directory: str | Path) -> None:
    """Show mean metrics with 5th--95th percentile error bars."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import colormaps
        from matplotlib.colors import to_rgba
        from matplotlib.lines import Line2D
    except ImportError as error:
        raise SystemExit("Matplotlib is required: python -m pip install matplotlib") from error

    (
        methods,
        errors,
        collision_failures,
        smoothness,
        path_runtime,
        metadata,
    ) = load_evaluation(directory)
    try:
        success_tolerance = float(metadata["config"]["tip_success_tolerance"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("eval_meta.json has no valid tip success tolerance") from error
    error_values = errors.reshape(len(methods), -1)
    smoothness_values = smoothness.reshape(len(methods), -1)
    path_successes = (error_values <= success_tolerance) & ~collision_failures.reshape(
        len(methods), -1
    )
    success_means = 100.0 * path_successes.mean(axis=1)

    def successful_statistics(values: np.ndarray) -> tuple[np.ndarray, ...]:
        selected = [
            values[index, path_successes[index]] for index in range(len(methods))
        ]
        if any(value.size == 0 for value in selected):
            raise ValueError("at least one method has no successful planning results")
        return (
            np.asarray([value.mean() for value in selected]),
            np.asarray([np.percentile(value, 5) for value in selected]),
            np.asarray([np.percentile(value, 95) for value in selected]),
        )

    error_means, error_minima, error_maxima = successful_statistics(
        error_values
    )
    smoothness_means, smoothness_minima, smoothness_maxima = successful_statistics(
        smoothness_values
    )
    runtime, runtime_minima, runtime_maxima = successful_statistics(
        path_runtime
    )
    colours = [
        colormaps["Blues"](0.35),
        colormaps["Blues"](0.60),
        colormaps["Blues"](0.85),
    ]

    plt.rcParams.update(
        {
            "font.size": 10,
            "font.weight": "normal",
            "axes.titlesize": 10,
            "axes.titleweight": "normal",
            "axes.labelsize": 10,
            "axes.labelweight": "normal",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    figure, axes = plt.subplots(1, 4, figsize=FIGURE_SIZE, facecolor="white")
    figure.canvas.manager.set_window_title("PCC Diffuser Metrics")
    positions = 0.20 * np.arange(len(methods))
    width = 0.10

    def draw_gradient_bar(
        axis, x: float, top: float, bottom: float, width: float, colour
    ) -> None:
        """Draw a bar reaching full opacity at 50% of visible height."""
        if not np.isfinite(top) or top <= bottom:
            return
        rgba = np.empty((256, 2, 4), dtype=float)
        rgba[..., :3] = to_rgba(colour)[:3]
        fraction = np.linspace(0.0, 1.0, rgba.shape[0])[:, None]
        rgba[..., 3] = np.minimum(fraction / 0.50, 1.0)

        # Convert the bar bounds to axes coordinates so that the alpha ramp
        # follows its rendered height.
        data_corners = np.array(
            [[x - 0.5 * width, bottom], [x + 0.5 * width, top]]
        )
        axes_corners = axis.transAxes.inverted().transform(
            axis.transData.transform(data_corners)
        )
        gradient = axis.imshow(
            rgba,
            origin="lower",
            interpolation="bilinear",
            aspect="auto",
            extent=(
                axes_corners[0, 0],
                axes_corners[1, 0],
                axes_corners[0, 1],
                axes_corners[1, 1],
            ),
            transform=axis.transAxes,
            zorder=6,
            clip_on=True,
        )
        gradient.set_clip_path(axis.patch)

    def draw_metric(
        axis,
        means: np.ndarray,
        minima: np.ndarray,
        maxima: np.ndarray,
        label: str,
    ) -> None:
        bottom = 0.0
        upper = max(float(np.max(np.maximum(means, maxima))) * 1.10, 1e-8)
        axis.set_xlim(positions[0] - 0.18, positions[-1] + 0.18)
        axis.set_ylim(bottom, upper)
        for position, mean, colour in zip(positions, means, colours):
            draw_gradient_bar(axis, float(position), float(mean), bottom, width, colour)
        axis.bar(
            positions,
            means - bottom,
            bottom=bottom,
            width=width,
            color="none",
            edgecolor="none",
            linewidth=0.0,
            zorder=7,
        )
        cap_half_width = 0.025
        axis.vlines(positions, minima, maxima, color="#000000", linewidth=0.8, zorder=8)
        axis.hlines(
            minima,
            positions - cap_half_width,
            positions + cap_half_width,
            color="#000000",
            linewidth=0.8,
            zorder=8,
        )
        axis.hlines(
            maxima,
            positions - cap_half_width,
            positions + cap_half_width,
            color="#000000",
            linewidth=0.8,
            zorder=8,
        )
        axis.set_xticks([])
        axis.set_ylabel(label)
        axis.set_facecolor("none")
        axis.grid(False)
        axis.minorticks_off()
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#000000")
            axis.spines[side].set_linewidth(0.8)
        axis.tick_params(colors="#000000", width=0.7, length=3, direction="out")

    success_axis = axes[0]
    success_axis.plot(
        positions,
        success_means,
        "-",
        color="#000000",
        linewidth=0.8,
        zorder=9,
    )
    for position, success, colour in zip(positions, success_means, colours):
        success_axis.plot(
            position,
            success,
            marker="s",
            linestyle="none",
            markersize=5.5,
            markerfacecolor=colour,
            markeredgecolor="#000000",
            markeredgewidth=0.8,
            zorder=10,
        )
    success_axis.set_xlim(positions[0] - 0.18, positions[-1] + 0.18)
    # success_axis.set_ylim(90.0, 100.0)
    success_axis.set_xticks([])
    # success_axis.set_yticks([90.0, 95.0, 100.0])
    success_axis.set_ylabel("Success Rate (%)")
    success_axis.set_facecolor("none")
    success_axis.grid(False)
    success_axis.minorticks_off()
    success_axis.spines["top"].set_visible(False)
    success_axis.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        success_axis.spines[side].set_color("#000000")
        success_axis.spines[side].set_linewidth(0.8)
    success_axis.tick_params(
        colors="#000000", width=0.7, length=3, direction="out"
    )

    draw_metric(
        axes[1],
        error_means,
        error_minima,
        error_maxima,
        "Tip Error (m)"
    )
    draw_metric(
        axes[2],
        smoothness_means,
        smoothness_minima,
        smoothness_maxima,
        "Max Configuration Step (rad)",
    )
    draw_metric(
        axes[3],
        runtime,
        runtime_minima,
        runtime_maxima,
        "Denoising Time (ms)",
    )
    legend_handles = [
        Line2D(
            [], [], linestyle="none", marker="s", markersize=6,
            markerfacecolor=colours[index],
            markeredgecolor="none",
            label=method,
        )
        for index, method in enumerate(methods)
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
    )
    figure.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.24, wspace=0.70)

    print(
        f"Loaded {metadata['cases']} test cases, each with "
        f"{metadata['samples_per_case']} samples\n"
    )
    for index, method in enumerate(methods):
        print(
            f"{method}:\n"
            f"  Success Rate {success_means[index]:.6g}%\n"
            f"  Tip Error {error_means[index]:.6g} m "
            f"[P5 {error_minima[index]:.6g}, P95 {error_maxima[index]:.6g}]\n"
            f"  Max Configuration Step {smoothness_means[index]:.6g} rad "
            f"[P5 {smoothness_minima[index]:.6g}, P95 {smoothness_maxima[index]:.6g}]\n"
            f"  Denoising Time {runtime[index]:.6g} ms "
            f"[P5 {runtime_minima[index]:.6g}, P95 {runtime_maxima[index]:.6g}]\n"
        )
    plt.show()


if __name__ == "__main__":
    plot_metrics(EVALUATION_DIR)
