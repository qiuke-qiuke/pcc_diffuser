"""Compare one diffusion method with unified planning-benchmark results."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from benchmarks.evaluation import BENCHMARK_FORMAT_VERSION
from pcc_diffuser.evaluation import EVALUATION_FORMAT_VERSION


DIFFUSION_EVALUATION_DIR = PROJECT_DIR / "runs" / "set_v1"
DIFFUSION_METHOD = "With Post Correction"
DIFFUSION_LABEL = "PccDiffuser"
BENCHMARK_EVALUATION_DIR = PROJECT_DIR / "runs" / "set_v1_benchmarks"
FIGURE_SIZE = (12.65, 3.5)

BENCHMARK_LABELS = {
    "c-rrt": "Configuration-Space RRT",
    "c-rrt-star": "Configuration-Space RRT*",
    "w-rrt": "Workspace RRT",
    "w-rrt-star": "Workspace RRT*",
    "repulsion": "Artificial Potential Field",
}


def _load_metadata(path: Path, expected_format: str, version: int) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("format") != expected_format or metadata.get("version") != version:
        raise ValueError(f"unsupported result format in {path}")
    return metadata


def _diffusion_result(directory: str | Path) -> dict[str, object]:
    directory = Path(directory).expanduser().resolve()
    metadata = _load_metadata(
        directory / "eval_meta.json",
        "pcc_diffuser_evaluation",
        EVALUATION_FORMAT_VERSION,
    )
    archive_path = directory / "eval.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        methods = [str(value) for value in archive["methods"].tolist()]
        if methods != metadata.get("methods"):
            raise ValueError("diffusion method order differs between NPZ and metadata")
        if DIFFUSION_METHOD not in methods:
            raise ValueError(
                f"unknown diffusion method {DIFFUSION_METHOD!r}; choose from {methods}"
            )
        method_index = methods.index(DIFFUSION_METHOD)
        errors = np.asarray(archive["terminal_tip_errors"][method_index], dtype=np.float64)
        collisions = np.asarray(archive["collision_failures"][method_index], dtype=bool)
        batch_seconds = np.asarray(
            archive["method_batch_elapsed_seconds"][method_index], dtype=np.float64
        )
        batch_counts = np.asarray(archive["batch_path_counts"], dtype=np.int64)
        runtime = np.repeat(1000.0 * batch_seconds / batch_counts, batch_counts)
        target_indices = np.asarray(archive["target_indices"])
        start_indices = np.asarray(archive["start_indices"])
        target_tips = np.asarray(archive["target_tips"])
    expected = (int(metadata["cases"]), int(metadata["samples_per_case"]))
    if errors.shape != expected or collisions.shape != expected:
        raise ValueError("diffusion metric arrays have inconsistent shapes")
    if runtime.shape != (int(np.prod(expected)),):
        raise ValueError("diffusion batch timing does not cover every path")
    if not np.isfinite(errors).all() or np.any(errors < 0):
        raise ValueError("diffusion terminal errors must be finite and non-negative")
    return {
        "methods": [DIFFUSION_LABEL],
        "errors": errors[None],
        "collisions": collisions[None],
        "planned": np.ones((1,) + expected, dtype=bool),
        "runtime": runtime.reshape((1,) + expected),
        "target_indices": target_indices,
        "start_indices": start_indices,
        "target_tips": target_tips,
        "metadata": metadata,
    }


def _benchmark_result(directory: str | Path) -> dict[str, object]:
    directory = Path(directory).expanduser().resolve()
    metadata = _load_metadata(
        directory / "benchmark_meta.json",
        "pcc_diffuser_benchmark",
        BENCHMARK_FORMAT_VERSION,
    )
    archive_path = directory / "benchmark.npz"
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        methods = [str(value) for value in archive["methods"].tolist()]
        errors = np.asarray(archive["terminal_tip_errors"], dtype=np.float64)
        collisions = np.asarray(archive["collision_failures"], dtype=bool)
        planned = np.asarray(archive["planning_success"], dtype=bool)
        runtime = 1000.0 * np.asarray(
            archive["attempt_elapsed_seconds"], dtype=np.float64
        )
        target_indices = np.asarray(archive["target_indices"])
        start_indices = np.asarray(archive["start_indices"])
        target_tips = np.asarray(archive["target_tips"])
    expected = (
        len(methods),
        int(metadata["cases"]),
        int(metadata["samples_per_case"]),
    )
    if methods != metadata.get("methods"):
        raise ValueError("benchmark method order differs between NPZ and metadata")
    if any(array.shape != expected for array in (errors, collisions, planned, runtime)):
        raise ValueError("benchmark metric arrays have inconsistent shapes")
    if not np.isfinite(runtime).all() or np.any(runtime < 0):
        raise ValueError("benchmark runtimes must be finite and non-negative")
    if not np.isfinite(errors[planned]).all() or np.any(errors[planned] < 0):
        raise ValueError("planned benchmark paths must have finite terminal errors")
    if not np.isnan(errors[~planned]).all():
        raise ValueError("failed benchmark attempts must have NaN terminal errors")
    return {
        "methods": methods,
        "errors": errors,
        "collisions": collisions,
        "planned": planned,
        "runtime": runtime,
        "target_indices": target_indices,
        "start_indices": start_indices,
        "target_tips": target_tips,
        "metadata": metadata,
    }


def load_paired_results() -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    diffusion = _diffusion_result(DIFFUSION_EVALUATION_DIR)
    benchmark = _benchmark_result(BENCHMARK_EVALUATION_DIR)
    diffusion_metadata = diffusion["metadata"]
    benchmark_metadata = benchmark["metadata"]
    for field in ("dataset", "split", "cases", "samples_per_case", "device"):
        if diffusion_metadata.get(field) != benchmark_metadata.get(field):
            print(
                f"Warning: diffusion and benchmark {field} values differ: "
                f"{diffusion_metadata.get(field)!r} != "
                f"{benchmark_metadata.get(field)!r}"
            )
    for field in ("target_indices", "start_indices"):
        if not np.array_equal(diffusion[field], benchmark[field]):
            print(f"Warning: diffusion and benchmark {field} differ")
    target_tips_match = (
        diffusion["target_tips"].shape == benchmark["target_tips"].shape
        and np.allclose(
            diffusion["target_tips"], benchmark["target_tips"], atol=0.0
        )
    )
    if not target_tips_match:
        print("Warning: diffusion and benchmark target tips differ")
    diffusion_tolerance = float(diffusion_metadata["config"]["tip_success_tolerance"])
    benchmark_tolerance = float(benchmark_metadata["tip_success_tolerance"])
    if diffusion_tolerance != benchmark_tolerance:
        print(
            "Warning: diffusion and benchmark tip success tolerances differ: "
            f"{diffusion_tolerance!r} != {benchmark_tolerance!r}; "
            "using the diffusion tolerance"
        )

    methods = list(diffusion["methods"]) + [
        BENCHMARK_LABELS.get(method, method) for method in benchmark["methods"]
    ]
    errors = np.concatenate((diffusion["errors"], benchmark["errors"]), axis=0)
    collisions = np.concatenate(
        (diffusion["collisions"], benchmark["collisions"]), axis=0
    )
    planned = np.concatenate((diffusion["planned"], benchmark["planned"]), axis=0)
    runtime = np.concatenate((diffusion["runtime"], benchmark["runtime"]), axis=0)
    successes = planned & (errors <= diffusion_tolerance) & ~collisions
    return methods, successes, errors, runtime


def plot_comparison() -> None:
    """Plot paired success, terminal error, and amortized planning runtime."""
    methods, successes, errors, runtime = load_paired_results()
    success_rates = 100.0 * successes.mean(axis=(1, 2))

    def successful_statistics(values: np.ndarray) -> tuple[np.ndarray, ...]:
        selected = [values[index][successes[index]] for index in range(len(methods))]
        if any(not len(value) for value in selected):
            raise ValueError("at least one method has no successful planning results")
        return (
            np.asarray([value.mean() for value in selected]),
            np.asarray([np.percentile(value, 5) for value in selected]),
            np.asarray([np.percentile(value, 95) for value in selected]),
        )

    error_means, error_minima, error_maxima = successful_statistics(errors)
    runtime_means, runtime_minima, runtime_maxima = successful_statistics(runtime)
    colours = [
        colormaps["Blues"](0.85),] + [
        colormaps["Reds"](value)
        for value in np.linspace(0.35, 0.85, len(methods) - 1)
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
    figure, axes = plt.subplots(1, 3, figsize=FIGURE_SIZE, facecolor="white")
    figure.canvas.manager.set_window_title("PCC Planner Comparison")
    bar_spacing = 0.20
    width = 0.10
    axis_padding = 0.18
    positions = bar_spacing * np.arange(len(methods))

    def style_axis(axis) -> None:
        axis.set_xticks([])
        axis.set_facecolor("none")
        axis.grid(False)
        axis.minorticks_off()
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#000000")
            axis.spines[side].set_linewidth(0.8)
        axis.tick_params(colors="#000000", width=0.7, length=3, direction="out")

    def draw_gradient_bar(
        axis, x: float, bottom: float, top: float, colour
    ) -> None:
        if not np.isfinite(top) or top <= 0:
            return
        rgba = np.empty((256, 2, 4), dtype=float)
        rgba[..., :3] = to_rgba(colour)[:3]
        rgba[..., 3] = np.minimum(
            np.linspace(0.0, 1.0, rgba.shape[0])[:, None] / 0.50, 1.0
        )
        corners = np.array(
            [[x - 0.5 * width, bottom], [x + 0.5 * width, top]]
        )
        axes_corners = axis.transAxes.inverted().transform(
            axis.transData.transform(corners)
        )
        gradient = axis.imshow(
            rgba,
            origin="lower",
            interpolation="bilinear",
            aspect="auto",
            extent=(
                axes_corners[0, 0], axes_corners[1, 0],
                axes_corners[0, 1], axes_corners[1, 1],
            ),
            transform=axis.transAxes,
            zorder=6,
            clip_on=True,
        )
        gradient.set_clip_path(axis.patch)

    def draw_metric(
        axis, means, minima, maxima, label: str, *, logarithmic: bool = False
    ) -> None:
        if logarithmic:
            values = np.concatenate((means, minima, maxima))
            if np.any(values <= 0):
                raise ValueError(f"{label} values must be positive for log scaling")
            lower = float(values.min()) / 1.5
            upper = float(values.max()) * 1.5
            axis.set_yscale("log")
        else:
            lower = 0.0
            upper = max(float(np.max(np.maximum(means, maxima))) * 1.10, 1e-8)
        axis.set_xlim(
            positions[0] - axis_padding,
            positions[-1] + axis_padding,
        )
        axis.set_ylim(lower, upper)
        for position, mean, colour in zip(positions, means, colours):
            draw_gradient_bar(axis, float(position), lower, float(mean), colour)
        axis.bar(
            positions, means - lower, bottom=lower, width=width,
            color="none", edgecolor="none",
        )
        axis.vlines(positions, minima, maxima, color="#000000", linewidth=0.8, zorder=8)
        axis.hlines(
            minima, positions - 0.025, positions + 0.025,
            color="#000000", linewidth=0.8, zorder=8,
        )
        axis.hlines(
            maxima, positions - 0.025, positions + 0.025,
            color="#000000", linewidth=0.8, zorder=8,
        )
        axis.set_ylabel(label)
        style_axis(axis)

    success_axis = axes[0]
    success_axis.plot(positions, success_rates, "-", color="#000000", linewidth=0.8)
    for position, success, colour in zip(positions, success_rates, colours):
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
    success_axis.set_xlim(
        positions[0] - axis_padding,
        positions[-1] + axis_padding,
    )
    success_axis.set_ylabel("Success Rate (%)")
    style_axis(success_axis)
    draw_metric(
        axes[1], error_means, error_minima, error_maxima,
        "Tip Error (m)", logarithmic=True,
    )
    draw_metric(
        axes[2],
        runtime_means,
        runtime_minima,
        runtime_maxima,
        "Denoising/Planning Time (ms)",
        logarithmic=True,
    )

    legend_handles = [
        Line2D(
            [], [], linestyle="none", marker="s", markersize=6,
            markerfacecolor=colour, markeredgecolor="none", label=method,
        )
        for method, colour in zip(methods, colours)
    ]
    columns = (len(legend_handles) + 1) // 2
    column_major_handles = [
        legend_handles[index]
        for column in range(columns)
        for index in (column, column + columns)
        if index < len(legend_handles)
    ]
    figure.legend(
        handles=column_major_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=columns,
        frameon=False,
        handlelength=0.8,
        handletextpad=0.5,
        columnspacing=2.0,
    )
    figure.subplots_adjust(left=0.07, right=0.99, top=0.96, bottom=0.24, wspace=0.70)

    print(f"Compared {len(methods)} methods on {successes.shape[1]} cases")
    for index, method in enumerate(methods):
        print(
            f"{method}:\n"
            f"  Success Rate {success_rates[index]:.6g}%\n"
            f"  Tip Error {error_means[index]:.6g} m "
            f"[P5 {error_minima[index]:.6g}, P95 {error_maxima[index]:.6g}]\n"
            f"  Denoising/Planning Time {runtime_means[index]:.6g} ms "
            f"[P5 {runtime_minima[index]:.6g}, P95 {runtime_maxima[index]:.6g}]\n"
        )
    plt.show()


if __name__ == "__main__":
    plot_comparison()
