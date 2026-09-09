"""Interactive 3-D viewer for generated PCC path datasets.

Each PCC section is drawn as its sampled centreline arc, section
boundaries are marked, and transforms are composed by the same
product-of-exponentials forward kinematics used by training.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
from matplotlib.ticker import MaxNLocator, NullLocator
import numpy as np
import torch

# Permit running directly from a source checkout without installation.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pcc_diffuser.data import DATASET_FORMAT_VERSION
from pcc_diffuser.kinematics import backbone_points, sphere_clearances_and_tips


DATASET_DIR = PROJECT_DIR / "data" / "set_v1"

BACKBONE_POINTS_PER_SECTION = 20
FPS = 30.0

def load_visualisation_data(
    dataset: str | Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    tuple[float, float, float],
    float,
    float,
]:
    """Load the dense NPZ dataset without changing its stored frame count."""
    root = Path(dataset).expanduser().resolve()
    metadata_path = root / "dataset_meta.json"
    archive_path = root / "dataset.npz"
    if not metadata_path.is_file() or not archive_path.is_file():
        raise FileNotFoundError(f"dataset.npz/dataset_meta.json not found in {root}")
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if (
        metadata.get("format") != "pcc_diffuser_npz"
        or metadata.get("version") != DATASET_FORMAT_VERSION
        or not metadata.get("complete")
    ):
        raise ValueError(f"unsupported or incomplete dataset: {metadata_path}")
    lengths_value = metadata.get("section_lengths")
    if lengths_value is None:
        raise ValueError("dataset metadata has no section_lengths")
    lengths = tuple(float(value) for value in lengths_value)
    if len(lengths) != 3 or not np.isfinite(lengths).all() or any(
        length <= 0 for length in lengths
    ):
        raise ValueError("dataset section_lengths must contain three positive values")
    sphere_radius = float(metadata["sphere_radius"])
    clearance_margin = float(metadata["clearance_margin"])
    with np.load(archive_path, allow_pickle=False) as archive:
        paths = np.asarray(archive["paths"], dtype=np.float32)
        targets = np.asarray(archive["target_tips"], dtype=np.float32)
        target_indices = np.asarray(archive["target_indices"], dtype=np.int64)
        start_indices = np.asarray(archive["start_indices"], dtype=np.int64)
        ik_indices = np.asarray(archive["ik_indices"], dtype=np.int64)
        split_indices = np.asarray(archive["split_indices"], dtype=np.int64)
        obstacle_centres = np.asarray(archive["obstacle_centres"], dtype=np.float32)
        obstacle_offsets = np.asarray(archive["obstacle_offsets"], dtype=np.int64)
    requested_targets = metadata.get("requested_targets")
    actual_targets = metadata.get("actual_targets")
    requested_paths = metadata.get("requested_paths")
    actual_paths = metadata.get("actual_paths")
    if (
        not isinstance(requested_targets, int)
        or not isinstance(actual_targets, int)
        or not 0 <= actual_targets <= requested_targets
        or not isinstance(requested_paths, int)
        or not 0 < len(paths) <= requested_paths
    ):
        raise ValueError("dataset target/path counts are invalid")
    path_failure_counts = metadata.get("path_failure_counts")
    target_discarded_counts = metadata.get("target_discarded_counts")
    generation_config = metadata.get("config")
    if (
        not isinstance(target_discarded_counts, dict)
        or any(
            not isinstance(value, int) or value < 0
            for value in target_discarded_counts.values()
        )
        or target_discarded_counts.get("total")
        != requested_targets - actual_targets
        or target_discarded_counts.get("total")
        != sum(
            value
            for reason, value in target_discarded_counts.items()
            if reason != "total"
        )
    ):
        raise ValueError("dataset target discard counts are invalid")
    if not isinstance(generation_config, dict):
        raise ValueError("dataset generation config is missing")
    start_count = generation_config.get("start_count")
    terminal_count = generation_config.get("terminal_count")
    if not isinstance(start_count, int) or not isinstance(terminal_count, int):
        raise ValueError("dataset start/terminal counts are invalid")
    unplanned_paths = start_count * terminal_count * (
        target_discarded_counts.get("terminal_ik", 0)
        + target_discarded_counts.get("start_sampling", 0)
    )
    if (
        not isinstance(path_failure_counts, dict)
        or any(
            not isinstance(value, int) or value < 0
            for value in path_failure_counts.values()
        )
        or path_failure_counts.get("total")
        != requested_paths - len(paths) - unplanned_paths
        or path_failure_counts.get("total")
        != sum(
            value
            for reason, value in path_failure_counts.items()
            if reason != "total"
        )
    ):
        raise ValueError("dataset failure counts do not explain missing paths")
    if actual_paths != len(paths):
        raise ValueError("dataset metadata and NPZ path counts differ")
    if actual_targets != len(np.unique(target_indices)):
        raise ValueError("dataset metadata and NPZ target counts differ")
    count = len(paths)
    if paths.ndim != 3 or paths.shape[2] != 6 or targets.shape != (count, 3):
        raise ValueError("dataset arrays have incompatible shapes")
    code_to_split = {
        int(code): name for name, code in metadata.get("split_codes", {}).items()
    }
    group_counts: dict[tuple[int, int], int] = {}
    for target_index, start_index in zip(target_indices, start_indices):
        key = (int(target_index), int(start_index))
        group_counts[key] = group_counts.get(key, 0) + 1
    order = np.lexsort((ik_indices, start_indices, target_indices))
    records = [
        {
            "target_id": f"target{int(target_indices[index]):05d}",
            "start_id": int(start_indices[index]),
            "ik_id": int(ik_indices[index]),
            "ik_count": group_counts[
                (int(target_indices[index]), int(start_indices[index]))
            ],
            "split": code_to_split[int(split_indices[index])],
            "obstacle_centres": obstacle_centres[
                obstacle_offsets[index] : obstacle_offsets[index + 1]
            ],
        }
        for index in order
    ]
    if np.array_equal(order, np.arange(count)):
        return paths, targets, records, lengths, sphere_radius, clearance_margin
    return (
        paths[order], targets[order], records, lengths,
        sphere_radius, clearance_margin,
    )


def compute_path_geometry(
    path: np.ndarray,
    section_lengths: Sequence[float],
) -> np.ndarray:
    """Compute one path's robot centrelines on demand."""
    configurations = torch.as_tensor(path, dtype=torch.float64)
    with torch.no_grad():
        points = backbone_points(
            configurations,
            section_lengths,
            BACKBONE_POINTS_PER_SECTION,
        )
    return points.cpu().numpy()


class DatasetViewer:
    """Matplotlib controller that updates existing artists instead of redrawing."""

    SECTION_COLOURS = ("#111111", "#111111", "#111111")

    def __init__(
        self,
        paths: np.ndarray,
        targets: np.ndarray,
        records: list[dict[str, Any]],
        section_lengths: Sequence[float],
        sphere_radius: float,
        clearance_margin: float,
        fps: float,
    ) -> None:
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
                "legend.fontsize": 10,
            }
        )
        # Reserve these keys for viewer navigation. Matplotlib otherwise maps
        # Left/Right to toolbar history and Home to resetting the axes, which
        # can unexpectedly restore a different 3-D camera view.
        plt.rcParams["keymap.back"] = [
            key for key in plt.rcParams["keymap.back"] if key != "left"
        ]
        plt.rcParams["keymap.forward"] = [
            key for key in plt.rcParams["keymap.forward"] if key != "right"
        ]
        plt.rcParams["keymap.home"] = [
            key for key in plt.rcParams["keymap.home"] if key != "home"
        ]
        self.plt = plt
        self.paths = paths
        self.targets = targets
        self.records = records
        self.section_lengths = tuple(float(value) for value in section_lengths)
        self.sphere_radius = float(sphere_radius)
        self.clearance_margin = float(clearance_margin)
        self.obstacle_surfaces: list[Any] = []
        self.top_obstacles: list[Any] = []
        self.count, self.horizon = paths.shape[:2]
        self._indices_by_target: dict[tuple[str, int], list[int]] = {}
        self._target_case: dict[tuple[str, int], int] = {}
        for index, record in enumerate(records):
            condition = (record["target_id"], record["start_id"])
            self._indices_by_target.setdefault(condition, []).append(index)
        for case_index, condition in enumerate(self._indices_by_target):
            self._target_case[condition] = case_index
        self.path_index = 0
        self.frame_index = 0
        self.requested_path_index = 0
        self.requested_frame_index = 0
        self.current_geometry: np.ndarray | None = None
        self.current_minimum_clearance: float | None = None
        self._active_slider: Any | None = None
        self.playing = False
        self._orbit_drag: tuple[float, float, float, float] | None = None

        figure_width = 15.0
        figure_height = 7.0
        plot_size = 5.0
        plot_slider_gap = 0.45
        slider_height = 0.20
        slider_gap = 0.10
        instruction_gap = 0.35
        plot_2d_size = 2.35
        legend_top_inset = 0.10
        x_label_offset = 0.45
        label_plot_gap = 0.45
        status_left = 0.02
        status_width = 0.18
        plot_left = 0.10
        plot_width = 0.65
        plot_2d_left = 0.75
        plot_2d_width = 0.25
        slider_right = 0.70
        outer_margin = 0.5 * (
            figure_height
            - plot_size
            - plot_slider_gap
            - 2.0 * slider_height
            - slider_gap
            - instruction_gap
        )
        plot_bottom = outer_margin + instruction_gap + 2.0 * slider_height
        plot_bottom += slider_gap + plot_slider_gap
        plot_top = plot_bottom + plot_size
        slider_bottoms = (
            plot_bottom - plot_slider_gap - slider_height,
            plot_bottom - plot_slider_gap - 2.0 * slider_height - slider_gap,
        )

        self.figure = plt.figure(
            figsize=(figure_width, figure_height), facecolor="white"
        )
        self.figure.canvas.manager.set_window_title("PCC Dataset Viewer")
        self.axis = self.figure.add_axes(
            [
                plot_left,
                plot_bottom / figure_height,
                plot_width,
                plot_size / figure_height,
            ],
            projection="3d",
            computed_zorder=False,
        )
        self.info_axis = self.figure.add_axes(
            [status_left, outer_margin / figure_height, status_width, 0.70]
        )
        self.info_axis.axis("off")
        top_2d_bottom = plot_top - legend_top_inset - plot_2d_size
        config_2d_bottom = (
            top_2d_bottom - x_label_offset - label_plot_gap - plot_2d_size
        )
        self.top_axis = self.figure.add_axes(
            [
                plot_2d_left,
                top_2d_bottom / figure_height,
                plot_2d_width,
                plot_2d_size / figure_height,
            ]
        )
        self.config_axis = self.figure.add_axes(
            [
                plot_2d_left,
                config_2d_bottom / figure_height,
                plot_2d_width,
                plot_2d_size / figure_height,
            ]
        )

        self._configure_3d_axis()
        self._configure_top_axis()
        self._create_context_artists()
        self._create_robot_artists()
        self._create_configuration_plot()

        plot_position = self.axis.get_position()
        slider_width = slider_right - plot_position.x0
        path_axis = self.figure.add_axes(
            [
                plot_position.x0,
                slider_bottoms[0] / figure_height,
                slider_width,
                slider_height / figure_height,
            ]
        )
        frame_axis = self.figure.add_axes(
            [
                plot_position.x0,
                slider_bottoms[1] / figure_height,
                slider_width,
                slider_height / figure_height,
            ]
        )
        self.path_slider = Slider(
            path_axis,
            "Path",
            0,
            self.count - 1,
            valinit=0,
            valstep=1,
            valfmt="%0.0f",
        )
        self.frame_slider = Slider(
            frame_axis,
            "Frame",
            0,
            self.horizon - 1,
            valinit=0,
            valstep=1,
            valfmt="%0.0f",
        )
        self.path_slider.on_changed(self._path_changed)
        self.frame_slider.on_changed(self._frame_changed)

        interval = max(1, int(round(1000.0 / fps)))
        self.timer = self.figure.canvas.new_timer(interval=interval)
        self.timer.add_callback(self._advance_playback)
        self.figure.canvas.mpl_connect("key_press_event", self._key_pressed)
        self.figure.canvas.mpl_connect("button_press_event", self._mouse_pressed)
        self.figure.canvas.mpl_connect("button_release_event", self._mouse_released)
        self.figure.canvas.mpl_connect("motion_notify_event", self._mouse_moved)
        self.figure.text(
            0.5 * (plot_position.x0 + slider_right),
            outer_margin / figure_height,
            "Space: Play/Pause  |  Left/Right: Frame  |  Up/Down: Path",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="normal",
            color="#475569",
        )
        self.status_labels = self.figure.text(
            0.0, 0.0, "", ha="left", va="bottom", fontsize=10,
            fontweight="normal", linespacing=2.0, color="#475569",
            transform=self.info_axis.transAxes,
        )
        self.status_values = self.figure.text(
            0.44, 0.0, "", ha="left", va="bottom", fontsize=10,
            fontweight="normal", linespacing=2.0, color="#475569",
            transform=self.info_axis.transAxes,
        )
        self.update(path_changed=True)
        self._standardise_text()

    def _configure_3d_axis(self) -> None:
        self.axis.set_xlabel(r"$x$ (m)")
        self.axis.set_ylabel(r"$y$ (m)")
        self.axis.set_zlabel(r"$z$ (m)")
        self.axis.grid(True)
        self.axis.set_facecolor("none")
        for coordinate_axis in (self.axis.xaxis, self.axis.yaxis, self.axis.zaxis):
            coordinate_axis.pane.fill = False
            coordinate_axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
            coordinate_axis.set_major_locator(MaxNLocator(5))
            coordinate_axis.set_minor_locator(NullLocator())
            coordinate_axis.line.set_color("#000000")
            coordinate_axis._axinfo["grid"].update(
                color=(0.87, 0.89, 0.86, 0.65), linewidth=0.6
            )
        self.axis.tick_params(
            direction="out", colors="#000000", width=0.7, length=3, pad=2
        )
        for axis in (self.axis.xaxis, self.axis.yaxis, self.axis.zaxis):
            axis._axinfo["tick"]["inward_factor"] = 0.2
            axis._axinfo["tick"]["outward_factor"] = 0.0
        self.axis.set_proj_type("ortho")
        self.axis.view_init(elev=24, azim=-56, roll=0.0)
        # Native Matplotlib 3-D mouse controls vary between backends. Disable
        # them and use a fixed yaw/pitch controller so left-drag never rolls.
        if hasattr(self.axis, "disable_mouse_rotation"):
            self.axis.disable_mouse_rotation()
        total_length = float(sum(self.section_lengths))
        margin = 0.1 * total_length
        self.axis.set_xlim(-total_length - margin, total_length + margin)
        self.axis.set_ylim(-total_length - margin, total_length + margin)
        self.axis.set_zlim(-margin, total_length + margin)
        section_boundaries = np.concatenate(
            ([0.0], np.cumsum(self.section_lengths, dtype=float))
        )
        self.axis.set_zticks(section_boundaries)
        self.axis.set_box_aspect(
            (2.0 * (total_length + margin),) * 2
            + (total_length + 2.0 * margin,)
        )
        origin = np.zeros(3)
        axis_length = 0.24 * total_length
        self.axis.quiver(*origin, axis_length, 0, 0, color="#dc2626", linewidth=1.5)
        self.axis.quiver(*origin, 0, axis_length, 0, color="#16a34a", linewidth=1.5)
        self.axis.quiver(*origin, 0, 0, axis_length, color="#2563eb", linewidth=1.5)

    @staticmethod
    def _style_2d_axis(axis: Any) -> None:
        axis.set_box_aspect(1.0)
        axis.grid(False)
        axis.yaxis.set_major_locator(MaxNLocator(5))
        axis.minorticks_off()
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color("#000000")
            axis.spines[side].set_linewidth(0.8)
        axis.tick_params(colors="#000000", width=0.7, length=3, direction="out")

    def _configure_top_axis(self) -> None:
        self.top_axis.set_xlim(-2.2, 2.2)
        self.top_axis.set_ylim(-2.2, 2.2)
        self.top_axis.set_xticks(np.linspace(-2.0, 2.0, 5))
        self.top_axis.set_xlabel(r"$x$ (m)")
        self.top_axis.set_ylabel(r"$y$ (m)")
        self._style_2d_axis(self.top_axis)

    @staticmethod
    def _set_top_line(line: Any, points: np.ndarray) -> None:
        line.set_data(points[:, 0], points[:, 1])

    @staticmethod
    def _set_line(line: Any, points: np.ndarray) -> None:
        line.set_data(points[:, 0], points[:, 1])
        line.set_3d_properties(points[:, 2])

    def _section(self, points: np.ndarray, section: int) -> np.ndarray:
        start = section * BACKBONE_POINTS_PER_SECTION
        stop = (section + 1) * BACKBONE_POINTS_PER_SECTION + 1
        return points[start:stop]

    def _create_context_artists(self) -> None:
        from matplotlib import colormaps
        from matplotlib.colors import LinearSegmentedColormap

        palette = colormaps["turbo"]
        maximum_group_size = max(record["ik_count"] for record in self.records)
        self.path_colours = [
            palette(index / max(1, maximum_group_size - 1))
            for index in range(maximum_group_size)
        ]
        tip_colour_map = LinearSegmentedColormap.from_list(
            "tip_path_blues",
            (colormaps["Blues"](0.35), colormaps["Blues"](0.85)),
        )
        self.tip_point_colours = tip_colour_map(
            np.linspace(0.0, 1.0, self.horizon)
        )
        self.context_tip_lines: list[Any] = []
        for colour in self.path_colours:
            line, = self.axis.plot(
                [], [], [], color=colour, alpha=0.20,
                linewidth=1.0,
            )
            self.context_tip_lines.append(line)
        self.terminal_lines: list[Any] = []
        for index in range(maximum_group_size):
            for section in range(3):
                line, = self.axis.plot(
                    [], [], [],
                    color=self.path_colours[index],
                    alpha=0.14,
                    linewidth=1.0,
                    visible=True,
                )
                self.terminal_lines.append(line)

    def _target_group_indices(self, index: int) -> list[int]:
        record = self.records[index]
        return self._indices_by_target[(record["target_id"], record["start_id"])]

    def _compute_target_geometry(self, selected_index: int) -> np.ndarray:
        group = self._target_group_indices(selected_index)
        record = self.records[selected_index]
        condition = (record["target_id"], record["start_id"])
        case_index = self._target_case[condition]
        started = time.perf_counter()
        geometries = compute_path_geometry(
            self.paths[group], self.section_lengths
        )
        elapsed = time.perf_counter() - started
        print(
            f"Read {len(group)} samples for case {case_index}/"
            f"{len(self._indices_by_target) - 1} in {elapsed:.2f} s",
            flush=True,
        )
        selected_slot = group.index(selected_index)
        self._update_context(selected_index, group, geometries)
        return geometries[selected_slot]

    def _update_context(
        self, selected_index: int, group: list[int], geometries: np.ndarray
    ) -> None:
        for slot, tip_line in enumerate(self.context_tip_lines):
            if slot < len(group):
                geometry = geometries[slot]
                tips = geometry[:, -1]
                self._set_line(tip_line, tips)
                tip_line.set_visible(group[slot] != selected_index)
                for section in range(3):
                    terminal_line = self.terminal_lines[3 * slot + section]
                    self._set_line(
                        terminal_line,
                        self._section(geometry[-1], section),
                    )
                    terminal_line.set_visible(True)
            else:
                self._set_line(tip_line, np.empty((0, 3)))
                tip_line.set_visible(False)
                for section in range(3):
                    self.terminal_lines[3 * slot + section].set_visible(False)

    def _create_robot_artists(self) -> None:
        self.robot_lines = []
        self.top_robot_lines = []
        for colour in self.SECTION_COLOURS:
            line, = self.axis.plot(
                [], [], [], color=colour, linewidth=3.0, zorder=10
            )
            self.robot_lines.append(line)
            top_line, = self.top_axis.plot(
                [], [], color=colour, linewidth=3.0, zorder=10
            )
            self.top_robot_lines.append(top_line)
        self.selected_endpoint_lines = []
        for colour in self.SECTION_COLOURS:
            line, = self.axis.plot(
                [], [], [], color=colour, linewidth=1.8, linestyle="--", alpha=0.65
            )
            self.selected_endpoint_lines.append(line)
        self.joint_markers, = self.axis.plot(
            [], [], [], linestyle="", marker="o", markersize=5.0,
            markerfacecolor="white", markeredgecolor="#0f172a", markeredgewidth=1.1,
            zorder=11,
        )
        self.current_tip, = self.axis.plot(
            [], [], [], linestyle="", marker="o", markersize=7,
            markerfacecolor="#0f172a", markeredgecolor="white", markeredgewidth=1.0,
            zorder=11,
        )
        zeros = np.zeros(self.horizon)
        self.tip_points = self.axis.scatter(
            zeros, zeros, zeros, marker=".", s=20,
            c=self.tip_point_colours, alpha=0.82, depthshade=False, zorder=20,
        )
        self.top_tip_points = self.top_axis.scatter(
            zeros, zeros, marker=".", s=20,
            c=self.tip_point_colours, alpha=0.82, zorder=20,
        )
        self.top_tip_line, = self.top_axis.plot(
            [], [], color=self.tip_point_colours[len(self.tip_point_colours) // 2],
            linewidth=1.0, alpha=0.55, zorder=19,
        )
        self.top_start_marker, = self.top_axis.plot(
            [], [], linestyle="", marker="D", markersize=6,
            markerfacecolor=self.tip_point_colours[0], markeredgecolor="white",
            markeredgewidth=0.75, zorder=21,
        )
        self.top_terminal_marker, = self.top_axis.plot(
            [], [], linestyle="", marker="*", markersize=12,
            markerfacecolor=self.tip_point_colours[-1], markeredgecolor="white",
            markeredgewidth=0.75, zorder=21,
        )
        self.selected_target, = self.axis.plot(
            [], [], [], linestyle="", marker="*", markersize=15,
            markerfacecolor=self.tip_point_colours[-1],
            markeredgecolor="white", markeredgewidth=0.75,
            zorder=21,
            label="Target",
        )
        self.start_tip, = self.axis.plot(
            [], [], [], linestyle="", marker="D", markersize=7,
            markerfacecolor=self.tip_point_colours[0],
            markeredgecolor="white", markeredgewidth=0.75,
            zorder=21,
            label="Start",
        )
        self.axis.legend(loc="upper left", frameon=False)

    def _update_obstacles(self, index: int) -> None:
        """Replace sphere surfaces only after a path selection is committed."""
        from matplotlib import colormaps
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.patches import Circle

        for surface in self.obstacle_surfaces:
            surface.remove()
        self.obstacle_surfaces.clear()
        for obstacle in self.top_obstacles:
            obstacle.remove()
        self.top_obstacles.clear()
        azimuth = np.linspace(0.0, 2.0 * np.pi, 64)
        polar = np.linspace(0.0, np.pi, 32)
        unit_x = np.outer(np.cos(azimuth), np.sin(polar))
        unit_y = np.outer(np.sin(azimuth), np.sin(polar))
        unit_z = np.outer(np.ones_like(azimuth), np.cos(polar))
        colour_map = LinearSegmentedColormap.from_list(
            "obstacle_greys",
            (colormaps["Greys"](0.35), colormaps["Greys"](0.15)),
        )
        face_colours = colour_map(0.5 * (unit_z + 1.0))
        face_colours[..., 3] = 0.5
        for centre in self.records[index]["obstacle_centres"]:
            surface = self.axis.plot_surface(
                centre[0] + self.sphere_radius * unit_x,
                centre[1] + self.sphere_radius * unit_y,
                centre[2] + self.sphere_radius * unit_z,
                facecolors=face_colours,
                linewidth=0.0,
                antialiased=True,
                shade=False,
                zorder=5,
            )
            self.obstacle_surfaces.append(surface)
            obstacle = Circle(
                centre[:2], self.sphere_radius,
                facecolor=colormaps["Greys"](0.25), edgecolor=colormaps["Greys"](0.35),
                linewidth=0.8, alpha=0.5,
                zorder=10.0 + float(centre[2]) + self.sphere_radius,
            )
            self.top_axis.add_patch(obstacle)
            self.top_obstacles.append(obstacle)

    def _create_configuration_plot(self) -> None:
        from matplotlib import colormaps

        frames = np.arange(self.horizon)
        self.config_lines = []
        labels = (
            r"$\xi_{11}$", r"$\xi_{12}$", r"$\xi_{21}$",
            r"$\xi_{22}$", r"$\xi_{31}$", r"$\xi_{32}$",
        )
        config_colours = (
            self.tip_point_colours[-1],
            colormaps["Blues"](0.60),
            colormaps["Blues"](0.35),
            colormaps["Reds"](0.35),
            colormaps["Reds"](0.60),
            colormaps["Reds"](0.85),
        )
        for dimension, (colour, label) in enumerate(zip(config_colours, labels)):
            line, = self.config_axis.plot(
                frames,
                self.paths[0, :, dimension],
                color=colour,
                linewidth=1.5,
                label=label,
            )
            self.config_lines.append(line)
        self.frame_indicator = self.config_axis.axvline(
            0, color="#475569", linewidth=0.8, linestyle="--"
        )
        limit = max(float(np.abs(self.paths).max()) * 1.08, 0.25)
        self.config_axis.set_xlim(0, self.horizon - 1)
        self.config_axis.set_ylim(-limit, limit)
        self.config_axis.set_xlabel("Path Frame")
        self.config_axis.set_ylabel("Exponential Co-ordinate")
        self.config_axis.set_xticks(
            np.unique(np.linspace(0, self.horizon - 1, 5).round().astype(int))
        )
        self._style_2d_axis(self.config_axis)
        self.config_axis.legend(
            handles=[
                Line2D(
                    [], [], linestyle="none", marker="s", markersize=6,
                    markerfacecolor=colour, markeredgecolor="none", label=label,
                )
                for colour, label in zip(config_colours, labels)
            ],
            ncol=3, fontsize=10, loc="upper left", frameon=False,
            handlelength=0.8, handletextpad=0.5, columnspacing=1.0,
        )

    def _standardise_text(self) -> None:
        """Use one regular 10-point type style throughout the viewer."""
        from matplotlib.text import Text

        for artist in self.figure.findobj(match=Text):
            artist.set_fontsize(10)
            artist.set_fontweight("normal")

    @staticmethod
    def _status_columns(fields: Sequence[tuple[str, str]]) -> tuple[str, str]:
        labels = "\n".join(f"{label}:" for label, _ in fields)
        values = "\n".join(value for _, value in fields)
        return labels, values

    def _set_status(self, fields: Sequence[tuple[str, str]]) -> None:
        self._status_fields = tuple(fields)
        self._refresh_status()

    def _refresh_status(self) -> None:
        fields = self._status_fields + (
            ("Elevation", f"{float(self.axis.elev):.2f}"),
            ("Azimuth", f"{float(self.axis.azim):.2f}"),
        )
        status_labels, status_values = self._status_columns(fields)
        self.status_labels.set_text(status_labels)
        self.status_values.set_text(status_values)

    def _path_changed(self, value: float) -> None:
        self.requested_path_index = int(round(value))

    def _frame_changed(self, value: float) -> None:
        self.requested_frame_index = int(round(value))
        self.frame_index = self.requested_frame_index
        self.update(path_changed=False)

    def _commit_slider_selection(self) -> None:
        path_changed = self.requested_path_index != self.path_index
        if not path_changed:
            return
        self.path_index = self.requested_path_index
        self.update(path_changed=True)

    def update(self, path_changed: bool) -> None:
        index = self.path_index
        frame = self.frame_index
        if path_changed or self.current_geometry is None:
            self.current_geometry = self._compute_target_geometry(index)
        geometry = self.current_geometry
        tip_path = geometry[:, -1]
        points = geometry[frame]
        for section, line in enumerate(self.robot_lines):
            section_points = self._section(points, section)
            self._set_line(line, section_points)
            top_line = self.top_robot_lines[section]
            top_line.set_zorder(10.0 + float(np.mean(section_points[:, 2])))
            self._set_top_line(top_line, section_points)
        joints = points[
            [
                0,
                BACKBONE_POINTS_PER_SECTION,
                2 * BACKBONE_POINTS_PER_SECTION,
                3 * BACKBONE_POINTS_PER_SECTION,
            ]
        ]
        self._set_line(self.joint_markers, joints)
        self._set_line(self.current_tip, points[-1:])
        self.tip_points._offsets3d = (
            tip_path[:, 0], tip_path[:, 1], tip_path[:, 2]
        )
        self.top_tip_points.set_offsets(tip_path[:, :2])
        self._set_top_line(self.top_tip_line, tip_path)
        self._set_top_line(self.top_start_marker, tip_path[:1])
        self._set_top_line(self.top_terminal_marker, tip_path[-1:])
        self.frame_indicator.set_xdata([frame, frame])

        record = self.records[index]
        if path_changed:
            self._update_obstacles(index)
            endpoint = geometry[-1]
            for section, line in enumerate(self.selected_endpoint_lines):
                self._set_line(line, self._section(endpoint, section))
            self._set_line(self.selected_target, self.targets[index : index + 1])
            self._set_line(self.start_tip, tip_path[:1])
            for dimension, line in enumerate(self.config_lines):
                line.set_ydata(self.paths[index, :, dimension])
            centres = torch.as_tensor(
                record["obstacle_centres"], dtype=torch.float64
            )
            with torch.no_grad():
                clearances, _ = sphere_clearances_and_tips(
                    torch.as_tensor(self.paths[index], dtype=torch.float64),
                    centres,
                    self.section_lengths,
                    self.sphere_radius,
                    self.clearance_margin,
                )
            self.current_minimum_clearance = float(clearances.min())

        condition = (record["target_id"], record["start_id"])
        case_index = self._target_case[condition]
        case_paths = self._indices_by_target[condition]
        sample_index = case_paths.index(index)
        q = self.paths[index, frame]
        tip = tip_path[frame]
        target = self.targets[index]
        error = float(np.linalg.norm(tip - target))
        terminal_error = float(np.linalg.norm(tip_path[-1] - target))
        has_obstacles = len(record["obstacle_centres"]) > 0
        minimum_clearance = self.current_minimum_clearance
        if minimum_clearance is None:
            raise RuntimeError("path clearance was not initialised")
        collision_text = "yes" if minimum_clearance < 0 else "no"
        surface_clearance = minimum_clearance + self.clearance_margin
        clearance_text = f"{surface_clearance:.8f}"
        if not has_obstacles:
            collision_text = "n/a"
            clearance_text = "n/a"
        self._set_status(
            (
                ("Target", record["target_id"]),
                ("Split", record["split"]),
                ("Obstacles", str(len(record["obstacle_centres"]))),
                ("Case", f"{case_index}/{len(self._indices_by_target) - 1}"),
                ("Sample", f"{sample_index}/{len(case_paths) - 1}"),
                ("Path", f"{index}/{self.count - 1}"),
                ("Frame", f"{frame}/{self.horizon - 1}"),
                ("q0, q1", f"{q[0]:.2f}, {q[1]:.2f}"),
                ("q2, q3", f"{q[2]:.2f}, {q[3]:.2f}"),
                ("q4, q5", f"{q[4]:.2f}, {q[5]:.2f}"),
                ("Tip Error", f"{error:.8f}"),
                ("Final Error", f"{terminal_error:.8f}"),
                ("Target Tip", f"{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}"),
                ("Collision", collision_text),
                ("Clearance", clearance_text),
            )
        )
        self.figure.canvas.draw_idle()

    def _toolbar_is_active(self) -> bool:
        manager = self.figure.canvas.manager
        toolbar = getattr(manager, "toolbar", None)
        return bool(toolbar is not None and getattr(toolbar, "mode", ""))

    def _mouse_pressed(self, event: Any) -> None:
        # A new press outside the 3-D axes must cancel any stale orbit state.
        # Some GUI backends can miss a release event when the cursor leaves
        # the window; without this reset, a later slider drag rotates the view.
        if event.inaxes is not self.axis:
            self._orbit_drag = None
        if event.inaxes is self.path_slider.ax:
            self._active_slider = event.inaxes
            return
        if (
            event.inaxes is not self.axis
            or event.button != 1
            or self._toolbar_is_active()
        ):
            return
        self._orbit_drag = (
            float(event.x),
            float(event.y),
            float(self.axis.azim),
            float(self.axis.elev),
        )

    def _mouse_released(self, event: Any) -> None:
        if event.button == 1:
            self._orbit_drag = None
            if self._active_slider is not None:
                self._active_slider = None
                self._commit_slider_selection()

    def _mouse_moved(self, event: Any) -> None:
        if (
            self._orbit_drag is None
            or event.inaxes is not self.axis
            or event.button != 1
            or self._toolbar_is_active()
        ):
            return
        start_x, start_y, start_azimuth, start_elevation = self._orbit_drag
        azimuth = start_azimuth - 0.35 * (float(event.x) - start_x)
        elevation = np.clip(
            start_elevation - 0.35 * (float(event.y) - start_y),
            -89.0,
            89.0,
        )
        self.axis.set_proj_type("ortho")
        self.axis.view_init(elev=float(elevation), azim=float(azimuth), roll=0.0)
        self._refresh_status()
        self.figure.canvas.draw_idle()

    def step_frame(self, amount: int) -> None:
        next_frame = int(np.clip(self.frame_index + amount, 0, self.horizon - 1))
        if next_frame == self.frame_index and amount > 0 and self.playing:
            self.playing = False
            self.timer.stop()
        self.frame_slider.set_val(next_frame)
        self._commit_slider_selection()

    def toggle_playback(self) -> None:
        if self.playing:
            self.playing = False
            self.timer.stop()
        else:
            if self.frame_index == self.horizon - 1:
                self.frame_slider.set_val(0)
                self._commit_slider_selection()
            self.playing = True
            self.timer.start()
        self.figure.canvas.draw_idle()

    def _advance_playback(self) -> None:
        if self.playing:
            self.step_frame(1)

    def _key_pressed(self, event: Any) -> None:
        if event.key == " ":
            self.toggle_playback()
        elif event.key == "right":
            self.step_frame(1)
        elif event.key == "left":
            self.step_frame(-1)
        elif event.key == "up":
            self.path_slider.set_val((self.path_index - 1) % self.count)
            self._commit_slider_selection()
        elif event.key == "down":
            self.path_slider.set_val((self.path_index + 1) % self.count)
            self._commit_slider_selection()
        elif event.key == "home":
            self.frame_slider.set_val(0)
            self._commit_slider_selection()
        elif event.key == "end":
            self.frame_slider.set_val(self.horizon - 1)
            self._commit_slider_selection()

def main() -> None:
    if FPS <= 0:
        raise ValueError("FPS must be positive")
    (
        paths,
        targets,
        records,
        lengths,
        sphere_radius,
        clearance_margin,
    ) = load_visualisation_data(DATASET_DIR)
    print(
        f"Loaded {len(records)} paths x {paths.shape[1]} points",
        flush=True,
    )
    viewer = DatasetViewer(
        paths,
        targets,
        records,
        lengths,
        sphere_radius,
        clearance_margin,
        FPS,
    )
    viewer.plt.show()


if __name__ == "__main__":
    main()
