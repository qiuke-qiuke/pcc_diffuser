"""Interactively inspect DDIM denoising for held-out PCC planning cases.

Edit the settings below, then run ``python tests/vis_denoising.py`` from the
``src/pcc_diffuser`` directory. A case is generated after the Path slider is
released; dragging the sliders therefore remains responsive.
"""

from __future__ import annotations

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
from torch_geometric.data import Batch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pcc_diffuser.data import PccDataset, obstacle_graph
from pcc_diffuser.evaluation import (
    INCORPORATE_ANALYTICAL_OBSTACLE_REPULSION,
    METHODS,
    REPULSION_CLEARANCE_FACTOR,
    evaluation_initial_noise,
    evaluation_noise_seed,
)
from pcc_diffuser.kinematics import (
    backbone_points,
    sphere_clearances_and_tips,
    tip_position,
)
from pcc_diffuser.training import reconstruct_diffusion


CHECKPOINT = PROJECT_DIR / "runs" / "set_v1" / "final.pt"
DATASET = PROJECT_DIR / "data" / "set_v1"
SPLIT = "test"
SAMPLES_PER_CASE = 10
DDIM_STEPS = 50
ETA = 0.0
SEED = 0
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
USE_EMA_WEIGHTS = True
METHOD = METHODS[0]
POST_CORRECTION_STEP_SIZE = 0.05
POST_CORRECTION_REPULSION_STEP_SIZE = 0.05
POST_CORRECTION_FRACTION = 0.4
GUIDANCE_WEIGHT = 50.0
GUIDANCE_REPULSION_WEIGHT = 50.0
GUIDANCE_FRACTION = 0.4

BACKBONE_POINTS_PER_SECTION = 20
DISPLAYED_ROBOTS = 7
FPS = 30.0

class DenoisingViewer:
    """Generate target cases lazily and display their complete DDIM traces."""

    def __init__(
        self,
        diffusion: Any,
        normaliser: Any,
        checkpoint: dict[str, Any],
        dataset: PccDataset,
        case_records: Sequence[int],
        device: torch.device,
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
        self.diffusion = diffusion
        self.normaliser = normaliser
        self.dataset = dataset
        self.case_records = list(case_records)
        self.device = device
        self.lengths = tuple(float(value) for value in checkpoint["section_lengths"])
        self.path_points = np.linspace(
            0, diffusion.horizon - 1, DISPLAYED_ROBOTS, dtype=int
        )
        self.case_index = 0
        self.sample_index = 0
        self.ddim_index = 0
        self.path_frame_index = 0
        self.path_index = 0
        self.requested_path_index = 0
        self.requested_ddim_index = 0
        self.current_case_data: dict[str, np.ndarray] | None = None
        self.obstacle_surfaces: list[Any] = []
        self.top_obstacles: list[Any] = []
        self.playback_mode: str | None = None
        self._active_slider: Any | None = None
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
            - 3.0 * slider_height
            - 2.0 * slider_gap
            - instruction_gap
        )
        plot_bottom = outer_margin + instruction_gap + 3.0 * slider_height
        plot_bottom += 2.0 * slider_gap + plot_slider_gap
        plot_top = plot_bottom + plot_size
        slider_bottoms = tuple(
            plot_bottom
            - plot_slider_gap
            - (index + 1.0) * slider_height
            - index * slider_gap
            for index in range(3)
        )

        self.figure = plt.figure(
            figsize=(figure_width, figure_height), facecolor="white"
        )
        self.figure.canvas.manager.set_window_title("PCC DDIM Denoising Viewer")
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
        self._create_3d_artists()
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
        step_axis = self.figure.add_axes(
            [
                plot_position.x0,
                slider_bottoms[2] / figure_height,
                slider_width,
                slider_height / figure_height,
            ]
        )
        self.path_slider = Slider(
            path_axis, "Path", 0, len(self.case_records) * SAMPLES_PER_CASE - 1,
            valinit=0, valstep=1, valfmt="%0.0f",
        )
        self.frame_slider = Slider(
            frame_axis, "Frame", 0, self.diffusion.horizon - 1,
            valinit=0, valstep=1, valfmt="%0.0f",
        )
        self.step_slider = Slider(
            step_axis, "Step", 0, DDIM_STEPS,
            valinit=0, valstep=1, valfmt="%0.0f",
        )
        self.path_slider.on_changed(self._path_changed)
        self.frame_slider.on_changed(self._path_frame_changed)
        self.step_slider.on_changed(self._step_changed)
        self.timer = self.figure.canvas.new_timer(
            interval=max(1, int(round(1000.0 / FPS)))
        )
        self.timer.add_callback(self._advance_playback)
        self.figure.canvas.mpl_connect("key_press_event", self._key_pressed)
        self.figure.canvas.mpl_connect("button_press_event", self._mouse_pressed)
        self.figure.canvas.mpl_connect("button_release_event", self._mouse_released)
        self.figure.canvas.mpl_connect("motion_notify_event", self._mouse_moved)
        self.figure.text(
            0.5 * (plot_position.x0 + slider_right),
            outer_margin / figure_height,
            "Space: Play/Pause  |  Enter: Play/Pause  |  "
            "Left/Right: Frame  |  Up/Down: Path",
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
        self.update(case_changed=True)

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
        if hasattr(self.axis, "disable_mouse_rotation"):
            self.axis.disable_mouse_rotation()
        total_length = float(sum(self.lengths))
        margin = 0.1 * total_length
        self.axis.set_xlim(-total_length - margin, total_length + margin)
        self.axis.set_ylim(-total_length - margin, total_length + margin)
        self.axis.set_zlim(-margin, total_length + margin)
        section_boundaries = np.concatenate(
            ([0.0], np.cumsum(self.lengths, dtype=float))
        )
        self.axis.set_zticks(section_boundaries)
        self.axis.set_box_aspect(
            (2.0 * (total_length + margin),) * 2
            + (total_length + 2.0 * margin,)
        )
        axis_length = 0.24 * total_length
        self.axis.quiver(0, 0, 0, axis_length, 0, 0, color="#dc2626", linewidth=1.5)
        self.axis.quiver(0, 0, 0, 0, axis_length, 0, color="#16a34a", linewidth=1.5)
        self.axis.quiver(0, 0, 0, 0, 0, axis_length, color="#2563eb", linewidth=1.5)

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

    def _create_3d_artists(self) -> None:
        from matplotlib import colormaps
        from matplotlib.colors import LinearSegmentedColormap

        palette = colormaps["turbo"]
        self.sample_colours = [
            palette(index / max(1, SAMPLES_PER_CASE - 1))
            for index in range(SAMPLES_PER_CASE)
        ]
        tip_colour_map = LinearSegmentedColormap.from_list(
            "tip_path_blues",
            (colormaps["Blues"](0.35), colormaps["Blues"](0.85)),
        )
        self.tip_point_colours = tip_colour_map(
            np.linspace(0.0, 1.0, self.diffusion.horizon)
        )
        zeros = np.zeros(self.diffusion.horizon)
        self.tip_points = self.axis.scatter(
            zeros,
            zeros,
            zeros,
            marker=".",
            s=20,
            c=self.tip_point_colours,
            alpha=0.82,
            depthshade=False,
            zorder=20,
        )
        self.top_tip_points = self.top_axis.scatter(
            zeros,
            zeros,
            marker=".",
            s=20,
            c=self.tip_point_colours,
            alpha=0.82,
            zorder=20,
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
        self.current_robot_lines: list[Any] = []
        for _path_point in self.path_points:
            for _section in range(3):
                line, = self.axis.plot(
                    [], [], [], color="#111111", linewidth=1.4,
                    alpha=0.72, zorder=10,
                )
                self.current_robot_lines.append(line)
        self.top_robot_lines: list[Any] = []
        for _section in range(3):
            line, = self.top_axis.plot(
                [], [], color="#111111", linewidth=1.4, zorder=10
            )
            self.top_robot_lines.append(line)
        self.final_tip_paths: list[Any] = []
        self.terminal_lines: list[Any] = []
        for colour in self.sample_colours:
            tip_line, = self.axis.plot(
                [], [], [], color=colour, linewidth=1.0, alpha=0.18
            )
            self.final_tip_paths.append(tip_line)
            for _section in range(3):
                line, = self.axis.plot(
                    [], [], [], color=colour, linewidth=1.0, alpha=0.18
                )
                self.terminal_lines.append(line)
        self.target_marker, = self.axis.plot(
            [], [], [], linestyle="", marker="*", markersize=15,
            markerfacecolor=self.tip_point_colours[-1],
            markeredgecolor="white", markeredgewidth=0.75,
            zorder=21,
            label="Target",
        )
        self.start_marker, = self.axis.plot(
            [], [], [], linestyle="", marker="D", markersize=7,
            markerfacecolor=self.tip_point_colours[0],
            markeredgecolor="white", markeredgewidth=0.75,
            zorder=21,
            label="Start",
        )
        self.axis.legend(loc="upper left", frameon=False)

    def _update_obstacles(self, centres: np.ndarray) -> None:
        """Replace sphere surfaces when a new test case is generated."""
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
        for centre in centres:
            surface = self.axis.plot_surface(
                centre[0] + self.dataset.sphere_radius * unit_x,
                centre[1] + self.dataset.sphere_radius * unit_y,
                centre[2] + self.dataset.sphere_radius * unit_z,
                facecolors=face_colours,
                linewidth=0.0,
                antialiased=True,
                shade=False,
                zorder=5,
            )
            self.obstacle_surfaces.append(surface)
            obstacle = Circle(
                centre[:2], self.dataset.sphere_radius,
                facecolor=colormaps["Greys"](0.25),
                edgecolor=colormaps["Greys"](0.35),
                linewidth=0.8, alpha=0.5,
                zorder=10.0 + float(centre[2]) + self.dataset.sphere_radius,
            )
            self.top_axis.add_patch(obstacle)
            self.top_obstacles.append(obstacle)

    def _create_configuration_plot(self) -> None:
        from matplotlib import colormaps

        points = np.arange(self.diffusion.horizon)
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
        self.config_lines: list[Any] = []
        for colour, label in zip(config_colours, labels):
            line, = self.config_axis.plot(
                points, np.zeros_like(points), color=colour, linewidth=1.5, label=label
            )
            self.config_lines.append(line)
        self.frame_indicator = self.config_axis.axvline(
            0, color="#475569", linewidth=0.8, linestyle="--"
        )
        self.config_axis.set_xlim(0, self.diffusion.horizon - 1)
        self.config_axis.set_xlabel("Path Frame")
        self.config_axis.set_ylabel("Exponential Co-ordinate")
        self.config_axis.set_xticks(
            [0, 16, 32, 48, self.diffusion.horizon - 1]
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
            ncol=3, loc="upper left", frameon=False,
            handlelength=0.8, handletextpad=0.5, columnspacing=1.0,
        )

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

    def _generate_case(self, case_index: int) -> dict[str, np.ndarray]:
        record = self.case_records[case_index]
        target_id = int(self.dataset.target_indices[record])
        start_id = int(self.dataset.start_indices[record])
        start_physical = torch.from_numpy(self.dataset.paths[record, 0]).float().to(
            self.device
        )
        target_physical = torch.from_numpy(
            self.dataset.target_tips[record]
        ).float().to(self.device)
        start = self.normaliser.normalise_q(start_physical).expand(
            SAMPLES_PER_CASE, -1
        ).clone()
        target = self.normaliser.normalise_workspace(target_physical).expand(
            SAMPLES_PER_CASE, -1
        ).clone()
        obstacle_centres = self.dataset.obstacles(record).copy()
        centres = torch.from_numpy(obstacle_centres).float()
        centres = self.normaliser.normalise_workspace(centres)
        radius = self.normaliser.normalise_workspace_radius(
            self.dataset.sphere_radius
        )
        obstacle_batch = Batch.from_data_list(
            [obstacle_graph(centres, radius) for _ in range(SAMPLES_PER_CASE)]
        ).to(self.device)
        physical_obstacles = torch.from_numpy(obstacle_centres).to(
            device=self.device, dtype=start.dtype
        )
        padded_obstacles = physical_obstacles[None].expand(
            SAMPLES_PER_CASE, -1, -1
        )
        obstacle_mask = torch.ones(
            padded_obstacles.shape[:2], dtype=torch.bool, device=self.device
        )
        repulsion_safe_radius = self.dataset.sphere_radius + (
            REPULSION_CLEARANCE_FACTOR * self.dataset.clearance_margin
        )
        noise_seed = evaluation_noise_seed(SEED, target_id, start_id)
        initial_noise = evaluation_initial_noise(
            SAMPLES_PER_CASE,
            self.diffusion.horizon,
            SEED,
            target_id,
            start_id,
            start.dtype,
            self.device,
        )
        sampler_generator = torch.Generator(device=self.device).manual_seed(
            noise_seed + 1
        )
        started = time.perf_counter()
        if METHOD == "With Guided Prediction":
            _, trace = self.diffusion.sample_ddim(
                start,
                target,
                obstacle_batch,
                sample_steps=DDIM_STEPS,
                eta=ETA,
                generator=sampler_generator,
                initial_noise=initial_noise,
                normaliser=self.normaliser,
                guidance_weight=GUIDANCE_WEIGHT,
                guidance_fraction=GUIDANCE_FRACTION,
                post_correction=False,
                post_correction_step_size=POST_CORRECTION_STEP_SIZE,
                post_correction_fraction=POST_CORRECTION_FRACTION,
                section_lengths=self.lengths,
                return_trace=True,
                obstacle_centres=padded_obstacles,
                obstacle_mask=obstacle_mask,
                guidance_repulsion_weight=(
                    GUIDANCE_REPULSION_WEIGHT
                    if INCORPORATE_ANALYTICAL_OBSTACLE_REPULSION
                    else 0.0
                ),
                repulsion_safe_radius=repulsion_safe_radius,
            )
        else:
            with torch.no_grad():
                _, trace = self.diffusion.sample_ddim(
                    start,
                    target,
                    obstacle_batch,
                    sample_steps=DDIM_STEPS,
                    eta=ETA,
                    generator=sampler_generator,
                    initial_noise=initial_noise,
                    normaliser=(
                        self.normaliser if METHOD == "With Post Correction" else None
                    ),
                    guidance_weight=0.0,
                    guidance_fraction=GUIDANCE_FRACTION,
                    post_correction=METHOD == "With Post Correction",
                    post_correction_step_size=POST_CORRECTION_STEP_SIZE,
                    post_correction_fraction=POST_CORRECTION_FRACTION,
                    section_lengths=self.lengths,
                    return_trace=True,
                    obstacle_centres=padded_obstacles,
                    obstacle_mask=obstacle_mask,
                    post_correction_repulsion_weight=(
                        POST_CORRECTION_REPULSION_STEP_SIZE
                        if INCORPORATE_ANALYTICAL_OBSTACLE_REPULSION
                        and METHOD == "With Post Correction"
                        else 0.0
                    ),
                    repulsion_safe_radius=repulsion_safe_radius,
                )
        with torch.no_grad():
            physical = self.normaliser.denormalise_q(trace).cpu()
            tips = tip_position(physical, self.lengths)
            shapes = backbone_points(
                physical[:, :, self.path_points],
                self.lengths,
                BACKBONE_POINTS_PER_SECTION,
            )
            start_tip = tip_position(start_physical.cpu(), self.lengths)
            physical_centres = torch.from_numpy(obstacle_centres).to(
                dtype=physical.dtype
            )
            clearances, _ = sphere_clearances_and_tips(
                physical[:, -1],
                physical_centres,
                self.lengths,
                self.dataset.sphere_radius,
                self.dataset.clearance_margin,
            )
        value = {
            "paths": physical.numpy(),
            "tips": tips.numpy(),
            "shapes": shapes.numpy(),
            "target": target_physical.cpu().numpy(),
            "start_tip": start_tip.numpy(),
            "clearances": clearances.numpy(),
            "target_id": np.asarray(target_id),
            "obstacle_centres": obstacle_centres,
        }
        print(
            f"Generating {SAMPLES_PER_CASE} samples for case {case_index}/"
            f"{len(self.case_records) - 1} with {METHOD} on "
            f"{self.device} in "
            f"{time.perf_counter() - started:.2f} s",
            flush=True,
        )
        return value

    def _path_changed(self, value: float) -> None:
        self.requested_path_index = int(round(value))

    def _path_frame_changed(self, value: float) -> None:
        self.path_frame_index = int(round(value))
        self.update(case_changed=False)

    def _step_changed(self, value: float) -> None:
        self.requested_ddim_index = int(round(value))
        self.ddim_index = self.requested_ddim_index
        self.update(case_changed=False)

    def _commit_slider_selection(self) -> None:
        requested_case, requested_sample = divmod(
            self.requested_path_index, SAMPLES_PER_CASE
        )
        case_changed = requested_case != self.case_index
        if self.requested_path_index == self.path_index:
            return
        self.path_index = self.requested_path_index
        self.case_index = requested_case
        self.sample_index = requested_sample
        self.update(case_changed=case_changed)

    def update(self, case_changed: bool) -> None:
        if case_changed or self.current_case_data is None:
            self.current_case_data = self._generate_case(self.case_index)
        data = self.current_case_data
        sample = self.sample_index
        step = self.ddim_index
        frame = self.path_frame_index
        tip_path = data["tips"][sample, step]
        self.tip_points._offsets3d = (
            tip_path[:, 0], tip_path[:, 1], tip_path[:, 2]
        )
        self.top_tip_points.set_offsets(tip_path[:, :2])
        self.top_tip_points.set_zorder(20.0 + float(np.mean(tip_path[:, 2])))
        shapes = data["shapes"][sample, step]
        final_step = step == data["paths"].shape[1] - 1
        displayed_clearances = data["clearances"][sample, self.path_points]
        for robot_index, robot in enumerate(shapes):
            for section in range(3):
                self.current_robot_lines[3 * robot_index + section].set_color(
                    "y"
                    if final_step and displayed_clearances[robot_index] < 0
                    else "black"
                )
                self._set_line(
                    self.current_robot_lines[3 * robot_index + section],
                    self._section(robot, section),
                )
        selected_path = data["paths"][sample, step]
        with torch.no_grad():
            selected_robot = backbone_points(
                torch.as_tensor(selected_path[frame : frame + 1]),
                self.lengths,
                BACKBONE_POINTS_PER_SECTION,
            )[0].numpy()
        selected_colour = (
            "y" if final_step and data["clearances"][sample, frame] < 0 else "#111111"
        )
        for section, line in enumerate(self.top_robot_lines):
            line.set_color(selected_colour)
            section_points = self._section(selected_robot, section)
            line.set_zorder(10.0 + float(np.mean(section_points[:, 2])))
            self._set_top_line(line, section_points)
        for dimension, line in enumerate(self.config_lines):
            line.set_ydata(selected_path[:, dimension])
        self.frame_indicator.set_xdata([frame, frame])
        limit = max(float(np.abs(selected_path).max()) * 1.08, 0.25)
        self.config_axis.set_ylim(-limit, limit)

        if case_changed:
            self._update_obstacles(data["obstacle_centres"])
            self._set_top_line(self.top_start_marker, data["start_tip"][None])
            self.top_start_marker.set_zorder(
                21.0 + float(data["start_tip"][2])
            )
            self._set_top_line(self.top_terminal_marker, data["target"][None])
            self.top_terminal_marker.set_zorder(21.0 + float(data["target"][2]))
            for sample_index in range(SAMPLES_PER_CASE):
                self._set_line(
                    self.final_tip_paths[sample_index], data["tips"][sample_index, -1]
                )
                terminal = data["shapes"][sample_index, -1, -1]
                for section in range(3):
                    self._set_line(
                        self.terminal_lines[3 * sample_index + section],
                        self._section(terminal, section),
                    )
            self._set_line(self.target_marker, data["target"][None])
            self._set_line(self.start_marker, data["start_tip"][None])

        endpoint_error = float(np.linalg.norm(tip_path[-1] - data["target"]))
        final_error = float(
            np.linalg.norm(data["tips"][sample, -1, -1] - data["target"])
        )
        final_minimum_clearance = float(data["clearances"][sample].min())
        final_collision = final_minimum_clearance < 0
        has_obstacles = len(data["obstacle_centres"]) > 0
        collision_text = "yes" if final_collision else "no"
        surface_clearance = (
            final_minimum_clearance + self.dataset.clearance_margin
        )
        clearance_text = f"{surface_clearance:.8f}"
        if not has_obstacles:
            collision_text = "n/a"
            clearance_text = "n/a"
        q = selected_path[frame]
        self._set_status(
            (
                ("Target", f"target{int(data['target_id']):05d}"),
                ("Split", SPLIT),
                ("Method", METHOD),
                ("Obstacles", str(len(data["obstacle_centres"]))),
                ("Case", f"{self.case_index}/{len(self.case_records) - 1}"),
                ("Sample", f"{sample}/{SAMPLES_PER_CASE - 1}"),
                (
                    "Path",
                    f"{self.path_index}/"
                    f"{len(self.case_records) * SAMPLES_PER_CASE - 1}",
                ),
                ("Frame", f"{frame}/{self.diffusion.horizon - 1}"),
                ("Step", f"{step}/{DDIM_STEPS}"),
                ("q0, q1", f"{q[0]:.2f}, {q[1]:.2f}"),
                ("q2, q3", f"{q[2]:.2f}, {q[3]:.2f}"),
                ("q4, q5", f"{q[4]:.2f}, {q[5]:.2f}"),
                ("Tip Error", f"{endpoint_error:.8f}"),
                ("Final Error", f"{final_error:.8f}"),
                (
                    "Target Tip",
                    f"{data['target'][0]:.2f}, {data['target'][1]:.2f}, "
                    f"{data['target'][2]:.2f}",
                ),
                ("Collision", collision_text),
                ("Clearance", clearance_text),
            )
        )
        self.figure.canvas.draw_idle()

    def step_path_frame(self, amount: int) -> None:
        next_frame = int(
            np.clip(
                self.path_frame_index + amount,
                0,
                self.diffusion.horizon - 1,
            )
        )
        if (
            next_frame == self.path_frame_index
            and amount > 0
            and self.playback_mode == "frame"
        ):
            self._stop_playback()
        self.frame_slider.set_val(next_frame)
        self._commit_slider_selection()

    def step_denoising(self, amount: int) -> None:
        next_frame = int(np.clip(self.ddim_index + amount, 0, DDIM_STEPS))
        if (
            next_frame == self.ddim_index
            and amount > 0
            and self.playback_mode == "step"
        ):
            self._stop_playback()
        self.step_slider.set_val(next_frame)
        self._commit_slider_selection()

    def _stop_playback(self) -> None:
        self.playback_mode = None
        self.timer.stop()

    def toggle_frame_playback(self) -> None:
        if self.playback_mode == "frame":
            self._stop_playback()
        else:
            if self.path_frame_index == self.diffusion.horizon - 1:
                self.frame_slider.set_val(0)
                self._commit_slider_selection()
            self.playback_mode = "frame"
            self.timer.start()
        self.figure.canvas.draw_idle()

    def toggle_step_playback(self) -> None:
        if self.playback_mode == "step":
            self._stop_playback()
        else:
            if self.ddim_index == DDIM_STEPS:
                self.step_slider.set_val(0)
                self._commit_slider_selection()
            self.playback_mode = "step"
            self.timer.start()
        self.figure.canvas.draw_idle()

    def _advance_playback(self) -> None:
        if self.playback_mode == "frame":
            self.step_path_frame(1)
        elif self.playback_mode == "step":
            self.step_denoising(1)

    def _key_pressed(self, event: Any) -> None:
        if event.key == " ":
            self.toggle_frame_playback()
        elif event.key in ("enter", "return"):
            self.toggle_step_playback()
        elif event.key == "right":
            self.step_path_frame(1)
        elif event.key == "left":
            self.step_path_frame(-1)
        elif event.key == "up":
            self.path_slider.set_val(
                (self.path_index - 1)
                % (len(self.case_records) * SAMPLES_PER_CASE)
            )
            self._commit_slider_selection()
        elif event.key == "down":
            self.path_slider.set_val(
                (self.path_index + 1)
                % (len(self.case_records) * SAMPLES_PER_CASE)
            )
            self._commit_slider_selection()
        elif event.key == "home":
            self.frame_slider.set_val(0)
            self._commit_slider_selection()
        elif event.key == "end":
            self.frame_slider.set_val(self.diffusion.horizon - 1)
            self._commit_slider_selection()

    def _toolbar_is_active(self) -> bool:
        toolbar = getattr(self.figure.canvas.manager, "toolbar", None)
        return bool(toolbar is not None and getattr(toolbar, "mode", ""))

    def _mouse_pressed(self, event: Any) -> None:
        if event.inaxes is not self.axis:
            self._orbit_drag = None
        if event.inaxes is self.path_slider.ax:
            self._active_slider = event.inaxes
            return
        if event.inaxes is not self.axis or event.button != 1 or self._toolbar_is_active():
            return
        self._orbit_drag = (
            float(event.x), float(event.y), float(self.axis.azim), float(self.axis.elev)
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
            start_elevation - 0.35 * (float(event.y) - start_y), -89.0, 89.0
        )
        self.axis.set_proj_type("ortho")
        self.axis.view_init(elev=float(elevation), azim=float(azimuth), roll=0.0)
        self._refresh_status()
        self.figure.canvas.draw_idle()

def main() -> None:
    if SAMPLES_PER_CASE < 1 or DDIM_STEPS < 1:
        raise ValueError("sample and DDIM-step counts must be positive")
    if METHOD not in METHODS:
        raise ValueError(f"METHOD must be one of {METHODS}")
    device = torch.device(DEVICE)
    diffusion, normaliser, checkpoint = reconstruct_diffusion(
        CHECKPOINT,
        weights="ema" if USE_EMA_WEIGHTS else "model",
        device=device,
    )
    dataset = PccDataset.load(DATASET, split=SPLIT, normaliser=normaliser)
    first_by_condition: dict[tuple[int, int], int] = {}
    for record_index, (target_index, start_index) in enumerate(
        zip(dataset.target_indices.tolist(), dataset.start_indices.tolist())
    ):
        first_by_condition.setdefault(
            (int(target_index), int(start_index)), record_index
        )
    case_records = list(first_by_condition.values())
    if not case_records:
        raise RuntimeError(f"no unique target cases found in split {SPLIT!r}")
    viewer = DenoisingViewer(
        diffusion, normaliser, checkpoint, dataset, case_records, device
    )
    viewer.plt.show()


if __name__ == "__main__":
    main()
