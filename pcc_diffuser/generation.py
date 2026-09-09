"""Generate geometric paths for conditional PCC diffusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Sequence

import numpy as np
import torch

from .data import DATASET_FORMAT_VERSION
from .kinematics import (
    backbone_point_jacobians,
    backbone_points,
    sphere_clearances_and_tips,
    tip_position,
    tip_position_and_jacobian,
)


ProgressCallback = Callable[[str], None]
SPLIT_CODES = {
    "train": 0,
    "validation": 1,
    "test": 2
}
PATH_FAILURE_REASONS = (
    "workspace_ik",
    "path_constraints",
    "collision_repair",
)
TARGET_DISCARD_REASONS = (
    "terminal_ik",
    "start_sampling",
    "zero_retained_paths"
)


@dataclass(frozen=True)
class PccGenerationConfig:
    """Settings for one dense NPZ dataset generation run."""

    output_dir: str | Path
    seed: int
    target_samples: int
    target_batch_size: int
    terminal_count: int
    sample_batch_size: int
    path_batch_size: int
    start_count: int
    obstacle_counts: tuple[int, ...]
    horizon: int
    section_lengths: tuple[float, float, float]
    max_bending: float
    radial_min: float
    radial_max: float
    z_min: float
    z_max: float
    ik_iterations: int
    ik_convergence_interval: int
    ik_damping: float
    ik_tolerance: float
    ik_continuity_gain: float
    separation: float
    max_adjacent_step: float
    min_path_distance: float
    split_seed: int
    train_fraction: float
    validation_fraction: float
    sphere_radius: float
    obstacle_layout_radius: float
    obstacle_z: float
    clearance_margin: float
    repulsion_clearance_factor: float
    max_collision_repairs: int
    spline_dp_tolerance: float
    device: str
    overwrite: bool

    @property
    def maximum_paths_per_target(self) -> int:
        return self.terminal_count * self.start_count

    @property
    def maximum_path_count(self) -> int:
        return self.target_samples * self.maximum_paths_per_target

    @property
    def repulsion_clearance_margin(self) -> float:
        return self.repulsion_clearance_factor * self.clearance_margin

    def validated(self) -> "PccGenerationConfig":
        positive_integers = {
            "target_samples": self.target_samples,
            "target_batch_size": self.target_batch_size,
            "terminal_count": self.terminal_count,
            "sample_batch_size": self.sample_batch_size,
            "path_batch_size": self.path_batch_size,
            "start_count": self.start_count,
            "horizon": self.horizon,
            "ik_iterations": self.ik_iterations,
            "ik_convergence_interval": self.ik_convergence_interval,
            "max_collision_repairs": self.max_collision_repairs,
        }
        for name, value in positive_integers.items():
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.sample_batch_size < max(self.terminal_count, self.start_count):
            raise ValueError(
                "sample_batch_size must be at least terminal_count and start_count"
            )
        obstacle_counts = tuple(self.obstacle_counts)
        if not obstacle_counts:
            raise ValueError("obstacle_counts must not be empty")
        if any(
            not isinstance(count, int) or count not in (0, 1, 2, 4)
            for count in obstacle_counts
        ):
            raise ValueError("obstacle_counts may contain only 0, 1, 2, and 4")
        if len(set(obstacle_counts)) != len(obstacle_counts):
            raise ValueError("obstacle_counts must not contain duplicates")
        if self.target_samples % len(obstacle_counts):
            raise ValueError(
                "target_samples must be divisible by the number of obstacle "
                "counts for equal target-scene allocation"
            )
        if self.horizon < 3:
            raise ValueError("horizon must be at least three")
        finite_positive = {
            "max_bending": self.max_bending,
            "ik_damping": self.ik_damping,
            "ik_tolerance": self.ik_tolerance,
            "separation": self.separation,
            "sphere_radius": self.sphere_radius,
            "clearance_margin": self.clearance_margin,
            "repulsion_clearance_factor": self.repulsion_clearance_factor,
            "spline_dp_tolerance": self.spline_dp_tolerance,
            "obstacle_layout_radius": self.obstacle_layout_radius,
            "max_adjacent_step": self.max_adjacent_step,
            "min_path_distance": self.min_path_distance,
        }
        for name, value in finite_positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.ik_continuity_gain)
            or self.ik_continuity_gain < 0
        ):
            raise ValueError("ik_continuity_gain must be finite and non-negative")
        if not (
            0 <= self.radial_min < self.radial_max
            and 0 <= self.z_min < self.z_max
        ):
            raise ValueError("target workspace bounds must be ordered and non-negative")
        if (
            4 in obstacle_counts
            and self.obstacle_layout_radius < math.sqrt(2) * self.sphere_radius
        ):
            raise ValueError(
                "obstacle_layout_radius must be at least sqrt(2) * sphere_radius "
                "for four non-overlapping obstacles"
            )
        if (
            2 in obstacle_counts
            and self.obstacle_layout_radius < self.sphere_radius
        ):
            raise ValueError(
                "obstacle_layout_radius must be at least sphere_radius for two "
                "non-overlapping obstacles"
            )
        if not self.z_min <= self.obstacle_z <= self.z_max:
            raise ValueError("obstacle_z must lie within [z_min,z_max]")
        if not (0 < self.train_fraction < 1):
            raise ValueError("train_fraction must lie strictly between zero and one")
        if not (0 <= self.validation_fraction < 1 - self.train_fraction):
            raise ValueError("validation_fraction leaves no test split")
        lengths = tuple(float(value) for value in self.section_lengths)
        if len(lengths) != 3 or not all(math.isfinite(v) and v > 0 for v in lengths):
            raise ValueError("section_lengths must contain three finite positive values")
        return replace(
            self,
            section_lengths=lengths,
            obstacle_counts=obstacle_counts,
        )


@dataclass(frozen=True)
class GenerationSummary:
    output_dir: Path
    path_count: int
    target_count: int
    path_failure_counts: dict[str, int]
    elapsed_seconds: float


SceneResult = tuple[
    np.ndarray,
    list[np.ndarray],
    list[int],
    list[int],
    np.ndarray,
]


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested, but CUDA is unavailable")
    return device


def _solver_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "cuda" else torch.float64


def _sample_configuration_batch(
    rng: np.random.Generator,
    count: int,
    max_bending: float,
) -> np.ndarray:
    radii = max_bending * np.sqrt(rng.random((count, 3)))
    angles = rng.uniform(-math.pi, math.pi, size=(count, 3))
    values = np.empty((count, 6), dtype=np.float64)
    values[:, 0::2] = radii * np.cos(angles)
    values[:, 1::2] = radii * np.sin(angles)
    return values


def _project_section_discs(configurations: torch.Tensor, limit: float) -> torch.Tensor:
    pairs = configurations.reshape(-1, 3, 2)
    norms = torch.linalg.vector_norm(pairs, dim=-1, keepdim=True)
    scales = torch.clamp(limit / torch.clamp(norms, min=1e-12), max=1.0)
    return (pairs * scales).reshape_as(configurations)


def _damped_newton(
    initial_configurations: torch.Tensor,
    desired_tips: torch.Tensor,
    config: PccGenerationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve independent tip IK rows and stop updating rows once converged."""
    configurations = initial_configurations.clone()
    desired_tips = torch.broadcast_to(desired_tips, configurations.shape[:-1] + (3,))
    original_shape = configurations.shape[:-1]
    configurations = configurations.reshape(-1, 6)
    desired_tips = desired_tips.reshape(-1, 3)
    active = torch.arange(len(configurations), device=configurations.device)
    identity = torch.eye(3, dtype=configurations.dtype, device=configurations.device)

    for iteration in range(config.ik_iterations):
        if active.numel() == 0:
            break
        active_configurations = configurations[active]
        current_tips, jacobian = tip_position_and_jacobian(
            active_configurations, config.section_lengths
        )
        residual = desired_tips[active] - current_tips
        errors = torch.linalg.vector_norm(residual, dim=-1)
        unconverged = errors > config.ik_tolerance
        check_convergence = iteration % config.ik_convergence_interval == 0
        if check_convergence:
            if not torch.any(unconverged):
                break
            active = active[unconverged]
            residual = residual[unconverged]
            jacobian = jacobian[unconverged]
        system = jacobian @ jacobian.transpose(-1, -2) + config.ik_damping * identity
        task_step = torch.linalg.solve(system, residual.unsqueeze(-1))
        update = (jacobian.transpose(-1, -2) @ task_step).squeeze(-1)
        if not check_convergence:
            update = update * unconverged[:, None]
        configurations[active] = _project_section_discs(
            configurations[active] + update, config.max_bending
        )

    final_tips = tip_position(configurations, config.section_lengths)
    errors = torch.linalg.vector_norm(final_tips - desired_tips, dim=-1)
    success = torch.isfinite(configurations).all(dim=-1) & (
        errors <= config.ik_tolerance
    )
    return (
        configurations.reshape(original_shape + (6,)),
        success.reshape(original_shape),
        errors.reshape(original_shape),
    )


def _continuity_aware_path_ik(
    initial_configurations: torch.Tensor,
    desired_tips: torch.Tensor,
    start_configurations: torch.Tensor,
    goal_configurations: torch.Tensor,
    config: PccGenerationConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Solve path interiors with tip tracking and null-space continuity."""
    if initial_configurations.ndim != 3:
        raise ValueError("initial_configurations must have shape [B,F,6]")
    if desired_tips.shape != initial_configurations.shape[:-1] + (3,):
        raise ValueError("desired_tips must have shape [B,F,3]")
    expected_endpoints = (len(initial_configurations), 6)
    if (
        start_configurations.shape != expected_endpoints
        or goal_configurations.shape != expected_endpoints
    ):
        raise ValueError("path endpoints must have shape [B,6]")

    configurations = initial_configurations.clone()
    identity = torch.eye(
        3, dtype=configurations.dtype, device=configurations.device
    )
    for iteration in range(config.ik_iterations):
        current_tips, jacobian = tip_position_and_jacobian(
            configurations, config.section_lengths
        )
        residual = desired_tips - current_tips
        errors = torch.linalg.vector_norm(residual, dim=-1)
        converged = torch.all(errors <= config.ik_tolerance, dim=1)
        if (
            iteration % config.ik_convergence_interval == 0
            and bool(torch.all(converged))
        ):
            break
        full_path = torch.cat(
            (
                start_configurations[:, None],
                configurations,
                goal_configurations[:, None],
            ),
            dim=1,
        )
        continuity_gradient = 2.0 * (
            2.0 * configurations - full_path[:, :-2] - full_path[:, 2:]
        )
        system = (
            jacobian @ jacobian.transpose(-1, -2)
            + config.ik_damping * identity
        )
        primary = (
            jacobian.transpose(-1, -2)
            @ torch.linalg.solve(system, residual.unsqueeze(-1))
        ).squeeze(-1)
        task_gradient = (
            jacobian @ continuity_gradient.unsqueeze(-1)
        )
        projected_gradient = continuity_gradient - (
            jacobian.transpose(-1, -2)
            @ torch.linalg.solve(system, task_gradient)
        ).squeeze(-1)
        secondary = -config.ik_continuity_gain * projected_gradient
        updated = _project_section_discs(
            configurations + primary + secondary,
            config.max_bending,
        )
        configurations = torch.where(
            converged[:, None, None], configurations, updated
        )

    final_tips = tip_position(configurations, config.section_lengths)
    errors = torch.linalg.vector_norm(final_tips - desired_tips, dim=-1)
    success = torch.isfinite(configurations).all(dim=-1) & (
        errors <= config.ik_tolerance
    )
    return configurations, success, errors


def _select_distinct_configurations(
    candidates: np.ndarray,
    errors: np.ndarray,
    separation: float,
    count: int,
) -> list[np.ndarray]:
    selected: list[np.ndarray] = []
    for index in np.argsort(errors):
        candidate = candidates[index]
        if all(np.linalg.norm(candidate - other) >= separation for other in selected):
            selected.append(candidate.copy())
            if len(selected) == count:
                break
    return selected


def _sample_target_candidates(
    target_indices: Sequence[int],
    obstacle_centres: np.ndarray,
    config: PccGenerationConfig,
) -> tuple[np.ndarray, np.ndarray, list[np.random.Generator]]:
    anchors = np.empty((len(target_indices), 6), dtype=np.float64)
    targets = np.empty((len(target_indices), 3), dtype=np.float64)
    random_generators = [
        np.random.default_rng(
            np.random.SeedSequence([config.seed, int(target_index)])
        )
        for target_index in target_indices
    ]
    pending = np.arange(len(target_indices))
    for _ in range(10_000):
        sampled = np.stack(
            [
                _sample_configuration_batch(
                    random_generators[index], 1, config.max_bending
                )[0]
                for index in pending
            ]
        )
        clearances, sampled_tips = _configuration_clearances_and_tips(
            sampled,
            obstacle_centres,
            config,
            clearance_margin=config.repulsion_clearance_margin,
        )
        clear = clearances >= 0
        still_pending: list[int] = []
        for row, index in enumerate(pending):
            anchor = sampled[row]
            tip = sampled_tips[row]
            radial = float(np.linalg.norm(tip[:2]))
            if (
                config.radial_min <= radial <= config.radial_max
                and config.z_min <= tip[2] <= config.z_max
                and clear[row]
            ):
                anchors[index] = anchor
                targets[index] = tip
            else:
                still_pending.append(int(index))
        if not still_pending:
            return targets, anchors, random_generators
        pending = np.asarray(still_pending, dtype=np.int64)
    raise RuntimeError("unable to sample targets inside the workspace bounds")


def _solve_terminal_configurations(
    target_tips: np.ndarray,
    anchor_configurations: np.ndarray,
    obstacle_centres: np.ndarray,
    config: PccGenerationConfig,
    random_generators: Sequence[np.random.Generator],
) -> list[np.ndarray]:
    """Return exactly the requested number of valid terminal IK modes per target."""
    device = _resolve_device(config.device)
    dtype = _solver_dtype(device)
    batch_size = len(target_tips)
    seeds = np.stack(
        [
            _sample_configuration_batch(
                random_generators[index],
                config.sample_batch_size,
                config.max_bending,
            )
            for index in range(batch_size)
        ]
    )
    seeds[:, 0] = anchor_configurations
    target_tensor = torch.as_tensor(
        target_tips, dtype=dtype, device=device
    )[:, None, :]
    solved, success, errors = _damped_newton(
        torch.as_tensor(seeds, dtype=dtype, device=device), target_tensor, config
    )
    solved_np = solved.detach().cpu().numpy()
    success_np = success.detach().cpu().numpy()
    errors_np = errors.detach().cpu().numpy()
    clear = _configurations_are_clear(
        solved_np.reshape(-1, 6),
        obstacle_centres,
        config,
        clearance_margin=config.repulsion_clearance_margin,
    ).reshape(success_np.shape)
    success_np &= clear

    selected: list[np.ndarray] = []
    for index in range(batch_size):
        valid = solved_np[index, success_np[index]]
        valid_errors = errors_np[index, success_np[index]]
        distinct = _select_distinct_configurations(
            valid, valid_errors, config.separation, config.terminal_count
        )
        selected.append(
            np.stack(distinct)
            if len(distinct) == config.terminal_count
            else np.empty((0, 6), dtype=np.float64)
        )

    return selected


def _sample_start_configurations(
    target_tips: np.ndarray,
    obstacle_centres: np.ndarray,
    config: PccGenerationConfig,
    random_generators: Sequence[np.random.Generator],
) -> list[np.ndarray]:
    """Select a fixed-size start set for every target in one layout batch."""
    if len(random_generators) != len(target_tips):
        raise ValueError("random_generators must contain one generator per target")
    if not len(target_tips):
        return []
    zero = np.zeros(6, dtype=np.float64)
    include_zero = bool(
        _configurations_are_clear(
            zero[None],
            obstacle_centres,
            config,
            clearance_margin=config.repulsion_clearance_margin,
        )[0]
    )
    random_count = config.start_count - int(include_zero)
    if random_count == 0:
        return [zero[None].copy() for _ in target_tips]

    accepted: list[list[np.ndarray]] = [[] for _ in range(len(target_tips))]
    pools = np.stack(
        [
            _sample_configuration_batch(
                random_generators[index], config.sample_batch_size, config.max_bending
            )
            for index in range(len(target_tips))
        ]
    )
    clearances, tips = _configuration_clearances_and_tips(
        pools,
        obstacle_centres,
        config,
        clearance_margin=config.repulsion_clearance_margin,
    )
    clear = clearances >= 0
    radial = np.linalg.norm(tips[..., :2], axis=-1)
    eligible = clear
    eligible &= (radial >= config.radial_min) & (radial <= config.radial_max)
    eligible &= (tips[..., 2] >= config.z_min) & (tips[..., 2] <= config.z_max)
    eligible &= (
        np.linalg.norm(tips - target_tips[:, None], axis=-1)
        >= config.min_path_distance
    )
    eligible &= np.linalg.norm(pools, axis=-1) >= config.separation

    for target_index in range(len(target_tips)):
        for candidate in pools[target_index, eligible[target_index]]:
            if any(
                np.linalg.norm(candidate - other) < config.separation
                for other in accepted[target_index]
            ):
                continue
            accepted[target_index].append(candidate.copy())
            if len(accepted[target_index]) == random_count:
                break

    result: list[np.ndarray] = []
    for values in accepted:
        if len(values) < random_count:
            result.append(np.empty((0, 6), dtype=np.float64))
            continue
        random_starts = np.stack(values)
        result.append(
            np.concatenate((zero[None], random_starts))
            if include_zero
            else random_starts
        )
    return result


def _fixed_obstacle_centres(
    count: int,
    config: PccGenerationConfig,
) -> np.ndarray:
    """Return the prescribed fixed-height obstacle layout for one count."""
    if count == 0:
        return np.empty((0, 3), dtype=np.float64)
    z = config.obstacle_z
    if count == 1:
        return np.asarray([[0.0, 0.0, z]], dtype=np.float64)
    offset = config.obstacle_layout_radius / math.sqrt(2.0)
    if count == 2:
        return np.asarray(
            [[-offset, -offset, z], [offset, offset, z]], dtype=np.float64
        )
    return np.asarray(
        [
            [-offset, -offset, z],
            [-offset, offset, z],
            [offset, -offset, z],
            [offset, offset, z],
        ],
        dtype=np.float64,
    )


def _configuration_clearances_and_tips(
    configurations: np.ndarray,
    obstacle_centres: np.ndarray,
    config: PccGenerationConfig,
    *,
    clearance_margin: float,
    compute_device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate sampled-backbone clearances and tips in one device call."""
    configurations = np.asarray(configurations, dtype=np.float64)
    if configurations.ndim < 2 or configurations.shape[-1] != 6:
        raise ValueError("configurations must have shape [...,6]")
    device = (
        _resolve_device(config.device)
        if compute_device is None
        else compute_device
    )
    dtype = _solver_dtype(device)
    with torch.no_grad():
        configuration_tensor = torch.as_tensor(
            configurations, dtype=dtype, device=device
        )
        if len(obstacle_centres):
            centre_tensor = torch.as_tensor(
                obstacle_centres, dtype=dtype, device=device
            )
            clearances, tips = sphere_clearances_and_tips(
                configuration_tensor,
                centre_tensor,
                config.section_lengths,
                config.sphere_radius,
                clearance_margin,
            )
        else:
            tips = tip_position(configuration_tensor, config.section_lengths)
            clearances = torch.full(
                configurations.shape[:-1],
                torch.inf,
                dtype=dtype,
                device=device,
            )
    return (
        clearances.detach().cpu().numpy(),
        tips.detach().cpu().numpy(),
    )


def _configurations_are_clear(
    configurations: np.ndarray,
    obstacle_centres: np.ndarray,
    config: PccGenerationConfig,
    *,
    clearance_margin: float,
    compute_device: torch.device | None = None,
) -> np.ndarray:
    """Check a flat configuration batch without unnecessary obstacle-free FK."""
    configurations = np.asarray(configurations, dtype=np.float64)
    if configurations.ndim != 2 or configurations.shape[1] != 6:
        raise ValueError("configurations must have shape [N,6]")
    if len(obstacle_centres) == 0:
        return np.isfinite(configurations).all(axis=1)
    clearances, _ = _configuration_clearances_and_tips(
        configurations,
        obstacle_centres,
        config,
        clearance_margin=clearance_margin,
        compute_device=compute_device,
    )
    return np.isfinite(clearances) & (clearances >= 0)


def _path_numerical_bending_validity(
    paths: np.ndarray,
    config: PccGenerationConfig,
) -> np.ndarray:
    """Validate numerical and section-bending path constraints."""
    valid = np.isfinite(paths).all(axis=(1, 2))
    valid &= np.all(
        np.linalg.norm(paths.reshape(len(paths), config.horizon, 3, 2), axis=-1)
        <= config.max_bending + 1e-6,
        axis=(1, 2),
    )
    return valid


def _path_adjacent_step_validity(
    paths: np.ndarray,
    config: PccGenerationConfig,
) -> np.ndarray:
    return np.all(
        np.linalg.norm(np.diff(paths, axis=1), axis=-1)
        <= config.max_adjacent_step,
        axis=1,
    )


def _path_noncollision_validity(
    paths: np.ndarray,
    config: PccGenerationConfig,
) -> np.ndarray:
    return _path_numerical_bending_validity(
        paths, config
    ) & _path_adjacent_step_validity(paths, config)


def _path_collision_free(
    paths: np.ndarray,
    obstacle_centres: np.ndarray,
    config: PccGenerationConfig,
) -> np.ndarray:
    clear = _configurations_are_clear(
        paths.reshape(-1, 6),
        obstacle_centres,
        config,
        clearance_margin=config.clearance_margin,
    ).reshape(len(paths), config.horizon)
    return np.all(clear, axis=1)


def _straight_workspace_path_candidates(
    start_configurations: np.ndarray,
    goal_configurations: np.ndarray,
    target_tips: np.ndarray,
    config: PccGenerationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the interior points of several straight Cartesian tip paths."""
    device = _resolve_device(config.device)
    dtype = _solver_dtype(device)
    fractions = torch.linspace(0, 1, config.horizon, dtype=dtype, device=device)
    interior_fractions = fractions[1:-1]
    start = torch.as_tensor(start_configurations, dtype=dtype, device=device)
    goal = torch.as_tensor(goal_configurations, dtype=dtype, device=device)
    start_tip = tip_position(start, config.section_lengths)
    target = torch.as_tensor(target_tips, dtype=dtype, device=device)
    if target.shape != start_tip.shape:
        raise ValueError("target_tips must have shape [B,3]")
    initial = start[:, None, :] + interior_fractions[None, :, None] * (
        goal - start
    )[:, None, :]
    desired = start_tip[:, None, :] + interior_fractions[None, :, None] * (
        target - start_tip
    )[:, None, :]
    solved, success, _ = _continuity_aware_path_ik(
        initial,
        desired,
        start,
        goal,
        config,
    )
    paths = np.empty(
        (len(start_configurations), config.horizon, 6), dtype=np.float64
    )
    paths[:, 0] = start_configurations
    paths[:, 1:-1] = solved.detach().cpu().numpy()
    paths[:, -1] = goal_configurations

    ik_valid = success.all(dim=1).detach().cpu().numpy()
    numerical_bending_valid = _path_numerical_bending_validity(paths, config)
    adjacent_valid = _path_adjacent_step_validity(paths, config)
    return paths, ik_valid, numerical_bending_valid, adjacent_valid


def _point_to_chord_distances(
    points: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    chord = end - start
    squared_length = float(chord @ chord)
    if squared_length <= 1e-24:
        return np.linalg.norm(points - start, axis=-1)
    fractions = np.clip(((points - start) @ chord) / squared_length, 0.0, 1.0)
    projections = start + fractions[:, None] * chord
    return np.linalg.norm(points - projections, axis=-1)


def _douglas_peucker_indices(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Select workspace controls, retaining at least four when available."""
    point_count = len(points)
    retained = {0, point_count - 1}
    pending = [(0, point_count - 1)]
    while pending:
        first, last = pending.pop()
        if last <= first + 1:
            continue
        distances = _point_to_chord_distances(
            points[first + 1 : last], points[first], points[last]
        )
        offset = int(np.argmax(distances))
        if distances[offset] <= tolerance:
            continue
        split = first + 1 + offset
        retained.add(split)
        pending.extend(((first, split), (split, last)))
    while len(retained) < min(4, point_count):
        indices = sorted(retained)
        first, last = max(
            zip(indices[:-1], indices[1:]), key=lambda pair: pair[1] - pair[0]
        )
        retained.add((first + last) // 2)
    return np.asarray(sorted(retained), dtype=np.int64)


def _clamped_bspline_knots(control_count: int, degree: int = 3) -> np.ndarray:
    interior_count = control_count - degree - 1
    interior = np.arange(1, interior_count + 1) / (interior_count + 1)
    return np.concatenate((np.zeros(degree + 1), interior, np.ones(degree + 1)))


def _evaluate_clamped_bspline_batch(
    controls: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    """Evaluate equally sized clamped cubic B-splines as one NumPy batch."""
    batch_size, control_count, _ = controls.shape
    degree = min(3, control_count - 1)
    knots = _clamped_bspline_knots(control_count, degree)
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.ndim == 1:
        parameters = np.broadcast_to(parameters, (batch_size, len(parameters)))
    evaluation = np.clip(parameters, 0.0, np.nextafter(1.0, 0.0))
    basis = (
        (evaluation[..., None] >= knots[:-1])
        & (evaluation[..., None] < knots[1:])
    ).astype(np.float64)
    for level in range(1, degree + 1):
        count = basis.shape[-1] - 1
        left_denominator = knots[level : level + count] - knots[:count]
        right_denominator = (
            knots[level + 1 : level + 1 + count] - knots[1 : 1 + count]
        )
        left = np.divide(
            evaluation[..., None] - knots[:count],
            left_denominator,
            out=np.zeros(basis.shape[:-1] + (count,)),
            where=left_denominator != 0.0,
        )
        right = np.divide(
            knots[level + 1 : level + 1 + count] - evaluation[..., None],
            right_denominator,
            out=np.zeros(basis.shape[:-1] + (count,)),
            where=right_denominator != 0.0,
        )
        basis = left * basis[..., :count] + right * basis[..., 1:]
    return np.einsum("bmc,bcd->bmd", basis, controls)


def _smooth_tip_paths(
    tip_paths: np.ndarray,
    config: PccGenerationConfig,
) -> np.ndarray:
    """Simplify, smooth, and workspace-arc-length resample tip paths."""
    groups: dict[int, list[tuple[int, np.ndarray]]] = {}
    for index, points in enumerate(tip_paths):
        retained = _douglas_peucker_indices(points, config.spline_dp_tolerance)
        groups.setdefault(len(retained), []).append((index, points[retained]))
    result = np.empty_like(tip_paths)
    dense_parameters = np.linspace(0.0, 1.0, max(256, 16 * config.horizon))
    for entries in groups.values():
        indices = [entry[0] for entry in entries]
        controls = np.stack([entry[1] for entry in entries])
        dense = _evaluate_clamped_bspline_batch(controls, dense_parameters)
        segment_lengths = np.linalg.norm(np.diff(dense, axis=1), axis=-1)
        cumulative = np.concatenate(
            (np.zeros((len(entries), 1)), np.cumsum(segment_lengths, axis=1)),
            axis=1,
        )
        parameters = np.stack(
            [
                np.interp(
                    np.linspace(0.0, lengths[-1], config.horizon),
                    lengths,
                    dense_parameters,
                )
                for lengths in cumulative
            ]
        )
        smoothed = _evaluate_clamped_bspline_batch(controls, parameters)
        smoothed[:, 0] = controls[:, 0]
        smoothed[:, -1] = controls[:, -1]
        result[indices] = smoothed
    return result


def _deepest_repulsion_updates(
    configurations: torch.Tensor,
    obstacle_centres: torch.Tensor,
    config: PccGenerationConfig,
) -> torch.Tensor:
    """Map the deepest inflated-clearance penetration per frame into q-space."""
    shape = configurations.shape
    flattened_points = backbone_points(
        configurations.reshape(-1, 6), config.section_lengths
    )[:, 1:]
    points = flattened_points.reshape(
        shape[:-1] + (flattened_points.shape[-2], 3)
    )
    offsets = points.unsqueeze(-2) - obstacle_centres[None, None, None]
    distances = torch.linalg.vector_norm(offsets, dim=-1)
    safe_radius = config.sphere_radius + config.repulsion_clearance_margin
    penetration = safe_radius - distances
    flat_penetration = penetration.flatten(start_dim=2)
    deepest, flat_indices = flat_penetration.max(dim=-1)
    obstacle_count = len(obstacle_centres)
    point_indices = torch.div(flat_indices, obstacle_count, rounding_mode="floor")
    obstacle_indices = flat_indices.remainder(obstacle_count)
    batch = torch.arange(len(configurations), device=configurations.device)[:, None]
    frames = torch.arange(configurations.shape[1], device=configurations.device)[None]
    selected_offsets = offsets[batch, frames, point_indices, obstacle_indices]
    active = deepest > 0
    active_configurations = configurations[active]
    active_point_indices = point_indices[active]
    selected_jacobians = backbone_point_jacobians(
        active_configurations,
        active_point_indices,
        config.section_lengths,
    )

    active_offsets = selected_offsets[active]
    xy = active_offsets[..., :2]
    xy_distance = torch.linalg.vector_norm(xy, dim=-1)
    z_offset = active_offsets[..., 2]
    required_xy = torch.sqrt(
        torch.clamp(safe_radius**2 - z_offset.square(), min=0.0)
    )
    xy_error = (
        torch.clamp(required_xy - xy_distance, min=0.0)
        / torch.clamp(xy_distance, min=1e-12)
    )[..., None] * xy
    vertical = xy_distance <= 1e-12
    z_direction = torch.where(z_offset < 0, -1.0, 1.0)
    z_error = torch.where(
        vertical,
        z_direction * torch.clamp(safe_radius - z_offset.abs(), min=0.0),
        torch.zeros_like(z_offset),
    )
    error = torch.cat((xy_error, z_error[..., None]), dim=-1)
    identity = torch.eye(3, dtype=configurations.dtype, device=configurations.device)
    system = (
        selected_jacobians @ selected_jacobians.transpose(-1, -2)
        + config.ik_damping * identity
    )
    active_update = (
        selected_jacobians.transpose(-1, -2)
        @ torch.linalg.solve(system, error.unsqueeze(-1))
    ).squeeze(-1)
    update = torch.zeros_like(configurations)
    update[active] = active_update
    update[:, 0] = 0
    update[:, -1] = 0
    return update


def _repair_colliding_paths(
    paths: np.ndarray,
    obstacle_centres: np.ndarray,
    config: PccGenerationConfig,
) -> list[np.ndarray | None]:
    """Iteratively repel colliding backbones and re-solve smoothed tip paths."""
    device = _resolve_device(config.device)
    dtype = _solver_dtype(device)
    centres = torch.as_tensor(obstacle_centres, dtype=dtype, device=device)
    active_indices = np.arange(len(paths))
    active = torch.as_tensor(paths, dtype=dtype, device=device)
    repaired: list[np.ndarray | None] = [None] * len(paths)
    for _ in range(config.max_collision_repairs):
        with torch.no_grad():
            repelled = _project_section_discs(
                active + _deepest_repulsion_updates(active, centres, config),
                config.max_bending,
            )
            repelled[:, 0] = active[:, 0]
            repelled[:, -1] = active[:, -1]
            repelled_tips = tip_position(repelled, config.section_lengths)
        desired = _smooth_tip_paths(repelled_tips.detach().cpu().numpy(), config)
        solved, success, _ = _continuity_aware_path_ik(
            repelled[:, 1:-1],
            torch.as_tensor(desired[:, 1:-1], dtype=dtype, device=device),
            repelled[:, 0],
            repelled[:, -1],
            config,
        )
        candidate = repelled.clone()
        candidate[:, 1:-1] = solved
        candidate_np = candidate.detach().cpu().numpy()
        reliable = success.all(dim=1).detach().cpu().numpy()
        reliable &= _path_numerical_bending_validity(candidate_np, config)
        adjacent = _path_adjacent_step_validity(candidate_np, config)
        clear = _path_collision_free(candidate_np, obstacle_centres, config)
        for local_index in np.flatnonzero(reliable & adjacent & clear):
            repaired[active_indices[local_index]] = candidate_np[local_index]
        keep = reliable & ~clear
        if not np.any(keep):
            break
        active_indices = active_indices[keep]
        active = candidate[torch.as_tensor(keep, device=device)]
    return repaired


def _plan_layout_paths(
    target_tips: np.ndarray,
    endpoint_configurations: Sequence[np.ndarray],
    start_configurations: Sequence[np.ndarray],
    obstacle_centres: np.ndarray,
    config: PccGenerationConfig,
) -> tuple[list[tuple[list[np.ndarray], list[int], list[int]]], dict[str, int]]:
    """Plan globally batched paths for targets sharing one obstacle layout."""
    target_count = len(target_tips)
    if not (
        len(endpoint_configurations) == target_count
        and len(start_configurations) == target_count
    ):
        raise ValueError("targets, endpoints, and starts must have equal lengths")

    pair_starts: list[np.ndarray] = []
    pair_goals: list[np.ndarray] = []
    pair_targets: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    start_ids: list[np.ndarray] = []
    ik_ids: list[np.ndarray] = []
    for target_row, (target, starts, endpoints) in enumerate(
        zip(target_tips, start_configurations, endpoint_configurations)
    ):
        start_count = len(starts)
        terminal_count = len(endpoints)
        pair_count = start_count * terminal_count
        pair_starts.append(np.repeat(starts, terminal_count, axis=0))
        pair_goals.append(np.tile(endpoints, (start_count, 1)))
        pair_targets.append(np.broadcast_to(target, (pair_count, 3)))
        target_rows.append(np.full(pair_count, target_row, dtype=np.int64))
        start_ids.append(np.repeat(np.arange(start_count), terminal_count))
        ik_ids.append(np.tile(np.arange(terminal_count), start_count))

    pair_starts_array = np.concatenate(pair_starts)
    pair_goals_array = np.concatenate(pair_goals)
    pair_targets_array = np.concatenate(pair_targets)
    target_rows_array = np.concatenate(target_rows)
    start_ids_array = np.concatenate(start_ids)
    ik_ids_array = np.concatenate(ik_ids)
    pair_count = len(pair_starts_array)
    candidates = np.empty((pair_count, config.horizon, 6), dtype=np.float64)
    ik_valid = np.empty(pair_count, dtype=bool)
    numerical_bending_valid = np.empty(pair_count, dtype=bool)
    adjacent_valid = np.empty(pair_count, dtype=bool)
    clear = np.empty(pair_count, dtype=bool)

    for begin in range(0, pair_count, config.path_batch_size):
        end = min(begin + config.path_batch_size, pair_count)
        (
            candidates[begin:end],
            ik_valid[begin:end],
            numerical_bending_valid[begin:end],
            adjacent_valid[begin:end],
        ) = _straight_workspace_path_candidates(
            pair_starts_array[begin:end],
            pair_goals_array[begin:end],
            pair_targets_array[begin:end],
            config,
        )
        clear[begin:end] = _path_collision_free(
            candidates[begin:end], obstacle_centres, config
        )

    reliable = ik_valid & numerical_bending_valid
    accepted: list[np.ndarray | None] = [None] * len(candidates)
    for index in np.flatnonzero(reliable & adjacent_valid & clear):
        accepted[index] = candidates[index]

    repair_indices = np.flatnonzero(reliable & ~clear)
    for begin in range(0, len(repair_indices), config.path_batch_size):
        indices = repair_indices[begin : begin + config.path_batch_size]
        repaired = _repair_colliding_paths(
            candidates[indices], obstacle_centres, config
        )
        for index, path in zip(indices, repaired):
            if path is not None:
                accepted[index] = path
    failures = {
        "workspace_ik": int(np.count_nonzero(~ik_valid)),
        "path_constraints": int(
            np.count_nonzero(
                ik_valid
                & (
                    ~numerical_bending_valid
                    | (clear & ~adjacent_valid)
                )
            )
        ),
        "collision_repair": sum(
            accepted[index] is None for index in repair_indices
        ),
    }

    retained = [([], [], []) for _ in range(target_count)]
    for target_row, start_id, ik_id, path in zip(
        target_rows_array, start_ids_array, ik_ids_array, accepted
    ):
        if path is not None:
            paths, retained_starts, retained_iks = retained[target_row]
            paths.append(path)
            retained_starts.append(int(start_id))
            retained_iks.append(int(ik_id))
    return retained, failures


def _split_for_target(target_index: int, config: PccGenerationConfig) -> str:
    digest = hashlib.sha256(
        f"{config.split_seed}:{target_index}".encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:8], "little") / 2**64
    if value < config.train_fraction:
        return "train"
    if value < config.train_fraction + config.validation_fraction:
        return "validation"
    return "test"


def _prepare_output_directory(config: PccGenerationConfig) -> Path:
    output = Path(config.output_dir).expanduser().resolve()
    target_files = (output / "dataset.npz", output / "dataset_meta.json")
    existing = [path for path in target_files if path.exists()]
    if existing and not config.overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(
            f"dataset output already exists in {output}: {names}; pass --overwrite"
        )
    if config.overwrite:
        for path in existing:
            path.unlink()
    output.mkdir(parents=True, exist_ok=True)
    return output


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _configuration_for_metadata(config: PccGenerationConfig) -> dict[str, object]:
    values = asdict(config)
    values.pop("output_dir")
    values.pop("section_lengths")
    return values


def _generate_obstacle_count_batch(
    target_indices: Sequence[int],
    obstacle_count: int,
    config: PccGenerationConfig,
) -> tuple[
    dict[int, SceneResult],
    dict[str, int],
    dict[str, int],
]:
    """Generate one batch whose targets share a prescribed obstacle layout."""
    completed: dict[int, SceneResult] = {}
    path_failures = {reason: 0 for reason in PATH_FAILURE_REASONS}
    target_discards = {reason: 0 for reason in TARGET_DISCARD_REASONS}
    centres = _fixed_obstacle_centres(obstacle_count, config)
    indices = [int(index) for index in target_indices]
    targets, anchors, random_generators = _sample_target_candidates(
        indices, centres, config
    )
    endpoints = _solve_terminal_configurations(
        targets,
        anchors,
        centres,
        config,
        random_generators,
    )
    valid_rows = [row for row, values in enumerate(endpoints) if len(values)]
    target_discards["terminal_ik"] += len(indices) - len(valid_rows)
    starts = _sample_start_configurations(
        targets[valid_rows],
        centres,
        config,
        [random_generators[row] for row in valid_rows],
    )
    planning_rows: list[int] = []
    planning_starts: list[np.ndarray] = []
    for row, start_values in zip(valid_rows, starts):
        if not len(start_values):
            target_discards["start_sampling"] += 1
            continue
        planning_rows.append(row)
        planning_starts.append(start_values)

    if planning_rows:
        retained_results, planning_failures = _plan_layout_paths(
            targets[planning_rows],
            [endpoints[row] for row in planning_rows],
            planning_starts,
            centres,
            config,
        )
    else:
        retained_results = []
        planning_failures = {reason: 0 for reason in PATH_FAILURE_REASONS}
    for reason, count in planning_failures.items():
        path_failures[reason] += count

    for row, (retained, start_ids, ik_ids) in zip(
        planning_rows, retained_results
    ):
        if not retained:
            target_discards["zero_retained_paths"] += 1
            continue
        target_index = indices[row]
        completed[target_index] = targets[row], retained, start_ids, ik_ids, centres
    return completed, path_failures, target_discards


def generate_dataset(
    config: PccGenerationConfig,
    progress: ProgressCallback | None = None,
) -> GenerationSummary:
    """Generate balanced static-sphere scenes and collision-free PCC paths."""
    config = config.validated()
    output = _prepare_output_directory(config)
    started = time.monotonic()
    obstacle_counts = config.obstacle_counts
    marker = {
        "format": "pcc_diffuser_npz",
        "version": DATASET_FORMAT_VERSION,
        "complete": False,
        "requested_targets": config.target_samples,
        "requested_paths": config.maximum_path_count,
    }
    _write_json(output / "dataset_meta.json", marker)

    paths: list[np.ndarray] = []
    target_tips: list[np.ndarray] = []
    target_indices: list[int] = []
    start_indices: list[int] = []
    ik_indices: list[int] = []
    split_indices: list[int] = []
    obstacle_centres: list[np.ndarray] = []
    obstacle_offsets = [0]
    path_failure_counts = {reason: 0 for reason in PATH_FAILURE_REASONS}
    target_discarded_counts = {reason: 0 for reason in TARGET_DISCARD_REASONS}
    scene_counts = {str(count): 0 for count in obstacle_counts}
    path_counts = {str(count): 0 for count in obstacle_counts}

    targets_per_layout = config.target_samples // len(obstacle_counts)
    batches_per_layout = math.ceil(targets_per_layout / config.target_batch_size)
    batch_count = len(obstacle_counts) * batches_per_layout
    batch_index = 0
    processed_targets = 0
    for obstacle_position, obstacle_count in enumerate(obstacle_counts):
        layout_indices = list(
            range(obstacle_position, config.target_samples, len(obstacle_counts))
        )
        for batch_begin in range(0, len(layout_indices), config.target_batch_size):
            batch_indices = layout_indices[
                batch_begin : batch_begin + config.target_batch_size
            ]
            (
                batch_results,
                batch_failures,
                batch_discards,
            ) = _generate_obstacle_count_batch(
                batch_indices, obstacle_count, config
            )
            for reason, count in batch_failures.items():
                path_failure_counts[reason] += count
            for reason, count in batch_discards.items():
                target_discarded_counts[reason] += count

            batch_path_count = 0
            for target_index in batch_indices:
                if target_index not in batch_results:
                    continue
                target, retained, starts, iks, centres = batch_results[
                    target_index
                ]
                split = _split_for_target(target_index, config)
                for path, start_id, ik_id in zip(retained, starts, iks):
                    paths.append(np.asarray(path, dtype=np.float32))
                    target_tips.append(np.asarray(target, dtype=np.float32))
                    target_indices.append(target_index)
                    start_indices.append(start_id)
                    ik_indices.append(ik_id)
                    split_indices.append(SPLIT_CODES[split])
                    obstacle_centres.append(np.asarray(centres, dtype=np.float32))
                    obstacle_offsets.append(obstacle_offsets[-1] + len(centres))
                scene_counts[str(obstacle_count)] += 1
                path_counts[str(obstacle_count)] += len(retained)
                batch_path_count += len(retained)

            batch_index += 1
            processed_targets += len(batch_indices)
            if progress is not None:
                percentage = 100 * processed_targets / config.target_samples
                batch_end = batch_begin + len(batch_indices)
                requested_batch_paths = (
                    len(batch_indices) * config.maximum_paths_per_target
                )
                progress(
                    f"batch {batch_index}/{batch_count} complete | "
                    f"obstacles {obstacle_count} | "
                    f"targets {batch_begin + 1}-{batch_end}/{config.target_samples} | "
                    f"paths {batch_path_count}/{requested_batch_paths} | "
                    f"{percentage:.1f}%"
                )

    path_array = (
        np.stack(paths).astype(np.float32)
        if paths
        else np.empty((0, config.horizon, 6), dtype=np.float32)
    )
    target_array = (
        np.stack(target_tips).astype(np.float32)
        if target_tips
        else np.empty((0, 3), dtype=np.float32)
    )
    packed_centres = (
        np.concatenate(obstacle_centres, axis=0).astype(np.float32)
        if any(len(value) for value in obstacle_centres)
        else np.empty((0, 3), dtype=np.float32)
    )
    split_array = np.asarray(split_indices, dtype=np.int8)
    np.savez_compressed(
        output / "dataset.npz",
        paths=path_array,
        target_tips=target_array,
        target_indices=np.asarray(target_indices, dtype=np.int32),
        start_indices=np.asarray(start_indices, dtype=np.int16),
        ik_indices=np.asarray(ik_indices, dtype=np.int16),
        split_indices=split_array,
        obstacle_centres=packed_centres,
        obstacle_offsets=np.asarray(obstacle_offsets, dtype=np.int64),
    )
    elapsed = time.monotonic() - started
    split_counts = {
        name: int(np.count_nonzero(split_array == code))
        for name, code in SPLIT_CODES.items()
    }
    actual_targets = sum(scene_counts.values())
    target_discarded_total = sum(target_discarded_counts.values())
    if target_discarded_total != config.target_samples - actual_targets:
        raise RuntimeError(
            "internal target accounting error: discard counts do not explain "
            "requested_targets - actual_targets"
        )
    target_discarded_counts["total"] = target_discarded_total
    path_failure_counts["total"] = sum(path_failure_counts.values())
    unplanned_paths = config.maximum_paths_per_target * (
        target_discarded_counts["terminal_ik"]
        + target_discarded_counts["start_sampling"]
    )
    expected_failures = config.maximum_path_count - len(paths) - unplanned_paths
    if path_failure_counts["total"] != expected_failures:
        raise RuntimeError(
            "internal path accounting error: failure counts do not explain "
            "requested_paths - actual_paths"
        )
    metadata = {
        "format": "pcc_diffuser_npz",
        "version": DATASET_FORMAT_VERSION,
        "complete": True,
        "units": {"length": "normalised", "configuration": "radian"},
        "section_lengths": list(config.section_lengths),
        "sphere_radius": config.sphere_radius,
        "clearance_margin": config.clearance_margin,
        "obstacle_layout_radius": config.obstacle_layout_radius,
        "requested_targets": config.target_samples,
        "actual_targets": actual_targets,
        "requested_paths": config.maximum_path_count,
        "actual_paths": len(paths),
        "obstacle_scene_counts": scene_counts,
        "obstacle_path_counts": path_counts,
        "planning_policy": {
            "obstacle_free": ["workspace_ik"],
            "obstacle": [
                "workspace_ik",
                "deepest_backbone_point_repulsion",
                "douglas_peucker_control_selection",
                "clamped_cubic_bspline_workspace_resampling",
                "batched_continuity_aware_path_ik",
            ],
        },
        "collision_policy": {
            "endpoint_region": "sphere_radius + repulsion_clearance_factor * clearance_margin",
            "repair_region": "sphere_radius + repulsion_clearance_factor * clearance_margin",
            "repulsion": "deepest_point_per_path_frame",
            "repulsion_plane": "xy_except_vertical_degeneracy",
            "acceptance_region": "sphere_radius + clearance_margin",
        },
        "maximum_adjacent_configuration_step": config.max_adjacent_step,
        "split_codes": SPLIT_CODES,
        "split_counts": split_counts,
        "total_elapsed_seconds": elapsed,
        "target_discarded_counts": target_discarded_counts,
        "path_failure_counts": path_failure_counts,
        "config": _configuration_for_metadata(config),
        "arrays": {
            "paths": list(path_array.shape),
            "target_tips": list(target_array.shape),
            "target_indices": [len(target_indices)],
            "start_indices": [len(start_indices)],
            "ik_indices": [len(ik_indices)],
            "split_indices": [len(split_indices)],
            "obstacle_centres": list(packed_centres.shape),
            "obstacle_offsets": [len(obstacle_offsets)],
        },
        "resolved_device": str(_resolve_device(config.device)),
        "solver_dtype": str(_solver_dtype(_resolve_device(config.device))).removeprefix("torch."),
    }
    metadata["workspace_path_ik"] = {
        "primary_task": "tip_position",
        "secondary_task": "adjacent_configuration_continuity",
        "primary_method": "damped_Jacobian_pseudoinverse",
        "secondary_method": "Jacobian_null_space_projection",
        "continuity_gain": config.ik_continuity_gain,
        "convergence_check_interval": config.ik_convergence_interval,
        "batch_axes": ["cross_target_path", "interior_frame"],
    }
    _write_json(output / "dataset_meta.json", metadata)
    return GenerationSummary(
        output,
        len(paths),
        actual_targets,
        path_failure_counts,
        elapsed,
    )


__all__ = [
    "GenerationSummary",
    "PccGenerationConfig",
    "generate_dataset",
]
