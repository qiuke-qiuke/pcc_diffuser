"""Dataset-generation repulsion planner exposed as an evaluation baseline."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from pcc_diffuser.evaluation import PathPlanner
from pcc_diffuser.generation import (
    PccGenerationConfig,
    _configurations_are_clear,
    _damped_newton,
    _path_collision_free,
    _repair_colliding_paths,
    _resolve_device,
    _sample_configuration_batch,
    _solver_dtype,
    _straight_workspace_path_candidates,
)


class RepulsionPlanner(PathPlanner):
    """Plan with terminal IK, straight-workspace IK, and collision repulsion."""

    def __init__(self, config: PccGenerationConfig) -> None:
        self.config = config.validated()
        self.device = _resolve_device(config.device)
        self.dtype = _solver_dtype(self.device)

    @property
    def name(self) -> str:
        return "repulsion"

    @property
    def horizon(self) -> int:
        return self.config.horizon

    def config_dict(self) -> dict[str, object]:
        values = asdict(self.config)
        values.pop("output_dir")
        values.pop("overwrite")
        return values

    def _solve_terminal(
        self,
        target_tip: np.ndarray,
        obstacle_centres: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        seeds = _sample_configuration_batch(
            rng, self.config.sample_batch_size, self.config.max_bending
        )
        desired = torch.as_tensor(
            target_tip, dtype=self.dtype, device=self.device
        )[None]
        solved, success, errors = _damped_newton(
            torch.as_tensor(seeds, dtype=self.dtype, device=self.device),
            desired,
            self.config,
        )
        solved_np = solved.detach().cpu().numpy()
        valid = success.detach().cpu().numpy()
        valid &= _configurations_are_clear(
            solved_np,
            obstacle_centres,
            self.config,
            clearance_margin=self.config.repulsion_clearance_margin,
            compute_device=self.device,
        )
        indices = np.flatnonzero(valid)
        if not len(indices):
            return None
        errors_np = errors.detach().cpu().numpy()
        return solved_np[indices[np.argmin(errors_np[indices])]].copy()

    def plan(
        self,
        start: np.ndarray,
        target_tip: np.ndarray,
        obstacle_centres: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        start = np.asarray(start, dtype=np.float64)
        target_tip = np.asarray(target_tip, dtype=np.float64)
        obstacle_centres = np.asarray(obstacle_centres, dtype=np.float64)
        terminal = self._solve_terminal(target_tip, obstacle_centres, rng)
        if terminal is None:
            return None
        paths, ik_valid, bending_valid, adjacent_valid = (
            _straight_workspace_path_candidates(
                start[None], terminal[None], target_tip[None], self.config
            )
        )
        if not bool(ik_valid[0] and bending_valid[0] and adjacent_valid[0]):
            return None
        if _path_collision_free(paths, obstacle_centres, self.config)[0]:
            return paths[0]
        repaired = _repair_colliding_paths(paths, obstacle_centres, self.config)[0]
        return repaired


__all__ = ["RepulsionPlanner"]
