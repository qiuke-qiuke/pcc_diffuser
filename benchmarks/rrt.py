"""Configuration- and projected-workspace RRT baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np
import torch

from pcc_diffuser.evaluation import PathPlanner
from pcc_diffuser.kinematics import sphere_clearances_and_tips, tip_position_and_jacobian


RRT_ALGORITHMS = ("c-rrt", "c-rrt-star", "w-rrt", "w-rrt-star")


@dataclass(frozen=True)
class RrtConfig:
    algorithm: str
    horizon: int
    section_lengths: tuple[float, float, float]
    max_bending: float
    max_adjacent_step: float
    radial_min: float
    radial_max: float
    z_min: float
    z_max: float
    sphere_radius: float
    clearance_margin: float
    ik_iterations: int
    ik_damping: float
    ik_tolerance: float
    max_iterations: int = 500
    goal_bias: float = 0.1
    goal_tolerance: float = 0.1
    step_size: float = 0.2
    rewire_radius: float = 0.3
    device: str = "auto"

    def validated(self) -> "RrtConfig":
        if self.algorithm not in RRT_ALGORITHMS:
            raise ValueError(f"algorithm must be one of {RRT_ALGORITHMS}")
        if self.horizon < 2 or self.max_iterations < 1 or self.ik_iterations < 1:
            raise ValueError("horizon and iteration counts must be positive")
        if not 0 <= self.goal_bias <= 1:
            raise ValueError("goal_bias must lie in [0,1]")
        positive = (
            self.max_bending,
            self.max_adjacent_step,
            self.radial_max,
            self.z_max,
            self.sphere_radius,
            self.clearance_margin,
            self.ik_damping,
            self.ik_tolerance,
            self.goal_tolerance,
            self.step_size,
            self.rewire_radius,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("RRT scalar limits must be finite and positive")
        if not 0 <= self.radial_min < self.radial_max:
            raise ValueError("radial workspace bounds must be ordered")
        if not 0 <= self.z_min < self.z_max:
            raise ValueError("z workspace bounds must be ordered")
        if len(self.section_lengths) != 3 or not all(
            math.isfinite(value) and value > 0 for value in self.section_lengths
        ):
            raise ValueError("section_lengths must contain three positive values")
        return self

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class _Node:
    q: np.ndarray
    tip: np.ndarray
    parent: int
    cost: float
    children: set[int]


class RrtPlanner(PathPlanner):
    """Run one unidirectional RRT or RRT* planning attempt."""

    def __init__(self, config: RrtConfig) -> None:
        self.config = config.validated()
        requested = config.device
        if requested == "auto":
            requested = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        self.dtype = torch.float32 if self.device.type == "cuda" else torch.float64
        self._workspace = config.algorithm.startswith("w-")
        self._star = config.algorithm.endswith("-star")

    @property
    def name(self) -> str:
        return self.config.algorithm

    @property
    def horizon(self) -> int:
        return self.config.horizon

    def _sample_configuration(self, rng: np.random.Generator) -> np.ndarray:
        radii = self.config.max_bending * np.sqrt(rng.random(3))
        angles = rng.uniform(-math.pi, math.pi, 3)
        q = np.empty(6, dtype=np.float64)
        q[0::2] = radii * np.cos(angles)
        q[1::2] = radii * np.sin(angles)
        return q

    def _sample_workspace(self, rng: np.random.Generator) -> np.ndarray:
        squared_radius = rng.uniform(
            self.config.radial_min**2, self.config.radial_max**2
        )
        angle = rng.uniform(-math.pi, math.pi)
        radius = math.sqrt(squared_radius)
        return np.asarray(
            (
                radius * math.cos(angle),
                radius * math.sin(angle),
                rng.uniform(self.config.z_min, self.config.z_max),
            ),
            dtype=np.float64,
        )

    def _project_bending(self, q: torch.Tensor) -> torch.Tensor:
        pairs = q.reshape(3, 2)
        norms = torch.linalg.vector_norm(pairs, dim=-1, keepdim=True)
        scale = torch.clamp(
            self.config.max_bending / torch.clamp(norms, min=1e-12), max=1.0
        )
        return (pairs * scale).reshape(6)

    def _solve_ik(self, seed: np.ndarray, target: np.ndarray) -> np.ndarray | None:
        q = torch.as_tensor(seed, dtype=self.dtype, device=self.device).clone()
        desired = torch.as_tensor(target, dtype=self.dtype, device=self.device)
        identity = torch.eye(3, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            for _ in range(self.config.ik_iterations):
                tip, jacobian = tip_position_and_jacobian(
                    q[None], self.config.section_lengths
                )
                residual = desired - tip[0]
                if torch.linalg.vector_norm(residual) <= self.config.ik_tolerance:
                    break
                system = (
                    jacobian[0] @ jacobian[0].transpose(-1, -2)
                    + self.config.ik_damping * identity
                )
                task_step = torch.linalg.solve(system, residual[:, None])
                update = (jacobian[0].transpose(-1, -2) @ task_step).squeeze(-1)
                q = self._project_bending(q + update)
            tip, _ = tip_position_and_jacobian(q[None], self.config.section_lengths)
            error = torch.linalg.vector_norm(tip[0] - desired)
            finite = torch.isfinite(q).all() & torch.isfinite(error)
        if not bool(finite) or float(error) > self.config.ik_tolerance:
            return None
        return q.detach().cpu().numpy().astype(np.float64, copy=False)

    def _state(
        self,
        q: np.ndarray,
        obstacle_centres: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not np.isfinite(q).all():
            return None
        bending = np.linalg.norm(q.reshape(3, 2), axis=-1)
        if np.any(bending > self.config.max_bending + 1e-6):
            return None
        configurations = torch.as_tensor(q, dtype=self.dtype, device=self.device)[None]
        with torch.no_grad():
            clearance, tip = sphere_clearances_and_tips(
                configurations,
                obstacle_centres,
                self.config.section_lengths,
                self.config.sphere_radius,
                self.config.clearance_margin,
            )
        if not bool(torch.isfinite(tip).all()) or float(clearance[0]) < 0:
            return None
        return q, tip[0].detach().cpu().numpy().astype(np.float64, copy=False)

    @staticmethod
    def _steer(source: np.ndarray, target: np.ndarray, step: float) -> np.ndarray:
        displacement = target - source
        distance = float(np.linalg.norm(displacement))
        if distance <= step:
            return target.copy()
        return source + displacement * (step / distance)

    def _native_values(self, nodes: Sequence[_Node]) -> np.ndarray:
        return np.stack([node.tip if self._workspace else node.q for node in nodes])

    def _near_indices(self, nodes: Sequence[_Node], node: _Node) -> np.ndarray:
        values = self._native_values(nodes)
        value = node.tip if self._workspace else node.q
        return np.flatnonzero(
            np.linalg.norm(values - value, axis=1) <= self.config.rewire_radius
        )

    def _choose_parent(
        self,
        nodes: Sequence[_Node],
        candidate: _Node,
        nearest: int,
    ) -> int:
        if not self._star:
            return nearest
        best_parent = nearest
        best_cost = nodes[nearest].cost + float(
            np.linalg.norm(nodes[nearest].tip - candidate.tip)
        )
        for index in self._near_indices(nodes, candidate):
            configuration_step = np.linalg.norm(nodes[index].q - candidate.q)
            if configuration_step > self.config.max_adjacent_step:
                continue
            cost = nodes[index].cost + float(
                np.linalg.norm(nodes[index].tip - candidate.tip)
            )
            if cost < best_cost:
                best_parent = int(index)
                best_cost = cost
        candidate.cost = best_cost
        return best_parent

    @staticmethod
    def _ancestors(nodes: Sequence[_Node], index: int) -> set[int]:
        result: set[int] = set()
        while index >= 0:
            result.add(index)
            index = nodes[index].parent
        return result

    @staticmethod
    def _shift_descendant_costs(nodes: Sequence[_Node], root: int, change: float) -> None:
        pending = list(nodes[root].children)
        while pending:
            index = pending.pop()
            nodes[index].cost += change
            pending.extend(nodes[index].children)

    def _rewire(self, nodes: list[_Node], new_index: int) -> None:
        if not self._star:
            return
        node = nodes[new_index]
        ancestors = self._ancestors(nodes, new_index)
        for index in self._near_indices(nodes[:-1], node):
            index = int(index)
            if index in ancestors:
                continue
            neighbour = nodes[index]
            if np.linalg.norm(neighbour.q - node.q) > self.config.max_adjacent_step:
                continue
            cost = node.cost + float(np.linalg.norm(neighbour.tip - node.tip))
            if cost >= neighbour.cost:
                continue
            old_cost = neighbour.cost
            nodes[neighbour.parent].children.remove(index)
            neighbour.parent = new_index
            neighbour.cost = cost
            node.children.add(index)
            self._shift_descendant_costs(nodes, index, cost - old_cost)

    @staticmethod
    def _chain(nodes: Sequence[_Node], index: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
        configurations: list[np.ndarray] = []
        tips: list[np.ndarray] = []
        while index >= 0:
            configurations.append(nodes[index].q)
            tips.append(nodes[index].tip)
            index = nodes[index].parent
        configurations.reverse()
        tips.reverse()
        return configurations, tips

    def _resample(
        self,
        configurations: Sequence[np.ndarray],
        tips: Sequence[np.ndarray],
        obstacle_centres: torch.Tensor,
        target_tip: np.ndarray,
    ) -> np.ndarray | None:
        q = np.stack(configurations)
        p = np.stack(tips)
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))))
        keep = np.concatenate(([True], np.diff(arc) > 1e-10))
        if not keep[-1]:
            last_distinct = np.flatnonzero(keep)[-1]
            keep[last_distinct] = False
            keep[-1] = True
        q = q[keep]
        arc = arc[keep]
        if len(q) < 2 or arc[-1] <= 0:
            return None
        samples = np.linspace(0, arc[-1], self.horizon)
        path = np.stack([np.interp(samples, arc, q[:, dim]) for dim in range(6)], axis=1)
        path[0] = configurations[0]
        path[-1] = configurations[-1]
        if np.max(np.linalg.norm(np.diff(path, axis=0), axis=1)) > self.config.max_adjacent_step:
            return None
        if not np.isfinite(path).all():
            return None
        bending = np.linalg.norm(path.reshape(-1, 3, 2), axis=-1)
        if np.any(bending > self.config.max_bending + 1e-6):
            return None
        configurations_tensor = torch.as_tensor(
            path, dtype=self.dtype, device=self.device
        )
        with torch.no_grad():
            clearances, tips_tensor = sphere_clearances_and_tips(
                configurations_tensor,
                obstacle_centres,
                self.config.section_lengths,
                self.config.sphere_radius,
                self.config.clearance_margin,
            )
        if not bool(torch.isfinite(tips_tensor).all()) or bool((clearances < 0).any()):
            return None
        terminal_tip = tips_tensor[-1].detach().cpu().numpy()
        if np.linalg.norm(terminal_tip - target_tip) > self.config.goal_tolerance:
            return None
        return path

    def plan(
        self,
        start: np.ndarray,
        target_tip: np.ndarray,
        obstacle_centres: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        start = np.asarray(start, dtype=np.float64)
        target_tip = np.asarray(target_tip, dtype=np.float64)
        centres = torch.as_tensor(
            obstacle_centres, dtype=self.dtype, device=self.device
        )
        start_state = self._state(start, centres)
        if start_state is None:
            return None
        start_q, start_tip = start_state
        terminal_q = None
        terminal_tip = target_tip
        if not self._workspace:
            terminal_q = self._solve_ik(self._sample_configuration(rng), target_tip)
            if terminal_q is None:
                return None
            terminal_state = self._state(terminal_q, centres)
            if terminal_state is None:
                return None
            terminal_q, terminal_tip = terminal_state

        nodes = [_Node(start_q.copy(), start_tip.copy(), -1, 0.0, set())]
        goal_candidates: list[tuple[int, np.ndarray, np.ndarray]] = []
        for _ in range(self.config.max_iterations):
            if self._workspace:
                sample = (
                    target_tip
                    if rng.random() < self.config.goal_bias
                    else self._sample_workspace(rng)
                )
            else:
                sample = (
                    terminal_q
                    if rng.random() < self.config.goal_bias
                    else self._sample_configuration(rng)
                )
            values = self._native_values(nodes)
            nearest = int(np.argmin(np.linalg.norm(values - sample, axis=1)))
            parent = nodes[nearest]
            if self._workspace:
                desired_tip = self._steer(parent.tip, sample, self.config.step_size)
                q_new = self._solve_ik(parent.q, desired_tip)
                if q_new is None:
                    continue
            else:
                q_new = self._steer(parent.q, sample, self.config.step_size)
            state = self._state(q_new, centres)
            if state is None:
                continue
            q_new, tip_new = state
            if np.linalg.norm(q_new - parent.q) > self.config.max_adjacent_step:
                continue
            candidate = _Node(q_new.copy(), tip_new.copy(), nearest, 0.0, set())
            chosen_parent = self._choose_parent(nodes, candidate, nearest)
            candidate.parent = chosen_parent
            if not self._star:
                candidate.cost = nodes[chosen_parent].cost + float(
                    np.linalg.norm(nodes[chosen_parent].tip - candidate.tip)
                )
            new_index = len(nodes)
            nodes.append(candidate)
            nodes[chosen_parent].children.add(new_index)
            self._rewire(nodes, new_index)

            goal_q: np.ndarray | None = None
            goal_tip: np.ndarray | None = None
            if self._workspace:
                goal_distance = np.linalg.norm(candidate.tip - target_tip)
                if goal_distance <= self.config.step_size:
                    exact_q = self._solve_ik(candidate.q, target_tip)
                    if exact_q is not None:
                        exact_state = self._state(exact_q, centres)
                        if (
                            exact_state is not None
                            and np.linalg.norm(exact_q - candidate.q)
                            <= self.config.max_adjacent_step
                        ):
                            goal_q, goal_tip = exact_state
            elif (
                np.linalg.norm(candidate.q - terminal_q) <= self.config.step_size
                and np.linalg.norm(candidate.q - terminal_q)
                <= self.config.max_adjacent_step
            ):
                goal_q, goal_tip = terminal_q, terminal_tip
            if goal_q is None or goal_tip is None:
                continue
            goal_candidates.append((new_index, goal_q.copy(), goal_tip.copy()))
            if not self._star:
                configurations, tips = self._chain(nodes, new_index)
                if np.linalg.norm(configurations[-1] - goal_q) > 1e-10:
                    configurations.append(goal_q)
                    tips.append(goal_tip)
                path = self._resample(
                    configurations, tips, centres, target_tip
                )
                if path is not None:
                    return path

        if not goal_candidates:
            return None
        ordered_goals = sorted(
            goal_candidates,
            key=lambda value: nodes[value[0]].cost
            + float(np.linalg.norm(nodes[value[0]].tip - value[2])),
        )
        for parent_index, goal_q, goal_tip in ordered_goals:
            configurations, tips = self._chain(nodes, parent_index)
            if np.linalg.norm(configurations[-1] - goal_q) > 1e-10:
                configurations.append(goal_q)
                tips.append(goal_tip)
            path = self._resample(configurations, tips, centres, target_tip)
            if path is not None:
                return path
        return None


__all__ = ["RRT_ALGORITHMS", "RrtConfig", "RrtPlanner"]
