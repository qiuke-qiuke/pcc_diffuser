"""Evaluation runner for independent PCC planning baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
import time
from typing import Callable, Mapping

import numpy as np
import torch

from pcc_diffuser.evaluation import EvaluationCases, EvaluationResult, PathEvaluator, evaluation_noise_seed
from pcc_diffuser.generation import PccGenerationConfig
from .repulsion import RepulsionPlanner
from .rrt import RRT_ALGORITHMS, RrtConfig, RrtPlanner

BENCHMARK_FORMAT_VERSION = 2
ALGORITHMS = (*RRT_ALGORITHMS, "repulsion")
ARCHIVE_NAME = "benchmark.npz"
METADATA_NAME = "benchmark_meta.json"
METRIC_NAMES = (
    "terminal_tip_errors", "workspace_path_mean_errors",
    "workspace_path_max_errors", "max_configuration_step",
    "minimum_clearance", "collision_failures",
)


@dataclass(frozen=True)
class BenchmarkEvaluationConfig:
    output_dir: str | Path
    dataset: str | Path
    algorithms: tuple[str, ...]
    device: str = "auto"
    split: str = "test"
    seed: int = 0
    overwrite: bool = False
    tip_success_tolerance: float = 0.1
    samples_per_case: int = 10
    max_iterations: int = 500
    goal_bias: float = 0.1
    step_size: float = 0.2
    rewire_radius: float = 0.3

    def validate(self) -> None:
        if not self.algorithms:
            raise ValueError("algorithms must not be empty")
        if len(set(self.algorithms)) != len(self.algorithms):
            raise ValueError("algorithms must not contain duplicates")
        if any(name not in ALGORITHMS for name in self.algorithms):
            raise ValueError(f"algorithms must be selected from {ALGORITHMS}")
        if self.samples_per_case < 1:
            raise ValueError("samples_per_case must be positive")
        if self.tip_success_tolerance <= 0:
            raise ValueError("tip_success_tolerance must be positive")


@dataclass(frozen=True)
class _MethodResult:
    planned: np.ndarray
    elapsed: np.ndarray
    metrics: Mapping[str, np.ndarray]
    planner_config: Mapping[str, object]
    evaluation_config: Mapping[str, object]


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _generation_metadata(cases: EvaluationCases) -> dict[str, object]:
    generation = cases.metadata.get("config")
    if not isinstance(generation, dict):
        raise ValueError("dataset metadata has no generation config")
    return generation


def _rrt_config(config, cases, device, algorithm: str) -> RrtConfig:
    generation = _generation_metadata(cases)
    required = (
        "max_bending", "radial_min", "radial_max", "z_min", "z_max",
        "ik_iterations", "ik_damping", "ik_tolerance",
    )
    missing = [name for name in required if name not in generation]
    if missing:
        raise ValueError(f"dataset generation config is missing: {', '.join(missing)}")
    maximum_step = cases.metadata.get("maximum_adjacent_configuration_step")
    if not isinstance(maximum_step, (int, float)):
        raise ValueError("dataset metadata has no maximum adjacent configuration step")
    return RrtConfig(
        algorithm=algorithm, horizon=cases.dataset.horizon,
        section_lengths=cases.dataset.section_lengths,
        max_bending=float(generation["max_bending"]),
        max_adjacent_step=float(maximum_step),
        radial_min=float(generation["radial_min"]),
        radial_max=float(generation["radial_max"]),
        z_min=float(generation["z_min"]), z_max=float(generation["z_max"]),
        sphere_radius=cases.dataset.sphere_radius,
        clearance_margin=cases.dataset.clearance_margin,
        ik_iterations=int(generation["ik_iterations"]),
        ik_damping=float(generation["ik_damping"]),
        ik_tolerance=float(generation["ik_tolerance"]),
        max_iterations=config.max_iterations, goal_bias=config.goal_bias,
        goal_tolerance=config.tip_success_tolerance, step_size=config.step_size,
        rewire_radius=config.rewire_radius, device=str(device),
    ).validated()


def _repulsion_config(cases, device) -> PccGenerationConfig:
    values = dict(_generation_metadata(cases))
    values.update(
        output_dir=".", section_lengths=cases.dataset.section_lengths,
        horizon=cases.dataset.horizon, sphere_radius=cases.dataset.sphere_radius,
        clearance_margin=cases.dataset.clearance_margin,
        device=str(device), overwrite=False,
    )
    names = {field.name for field in fields(PccGenerationConfig)}
    missing = sorted(names - values.keys())
    if missing:
        raise ValueError(f"dataset generation config is missing: {', '.join(missing)}")
    return PccGenerationConfig(**{name: values[name] for name in names}).validated()


def _planner(algorithm, config, cases, device):
    if algorithm == "repulsion":
        planner = RepulsionPlanner(_repulsion_config(cases, device))
        return planner, planner.config_dict()
    planner_config = _rrt_config(config, cases, device, algorithm)
    return RrtPlanner(planner_config), planner_config.as_dict()


def _evaluation_config(config, device) -> dict[str, object]:
    values = asdict(config)
    values.pop("algorithms")
    values.pop("overwrite")
    values["output_dir"] = str(Path(config.output_dir).expanduser().resolve())
    values["dataset"] = str(Path(config.dataset).expanduser().resolve())
    values["device"] = str(device)
    return values


def _load_existing(output_dir, config, cases, device) -> dict[str, _MethodResult]:
    archive_path = output_dir / ARCHIVE_NAME
    metadata_path = output_dir / METADATA_NAME
    if not archive_path.exists() and not metadata_path.exists():
        return {}
    if not archive_path.is_file() or not metadata_path.is_file():
        raise RuntimeError("benchmark archive and metadata must both exist")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "format": "pcc_diffuser_benchmark", "version": BENCHMARK_FORMAT_VERSION,
        "dataset": str(Path(config.dataset).expanduser().resolve()),
        "split": config.split, "seed": config.seed, "device": str(device),
        "cases": len(cases), "samples_per_case": config.samples_per_case,
        "tip_success_tolerance": config.tip_success_tolerance,
    }
    mismatches = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatches:
        raise ValueError("existing benchmark output is incompatible in: " + ", ".join(mismatches))
    with np.load(archive_path, allow_pickle=False) as archive:
        methods = [str(value) for value in archive["methods"].tolist()]
        for name, expected_values in (
            ("target_indices", cases.target_indices),
            ("start_indices", cases.start_indices),
            ("target_tips", cases.target_tips),
        ):
            if not np.array_equal(archive[name], expected_values):
                raise ValueError(f"existing benchmark output has different {name}")
        return {
            method: _MethodResult(
                archive["planning_success"][index].copy(),
                archive["attempt_elapsed_seconds"][index].copy(),
                {name: archive[name][index].copy() for name in METRIC_NAMES},
                metadata["planner_configs"][method],
                metadata["evaluation_configs"][method],
            )
            for index, method in enumerate(methods)
        }


def _evaluate_method(algorithm, config, cases, evaluator, device, progress):
    planner, planner_config = _planner(algorithm, config, cases, device)
    shape = (len(cases), config.samples_per_case)
    planned = np.zeros(shape, dtype=bool)
    elapsed = np.zeros(shape, dtype=np.float64)
    metrics = {
        name: np.zeros(shape, dtype=bool) if name == "collision_failures"
        else np.full(shape, np.nan, dtype=np.float32)
        for name in METRIC_NAMES
    }
    started = time.perf_counter()
    for case_index in range(len(cases)):
        start, target = cases.starts[case_index], cases.target_tips[case_index]
        obstacles = cases.obstacles(case_index)
        base_seed = evaluation_noise_seed(
            config.seed, int(cases.target_indices[case_index]),
            int(cases.start_indices[case_index]),
        )
        for sample_index in range(config.samples_per_case):
            rng = np.random.default_rng((base_seed + 65_537 * sample_index) % 2**63)
            _synchronise(device)
            attempt_started = time.perf_counter()
            path = planner.plan(start, target, obstacles, rng)
            _synchronise(device)
            elapsed[case_index, sample_index] = time.perf_counter() - attempt_started
            if path is None:
                continue
            planned[case_index, sample_index] = True
            physical = torch.as_tensor(path, dtype=torch.float32, device=device)
            with torch.no_grad():
                values = evaluator.evaluate_batch(
                    physical[None, None],
                    torch.as_tensor(start, dtype=torch.float32, device=device)[None],
                    torch.as_tensor(target, dtype=torch.float32, device=device)[None],
                    [torch.as_tensor(obstacles, dtype=torch.float32, device=device)],
                )
            for name, value in values.items():
                metrics[name][case_index, sample_index] = value[0, 0].cpu().numpy()
        if progress is not None:
            seconds = time.perf_counter() - started
            progress(
                f"[{seconds:9.1f} s] {algorithm} | "
                f"cases {case_index + 1}/{len(cases)} evaluated | "
                f"{100 * (case_index + 1) / len(cases):.1f}%"
            )
    return _MethodResult(
        planned, elapsed, metrics, planner_config, _evaluation_config(config, device)
    )


def _statistics(result, evaluator, obstacle_cases, tolerance):
    tip_failures = result.planned & (result.metrics["terminal_tip_errors"] > tolerance)
    collision_failures = result.planned & result.metrics["collision_failures"]
    successes = result.planned & ~tip_failures & ~collision_failures
    total_paths = result.planned.size
    elapsed = float(result.elapsed.sum())
    return {
        "total_paths": total_paths,
        "paths_with_obstacles": int(obstacle_cases.sum() * result.planned.shape[1]),
        "paths_failed_planning": int((~result.planned).sum()),
        "paths_failed_collision": int(collision_failures.sum()),
        "paths_failed_tip_error": int(tip_failures.sum()),
        "paths_failed_both": int((tip_failures & collision_failures).sum()),
        "success_rate": float(successes.mean()),
        "terminal_tip_error": evaluator.distribution_statistics(result.metrics["terminal_tip_errors"]),
        "workspace_path_error": evaluator.distribution_statistics(result.metrics["workspace_path_mean_errors"]),
        "max_configuration_step": evaluator.distribution_statistics(result.metrics["max_configuration_step"]),
        "elapsed_seconds": elapsed,
        "seconds_per_case": elapsed / result.planned.shape[0],
        "seconds_per_path": elapsed / total_paths,
    }


def _save(output_dir, results, config, cases, evaluator, device):
    methods = [name for name in ALGORITHMS if name in results]
    obstacle_cases = np.asarray([len(cases.obstacles(i)) > 0 for i in range(len(cases))])
    elapsed = np.stack([results[name].elapsed for name in methods])
    method_seconds = elapsed.sum(axis=(1, 2))
    path_count = len(cases) * config.samples_per_case
    arrays = {
        "methods": np.asarray(methods), "target_indices": cases.target_indices,
        "start_indices": cases.start_indices, "target_tips": cases.target_tips,
        "planning_success": np.stack([results[name].planned for name in methods]),
        **{metric: np.stack([results[name].metrics[metric] for name in methods]) for metric in METRIC_NAMES},
        "attempt_elapsed_seconds": elapsed, "method_elapsed_seconds": method_seconds,
        "method_seconds_per_case": method_seconds / len(cases),
        "method_seconds_per_path": method_seconds / path_count,
    }
    metadata = {
        "format": "pcc_diffuser_benchmark", "version": BENCHMARK_FORMAT_VERSION,
        "methods": methods, "dataset": str(Path(config.dataset).expanduser().resolve()),
        "split": config.split, "seed": config.seed, "device": str(device),
        "cases": len(cases), "samples_per_case": config.samples_per_case,
        "tip_success_tolerance": config.tip_success_tolerance,
        "metric_definitions": {
            "success": "planner returned a path, terminal tip error <= tip_success_tolerance, and no sampled-backbone collision",
            "runtime": "synchronised wall time including terminal IK, planning, resampling, and planner-internal validity checks; excludes shared metric computation and output I/O",
            "cost": "accumulated Euclidean tip-position distance for RRT methods",
            "collision": "node-only checking for RRT; full-path checking for repulsion",
        },
        "method_statistics": {
            name: _statistics(results[name], evaluator, obstacle_cases, config.tip_success_tolerance)
            for name in methods
        },
        "total_elapsed_seconds": float(method_seconds.sum()),
        "evaluation_configs": {name: dict(results[name].evaluation_config) for name in methods},
        "planner_configs": {name: dict(results[name].planner_config) for name in methods},
    }
    EvaluationResult(arrays, metadata).save(output_dir, ARCHIVE_NAME, METADATA_NAME)
    return metadata


def evaluate_benchmarks(config, progress: Callable[[str], None] | None = None):
    """Evaluate methods and atomically checkpoint the combined archive per method."""
    config.validate()
    dataset_path = Path(config.dataset).expanduser().resolve()
    output_dir = Path(config.output_dir).expanduser().resolve() / f"{dataset_path.name}_benchmarks"
    device = _resolve_device(config.device)
    cases = EvaluationCases.load(dataset_path, config.split)
    evaluator = PathEvaluator(
        cases.dataset.section_lengths, cases.dataset.sphere_radius,
        cases.dataset.clearance_margin,
    )
    results = _load_existing(output_dir, config, cases, device)
    metadata = None
    for algorithm in config.algorithms:
        if algorithm in results and not config.overwrite:
            if progress is not None:
                progress(f"[skipped] {algorithm} | result already present")
            continue
        results[algorithm] = _evaluate_method(
            algorithm, config, cases, evaluator, device, progress
        )
        metadata = _save(output_dir, results, config, cases, evaluator, device)
    return metadata or _save(output_dir, results, config, cases, evaluator, device)


__all__ = ["ALGORITHMS", "BENCHMARK_FORMAT_VERSION", "BenchmarkEvaluationConfig", "evaluate_benchmarks"]
