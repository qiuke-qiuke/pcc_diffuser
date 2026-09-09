"""Paired evaluation of three PCC diffusion prediction methods."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch_geometric.data import Batch

from .data import PccDataset, obstacle_graph
from .kinematics import sphere_clearances_and_tips, tip_position
from .training import reconstruct_diffusion


EVALUATION_FORMAT_VERSION = 15
METHODS = ("Diffusion", "With Post Correction", "With Guided Prediction")

# Convenient experiment switch. The uncorrected Diffusion arm is never changed.
INCORPORATE_ANALYTICAL_OBSTACLE_REPULSION = True
REPULSION_CLEARANCE_FACTOR = 1.5


def evaluation_noise_seed(base_seed: int, target_index: int, start_index: int) -> int:
    """Return a stable per-condition seed independent of evaluation batching."""
    modulus = 2**63 - 1
    return int(
        (int(base_seed) + 1_000_003 * int(target_index) + 97_409 * int(start_index))
        % modulus
    )


def evaluation_initial_noise(
    samples: int,
    horizon: int,
    base_seed: int,
    target_index: int,
    start_index: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Generate device-independent evaluation noise and move it to ``device``."""
    generator = torch.Generator(device="cpu").manual_seed(
        evaluation_noise_seed(base_seed, target_index, start_index)
    )
    noise = torch.randn(samples, horizon, 6, dtype=dtype, generator=generator)
    return noise.to(device)


@dataclass(frozen=True)
class PccEvaluationConfig:
    """Settings for a paired three-method evaluation on held-out cases."""

    output_dir: str | Path
    dataset: str | Path
    checkpoint: str | Path
    split: str
    samples_per_case: int
    case_batch_size: int
    ddim_steps: int
    eta: float
    seed: int
    device: str
    tip_success_tolerance: float
    use_ema_weights: bool
    post_correction_step_size: float
    post_correction_repulsion_step_size: float
    post_correction_fraction: float
    guidance_weight: float
    guidance_repulsion_weight: float
    guidance_fraction: float
    overwrite: bool

    def validate(self) -> None:
        if self.samples_per_case < 1 or self.case_batch_size < 1:
            raise ValueError("sample and case batch sizes must be positive")
        if self.ddim_steps < 1:
            raise ValueError("ddim_steps must be positive")
        if self.eta < 0:
            raise ValueError("eta must be non-negative")
        if self.tip_success_tolerance <= 0:
            raise ValueError("tip_success_tolerance must be positive")
        if self.post_correction_step_size < 0:
            raise ValueError("post_correction_step_size must be non-negative")
        if self.post_correction_repulsion_step_size < 0:
            raise ValueError("post_correction_repulsion_step_size must be non-negative")
        if not 0 <= self.post_correction_fraction <= 1:
            raise ValueError("post_correction_fraction must lie in [0,1]")
        if self.guidance_weight < 0:
            raise ValueError("guidance_weight must be non-negative")
        if self.guidance_repulsion_weight < 0:
            raise ValueError("guidance_repulsion_weight must be non-negative")
        if not 0 <= self.guidance_fraction <= 1:
            raise ValueError("guidance_fraction must lie in [0,1]")


@dataclass(frozen=True)
class EvaluationCases:
    """Unique physical planning conditions selected from one dataset split."""

    dataset: PccDataset
    metadata: Mapping[str, Any]
    record_indices: np.ndarray
    starts: np.ndarray
    target_tips: np.ndarray

    @classmethod
    def load(
        cls,
        dataset_path: str | Path,
        split: str,
        normaliser: Any | None = None,
    ) -> "EvaluationCases":
        root = Path(dataset_path).expanduser().resolve()
        metadata_path = root / "dataset_meta.json"
        archive_path = root / "dataset.npz"
        missing = [
            path.name for path in (metadata_path, archive_path) if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"dataset directory is missing {', '.join(missing)}: {root}"
            )
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        dataset = PccDataset.load(root, split=split, normaliser=normaliser)
        first_by_condition: dict[tuple[int, int], int] = {}
        for record_index, condition in enumerate(
            zip(dataset.target_indices.tolist(), dataset.start_indices.tolist())
        ):
            first_by_condition.setdefault(
                (int(condition[0]), int(condition[1])), record_index
            )
        records = np.asarray(list(first_by_condition.values()), dtype=np.int64)
        if not len(records):
            raise RuntimeError(f"no evaluation cases found in split {split!r}")
        return cls(
            dataset=dataset,
            metadata=metadata,
            record_indices=records,
            starts=np.ascontiguousarray(dataset.paths[records, 0]),
            target_tips=np.ascontiguousarray(dataset.target_tips[records]),
        )

    def obstacles(self, case_index: int) -> np.ndarray:
        return self.dataset.obstacles(int(self.record_indices[case_index]))

    @property
    def target_indices(self) -> np.ndarray:
        return self.dataset.target_indices[self.record_indices]

    @property
    def start_indices(self) -> np.ndarray:
        return self.dataset.start_indices[self.record_indices]

    def __len__(self) -> int:
        return len(self.record_indices)


class PathPlanner:
    """Common physical-unit interface for one stochastic planning attempt."""

    @property
    def name(self) -> str:
        """Return the stable method name recorded in evaluation output."""
        raise NotImplementedError

    @property
    def horizon(self) -> int:
        """Return the number of configurations in each successful path."""
        raise NotImplementedError

    def plan(
        self,
        start: np.ndarray,
        target_tip: np.ndarray,
        obstacle_centres: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        """Return one physical path ``[H,6]``, or ``None`` on failure."""
        raise NotImplementedError


@dataclass(frozen=True)
class PathEvaluator:
    """Evaluate physical paths with one shared kinematic and collision policy."""

    section_lengths: tuple[float, float, float]
    sphere_radius: float
    clearance_margin: float

    @staticmethod
    def distribution_statistics(values: np.ndarray) -> dict[str, float] | None:
        """Summarise finite metric values, or return ``None`` when absent."""
        finite = np.asarray(values)[np.isfinite(values)]
        if not len(finite):
            return None
        return _distribution_statistics(finite)

    def evaluate_batch(
        self,
        paths: torch.Tensor,
        starts: torch.Tensor,
        target_tips: torch.Tensor,
        obstacle_centres: list[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if paths.ndim != 4 or paths.shape[-1] != 6:
            raise ValueError("paths must have shape [C,S,H,6]")
        case_count, _, horizon, _ = paths.shape
        if starts.shape != (case_count, 6) or target_tips.shape != (case_count, 3):
            raise ValueError("case condition shapes do not match paths")
        if len(obstacle_centres) != case_count:
            raise ValueError("one obstacle array is required per case")
        predicted_tips = tip_position(paths, self.section_lengths)
        start_tips = tip_position(starts, self.section_lengths)
        fractions = torch.linspace(
            0, 1, horizon, dtype=paths.dtype, device=paths.device
        )
        desired_tips = start_tips[:, None] + fractions[:, None] * (
            target_tips - start_tips
        )[:, None]
        terminal_errors = torch.linalg.vector_norm(
            predicted_tips[:, :, -1] - target_tips[:, None], dim=-1
        )
        path_errors = torch.linalg.vector_norm(
            predicted_tips - desired_tips[:, None], dim=-1
        )
        minimum_clearances = paths.new_full(paths.shape[:2], torch.inf)
        for case_index, centres in enumerate(obstacle_centres):
            if not len(centres):
                continue
            clearances, _ = sphere_clearances_and_tips(
                paths[case_index],
                centres.to(device=paths.device, dtype=paths.dtype),
                self.section_lengths,
                self.sphere_radius,
                self.clearance_margin,
            )
            minimum_clearances[case_index] = clearances.amin(dim=-1)
        return {
            "terminal_tip_errors": terminal_errors,
            "workspace_path_mean_errors": path_errors.mean(dim=-1),
            "workspace_path_max_errors": path_errors.amax(dim=-1),
            "max_configuration_step": _maximum_configuration_step(paths),
            "minimum_clearance": minimum_clearances,
            "collision_failures": minimum_clearances < 0,
        }


@dataclass(frozen=True)
class EvaluationResult:
    """Serializable evaluation arrays and their self-describing metadata."""

    arrays: Mapping[str, np.ndarray]
    metadata: Mapping[str, Any]

    def save(
        self,
        output_dir: str | Path,
        archive_name: str,
        metadata_name: str,
    ) -> None:
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        archive = output / archive_name
        temporary_archive = output / f"{archive_name}.tmp.npz"
        np.savez_compressed(temporary_archive, **self.arrays)
        temporary_archive.replace(archive)
        metadata_path = output / metadata_name
        temporary_metadata = output / f"{metadata_name}.tmp"
        temporary_metadata.write_text(
            json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8"
        )
        temporary_metadata.replace(metadata_path)


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


def _padded_obstacle_batch(
    centres_by_case: list[torch.Tensor],
    samples_per_case: int,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Repeat ragged physical obstacle sets and return padded centres plus a mask."""
    repeated = [
        centres.to(device=reference.device, dtype=reference.dtype)
        for centres in centres_by_case
        for _ in range(samples_per_case)
    ]
    maximum = max((len(centres) for centres in repeated), default=0)
    padded = reference.new_zeros((len(repeated), maximum, 3))
    mask = torch.zeros(
        (len(repeated), maximum), dtype=torch.bool, device=reference.device
    )
    for index, centres in enumerate(repeated):
        padded[index, : len(centres)] = centres
        mask[index, : len(centres)] = True
    return padded, mask


def _distribution_statistics(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def _maximum_configuration_step(configurations: torch.Tensor) -> torch.Tensor:
    """Return the maximum adjacent configuration change along each path."""
    configuration_steps = torch.diff(configurations, dim=-2)
    step_norms = configuration_steps.norm(dim=-1)
    return step_norms.amax(dim=-1)


def evaluate_model(
    config: PccEvaluationConfig,
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Evaluate three methods using identical cases and initial DDIM noise."""
    config.validate()
    started = time.perf_counter()
    dataset_path = Path(config.dataset).expanduser().resolve()
    checkpoint_path = Path(config.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    output_dir = Path(config.output_dir).expanduser().resolve()
    evaluation_outputs = (output_dir / "eval.npz", output_dir / "eval_meta.json")
    existing_outputs = [path for path in evaluation_outputs if path.exists()]
    if existing_outputs and not config.overwrite:
        names = ", ".join(path.name for path in existing_outputs)
        raise FileExistsError(
            f"evaluation output already exists in {output_dir}: {names}; "
            "pass --overwrite to replace it"
        )
    device = _resolve_device(config.device)
    weights = "ema" if config.use_ema_weights else "model"
    diffusion, normaliser, checkpoint = reconstruct_diffusion(
        checkpoint_path, weights=weights, device=device
    )
    cases = EvaluationCases.load(dataset_path, config.split, normaliser)
    dataset = cases.dataset
    record_indices = cases.record_indices
    targets_physical = torch.from_numpy(cases.target_tips).float()
    starts_physical = torch.from_numpy(cases.starts).float()
    lengths = tuple(float(value) for value in checkpoint["section_lengths"])
    path_evaluator = PathEvaluator(
        lengths, dataset.sphere_radius, dataset.clearance_margin
    )
    terminal_errors: list[list[np.ndarray]] = [[] for _ in METHODS]
    path_mean_errors: list[list[np.ndarray]] = [[] for _ in METHODS]
    path_max_errors: list[list[np.ndarray]] = [[] for _ in METHODS]
    max_configuration_steps: list[list[np.ndarray]] = [[] for _ in METHODS]
    minimum_clearances: list[list[np.ndarray]] = [[] for _ in METHODS]
    collision_failures: list[list[np.ndarray]] = [[] for _ in METHODS]
    method_seconds = np.zeros(len(METHODS), dtype=np.float64)
    method_batch_seconds: list[list[float]] = [[] for _ in METHODS]
    batch_path_counts: list[int] = []

    for begin in range(0, len(record_indices), config.case_batch_size):
        end = min(begin + config.case_batch_size, len(record_indices))
        case_count = end - begin
        starts = starts_physical[begin:end].to(device)
        targets = targets_physical[begin:end].to(device)
        starts_normalised = normaliser.normalise_q(starts).repeat_interleave(
            config.samples_per_case, dim=0
        )
        targets_normalised = normaliser.normalise_workspace(targets).repeat_interleave(
            config.samples_per_case, dim=0
        )
        sample_count = case_count * config.samples_per_case
        batch_path_counts.append(sample_count)
        graphs = []
        physical_obstacle_centres: list[torch.Tensor] = []
        normalised_radius = normaliser.normalise_workspace_radius(
            dataset.sphere_radius
        )
        for record_index in record_indices[begin:end]:
            centres = torch.from_numpy(dataset.obstacles(int(record_index))).float()
            physical_obstacle_centres.append(centres.to(device))
            centres = normaliser.normalise_workspace(centres)
            graphs.extend(
                obstacle_graph(centres, normalised_radius)
                for _ in range(config.samples_per_case)
            )
        obstacle_batch = Batch.from_data_list(graphs).to(device)
        padded_obstacles, obstacle_mask = _padded_obstacle_batch(
            physical_obstacle_centres, config.samples_per_case, starts_normalised
        )
        repulsion_safe_radius = dataset.sphere_radius + (
            REPULSION_CLEARANCE_FACTOR * dataset.clearance_margin
        )
        condition_target_indices = dataset.target_indices[record_indices[begin:end]]
        condition_start_indices = dataset.start_indices[record_indices[begin:end]]
        initial_noise = torch.cat(
            [
                evaluation_initial_noise(
                    config.samples_per_case,
                    diffusion.horizon,
                    config.seed,
                    int(target_index),
                    int(start_index),
                    starts.dtype,
                    device,
                )
                for target_index, start_index in zip(
                    condition_target_indices, condition_start_indices
                )
            ],
            dim=0,
        )
        arm_seed = config.seed + 1_000_003 + begin

        _synchronise(device)
        arm_started = time.perf_counter()
        diffusion_paths = diffusion.sample_ddim(
            starts_normalised,
            targets_normalised,
            obstacle_batch,
            section_lengths=lengths,
            sample_steps=config.ddim_steps,
            eta=config.eta,
            generator=torch.Generator(device=device).manual_seed(arm_seed),
            initial_noise=initial_noise,
            normaliser=None,
            guidance_weight=0.0,
            guidance_fraction=config.guidance_fraction,
            post_correction=False,
            post_correction_step_size=config.post_correction_step_size,
            post_correction_fraction=config.post_correction_fraction,
            return_trace=False,
        )
        _synchronise(device)
        diffusion_elapsed = time.perf_counter() - arm_started
        method_seconds[0] += diffusion_elapsed
        method_batch_seconds[0].append(diffusion_elapsed)

        _synchronise(device)
        arm_started = time.perf_counter()
        post_corrected_paths = diffusion.sample_ddim(
            starts_normalised,
            targets_normalised,
            obstacle_batch,
            sample_steps=config.ddim_steps,
            eta=config.eta,
            generator=torch.Generator(device=device).manual_seed(arm_seed),
            initial_noise=initial_noise,
            normaliser=normaliser,
            guidance_weight=0.0,
            guidance_fraction=config.guidance_fraction,
            post_correction=True,
            post_correction_step_size=config.post_correction_step_size,
            post_correction_fraction=config.post_correction_fraction,
            section_lengths=lengths,
            return_trace=False,
            obstacle_centres=padded_obstacles,
            obstacle_mask=obstacle_mask,
            post_correction_repulsion_weight=(
                config.post_correction_repulsion_step_size
                if INCORPORATE_ANALYTICAL_OBSTACLE_REPULSION
                else 0.0
            ),
            repulsion_safe_radius=repulsion_safe_radius,
        )
        _synchronise(device)
        post_correction_elapsed = time.perf_counter() - arm_started
        method_seconds[1] += post_correction_elapsed
        method_batch_seconds[1].append(post_correction_elapsed)

        _synchronise(device)
        arm_started = time.perf_counter()
        guided_prediction_paths = diffusion.sample_ddim(
            starts_normalised,
            targets_normalised,
            obstacle_batch,
            sample_steps=config.ddim_steps,
            eta=config.eta,
            generator=torch.Generator(device=device).manual_seed(arm_seed),
            initial_noise=initial_noise,
            normaliser=normaliser,
            guidance_weight=config.guidance_weight,
            guidance_fraction=config.guidance_fraction,
            post_correction=False,
            post_correction_step_size=config.post_correction_step_size,
            post_correction_fraction=config.post_correction_fraction,
            section_lengths=lengths,
            return_trace=False,
            obstacle_centres=padded_obstacles,
            obstacle_mask=obstacle_mask,
            guidance_repulsion_weight=(
                config.guidance_repulsion_weight
                if INCORPORATE_ANALYTICAL_OBSTACLE_REPULSION
                else 0.0
            ),
            repulsion_safe_radius=repulsion_safe_radius,
        )
        _synchronise(device)
        guided_elapsed = time.perf_counter() - arm_started
        method_seconds[2] += guided_elapsed
        method_batch_seconds[2].append(guided_elapsed)

        for method_index, sampled in enumerate(
            (diffusion_paths, post_corrected_paths, guided_prediction_paths)
        ):
            with torch.no_grad():
                physical = normaliser.denormalise_q(sampled).reshape(
                    case_count, config.samples_per_case, diffusion.horizon, 6
                )
                metrics = path_evaluator.evaluate_batch(
                    physical, starts, targets, physical_obstacle_centres
                )
            terminal_errors[method_index].append(
                metrics["terminal_tip_errors"].cpu().numpy()
            )
            path_mean_errors[method_index].append(
                metrics["workspace_path_mean_errors"].cpu().numpy()
            )
            path_max_errors[method_index].append(
                metrics["workspace_path_max_errors"].cpu().numpy()
            )
            max_configuration_steps[method_index].append(
                metrics["max_configuration_step"].cpu().numpy()
            )
            minimum_clearances[method_index].append(
                metrics["minimum_clearance"].cpu().numpy()
            )
            collision_failures[method_index].append(
                metrics["collision_failures"].cpu().numpy()
            )

        if progress is not None:
            elapsed = time.perf_counter() - started
            progress(
                f"[{elapsed:9.1f} s] cases {end}/{len(record_indices)} evaluated | "
                f"{100 * end / len(record_indices):.1f}%"
            )

    error_array = np.stack([np.concatenate(values, axis=0) for values in terminal_errors])
    path_mean_array = np.stack([np.concatenate(values, axis=0) for values in path_mean_errors])
    path_max_array = np.stack([np.concatenate(values, axis=0) for values in path_max_errors])
    max_step_array = np.stack(
        [np.concatenate(values, axis=0) for values in max_configuration_steps]
    )
    minimum_clearance_array = np.stack(
        [np.concatenate(values, axis=0) for values in minimum_clearances]
    )
    collision_failure_array = np.stack(
        [np.concatenate(values, axis=0) for values in collision_failures]
    )
    method_batch_seconds_array = np.asarray(method_batch_seconds, dtype=np.float64)
    batch_path_counts_array = np.asarray(batch_path_counts, dtype=np.int64)
    total_generated_paths = len(record_indices) * config.samples_per_case
    seconds_per_path = method_seconds / total_generated_paths
    seconds_per_case = method_seconds / len(record_indices)

    total_elapsed_seconds = time.perf_counter() - started
    cases_with_obstacles = sum(
        len(dataset.obstacles(int(record_index))) > 0
        for record_index in record_indices
    )
    paths_with_obstacles = cases_with_obstacles * config.samples_per_case
    method_statistics = {}
    for index, method in enumerate(METHODS):
        tip_failures = error_array[index] > config.tip_success_tolerance
        method_collision_failures = collision_failure_array[index]
        both_failures = tip_failures & method_collision_failures
        successes = ~(tip_failures | method_collision_failures)
        method_statistics[method] = {
            "total_paths": int(total_generated_paths),
            "paths_with_obstacles": int(paths_with_obstacles),
            "paths_failed_collision": int(method_collision_failures.sum()),
            "paths_failed_tip_error": int(tip_failures.sum()),
            "paths_failed_both": int(both_failures.sum()),
            "success_rate": float(successes.mean()),
            "terminal_tip_error": _distribution_statistics(error_array[index]),
            "workspace_path_error": _distribution_statistics(path_mean_array[index]),
            "max_configuration_step": _distribution_statistics(max_step_array[index]),
            "elapsed_seconds": float(method_seconds[index]),
            "seconds_per_case": float(seconds_per_case[index]),
            "seconds_per_path": float(seconds_per_path[index]),
        }
    metadata: dict[str, object] = {
        "format": "pcc_diffuser_evaluation",
        "version": EVALUATION_FORMAT_VERSION,
        "checkpoint": str(checkpoint_path),
        "weights": weights,
        "dataset": str(dataset_path),
        "split": config.split,
        "seed": config.seed,
        "device": str(device),
        "cases": int(len(record_indices)),
        "samples_per_case": config.samples_per_case,
        "paired_initial_noise": True,
        "noise_policy": (
            "device-independent CPU Gaussian seed per target-start condition"
        ),
        "methods": list(METHODS),
        "analytical_correction": {
            "obstacle_repulsion_enabled": INCORPORATE_ANALYTICAL_OBSTACLE_REPULSION,
            "repulsion_method": "deepest_sampled_backbone_point_jacobian_transpose",
            "repulsion_clearance_factor": REPULSION_CLEARANCE_FACTOR,
            "frame_zero_inpainted": True,
            "configuration_update_clamped": False,
            "section_bending_projected": False,
        },
        "metric_definitions": {
            "success": (
                "terminal tip error <= tip_success_tolerance and no sampled-backbone "
                "collision at any path frame"
            ),
            "collision": (
                "minimum sampled-backbone clearance is negative; clearance subtracts "
                "sphere_radius and clearance_margin"
            ),
            "failure_counts": (
                "paths_failed_collision and paths_failed_tip_error are inclusive; "
                "paths_failed_both is their overlap"
            ),
            "workspace_path_error": (
                "mean framewise distance from the straight start-tip-to-target line"
            ),
            "max_configuration_step": (
                "max_h ||q[h+1]-q[h]||; lower is smoother"
            ),
            "runtime": (
                "synchronised DDIM wall time only; excludes metric computation and I/O"
            ),
            "seconds_per_case": (
                "runtime divided by test cases; each case generates samples_per_case paths"
            ),
            "seconds_per_path": (
                "amortised batched runtime divided by all generated paths"
            ),
        },
        "method_statistics": method_statistics,
        "total_elapsed_seconds": float(total_elapsed_seconds),
        "config": asdict(config) | {
            "output_dir": str(Path(config.output_dir)),
            "dataset": str(Path(config.dataset)),
            "checkpoint": str(Path(config.checkpoint)),
        },
    }
    arrays = {
        "methods": np.asarray(METHODS),
        "target_indices": dataset.target_indices[record_indices],
        "start_indices": dataset.start_indices[record_indices],
        "target_tips": targets_physical.numpy(),
        "terminal_tip_errors": error_array,
        "workspace_path_mean_errors": path_mean_array,
        "workspace_path_max_errors": path_max_array,
        "max_configuration_step": max_step_array,
        "minimum_clearance": minimum_clearance_array,
        "collision_failures": collision_failure_array,
        "method_elapsed_seconds": method_seconds,
        "method_batch_elapsed_seconds": method_batch_seconds_array,
        "batch_path_counts": batch_path_counts_array,
        "method_seconds_per_case": seconds_per_case,
        "method_seconds_per_path": seconds_per_path,
    }
    EvaluationResult(arrays, metadata).save(
        output_dir, "eval.npz", "eval_meta.json"
    )
    return metadata


__all__ = [
    "EVALUATION_FORMAT_VERSION",
    "EvaluationCases",
    "EvaluationResult",
    "METHODS",
    "INCORPORATE_ANALYTICAL_OBSTACLE_REPULSION",
    "PathEvaluator",
    "PathPlanner",
    "PccEvaluationConfig",
    "evaluate_model",
    "evaluation_initial_noise",
    "evaluation_noise_seed",
]
