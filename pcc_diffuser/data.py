"""Dense path arrays and ragged PyG obstacle graphs for PCC diffusion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from .normalisation import PccNormaliser


DATASET_FORMAT_VERSION = 25


def directed_edge_index(node_count: int) -> torch.Tensor:
    """Return every directed edge ``source -> target`` except self-edges."""
    if node_count < 0:
        raise ValueError("node_count must be non-negative")
    if node_count < 2:
        return torch.empty((2, 0), dtype=torch.long)
    nodes = torch.arange(node_count, dtype=torch.long)
    source = nodes.repeat_interleave(node_count)
    target = nodes.repeat(node_count)
    keep = source != target
    return torch.stack((source[keep], target[keep]))


def obstacle_graph(centres: torch.Tensor, radii: torch.Tensor | float) -> Data:
    """Construct a fully directed sphere graph with ``[centre, radius]`` nodes."""
    centres = torch.as_tensor(centres, dtype=torch.float32)
    if centres.ndim != 2 or centres.shape[1] != 3:
        raise ValueError("obstacle centres must have shape [K,3]")
    if not torch.isfinite(centres).all():
        raise ValueError("obstacle centres must be finite")
    radii = torch.as_tensor(radii, dtype=centres.dtype, device=centres.device)
    if radii.ndim == 0:
        if not bool(torch.isfinite(radii)) or not bool(radii > 0):
            raise ValueError("obstacle radius must be finite and positive")
        radii = radii.expand(len(centres))
    elif radii.shape != (len(centres),):
        raise ValueError("obstacle radii must be scalar or have shape [K]")
    elif not bool(torch.isfinite(radii).all()) or bool((radii <= 0).any()):
        raise ValueError("obstacle radii must be finite and positive")
    return Data(
        x=torch.cat((centres, radii.unsqueeze(-1)), dim=-1),
        edge_index=directed_edge_index(len(centres)),
        num_nodes=len(centres),
    )


def collate_pcc_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Stack path tensors and batch their variable-size PyG graphs."""
    if not items:
        raise ValueError("cannot collate an empty batch")
    return {
        "path": torch.stack([item["path"] for item in items]),
        "start": torch.stack([item["start"] for item in items]),
        "target_tip": torch.stack([item["target_tip"] for item in items]),
        "obstacle_graph": Batch.from_data_list(
            [item["obstacle_graph"] for item in items]
        ),
    }


class PccDataset(Dataset):
    """Array-backed dataset with one ragged obstacle graph per path."""

    def __init__(
        self,
        paths: np.ndarray,
        target_tips: np.ndarray,
        target_indices: np.ndarray,
        start_indices: np.ndarray,
        ik_indices: np.ndarray,
        obstacle_centres: np.ndarray,
        obstacle_offsets: np.ndarray,
        section_lengths: tuple[float, float, float],
        sphere_radius: float,
        clearance_margin: float,
        normaliser: PccNormaliser | Mapping[str, Any] | None = None,
    ) -> None:
        paths = np.asarray(paths, dtype=np.float32)
        target_tips = np.asarray(target_tips, dtype=np.float32)
        count = len(paths)
        if paths.ndim != 3 or paths.shape[1] < 2 or paths.shape[2] != 6:
            raise ValueError("paths must have shape [N,H,6] with H >= 2")
        if target_tips.shape != (count, 3):
            raise ValueError("target_tips must have shape [N,3]")
        if count == 0 or not np.isfinite(paths).all() or not np.isfinite(target_tips).all():
            raise ValueError("dataset split must be nonempty and finite")

        target_indices = np.asarray(target_indices, dtype=np.int32)
        start_indices = np.asarray(start_indices, dtype=np.int16)
        ik_indices = np.asarray(ik_indices, dtype=np.int16)
        if any(values.shape != (count,) for values in (
            target_indices, start_indices, ik_indices
        )):
            raise ValueError("path index arrays must have shape [N]")
        if np.any(start_indices < 0):
            raise ValueError("start_indices must be non-negative")

        obstacle_centres = np.asarray(obstacle_centres, dtype=np.float32)
        obstacle_offsets = np.asarray(obstacle_offsets, dtype=np.int64)
        if obstacle_centres.ndim != 2 or obstacle_centres.shape[1] != 3:
            raise ValueError("obstacle_centres must have shape [M,3]")
        if not np.isfinite(obstacle_centres).all():
            raise ValueError("obstacle centres must be finite")
        if obstacle_offsets.shape != (count + 1,):
            raise ValueError("obstacle_offsets must have shape [N+1]")
        if (
            obstacle_offsets[0] != 0
            or obstacle_offsets[-1] != len(obstacle_centres)
            or np.any(np.diff(obstacle_offsets) < 0)
        ):
            raise ValueError("obstacle_offsets are inconsistent with obstacle centres")

        lengths = np.asarray(section_lengths, dtype=np.float64)
        if lengths.shape != (3,) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
            raise ValueError("section_lengths must contain three finite positive values")
        if not np.isfinite(sphere_radius) or sphere_radius <= 0:
            raise ValueError("sphere_radius must be finite and positive")
        if not np.isfinite(clearance_margin) or clearance_margin <= 0:
            raise ValueError("clearance_margin must be finite and positive")

        self.paths = np.ascontiguousarray(paths)
        self.target_tips = np.ascontiguousarray(target_tips)
        self.target_indices = np.ascontiguousarray(target_indices)
        self.start_indices = np.ascontiguousarray(start_indices)
        self.ik_indices = np.ascontiguousarray(ik_indices)
        self.obstacle_centres = np.ascontiguousarray(obstacle_centres)
        self.obstacle_offsets = np.ascontiguousarray(obstacle_offsets)
        self.section_lengths = tuple(float(value) for value in lengths)
        self.sphere_radius = float(sphere_radius)
        self.clearance_margin = float(clearance_margin)
        self.horizon = int(paths.shape[1])
        self.normaliser = self._make_normaliser(normaliser)

    def _make_normaliser(
        self, normaliser: PccNormaliser | Mapping[str, Any] | None
    ) -> PccNormaliser:
        if isinstance(normaliser, PccNormaliser):
            return normaliser
        if isinstance(normaliser, Mapping):
            return PccNormaliser.from_dict(normaliser)
        if normaliser is not None:
            raise TypeError("normaliser must be PccNormaliser, mapping, or None")
        q_min = np.minimum(self.paths.min(axis=(0, 1)), 0.0)
        q_max = np.maximum(self.paths.max(axis=(0, 1)), 0.0)
        q_centre = 0.5 * (q_min + q_max)
        q_scale = np.maximum(0.5 * (q_max - q_min), 1e-6)
        workspace_values = [self.target_tips]
        if len(self.obstacle_centres):
            workspace_values.append(self.obstacle_centres)
        workspace = np.concatenate(workspace_values, axis=0)
        workspace_min = workspace.min(axis=0)
        workspace_max = workspace.max(axis=0)
        workspace_centre = 0.5 * (workspace_min + workspace_max)
        workspace_scale = max(
            float(np.max(0.5 * (workspace_max - workspace_min))), 1e-6
        )
        return PccNormaliser(q_centre, q_scale, workspace_centre, workspace_scale)

    @classmethod
    def load(
        cls,
        dataset: str | Path,
        split: str = "train",
        normaliser: PccNormaliser | Mapping[str, Any] | None = None,
    ) -> "PccDataset":
        root = Path(dataset).expanduser().resolve()
        metadata_path = root / "dataset_meta.json"
        archive_path = root / "dataset.npz"
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if (
            metadata.get("format") != "pcc_diffuser_npz"
            or metadata.get("version") != DATASET_FORMAT_VERSION
        ):
            raise ValueError(f"unsupported dataset format in {metadata_path}")
        if not metadata.get("complete", False):
            raise ValueError(f"dataset is incomplete: {metadata_path}")
        requested_targets = metadata.get("requested_targets")
        actual_targets = metadata.get("actual_targets")
        requested_paths = metadata.get("requested_paths")
        actual_paths = metadata.get("actual_paths")
        if (
            not isinstance(requested_targets, int)
            or not isinstance(actual_targets, int)
            or not 0 <= actual_targets <= requested_targets
            or not isinstance(requested_paths, int)
            or not isinstance(actual_paths, int)
            or not 0 < actual_paths <= requested_paths
        ):
            raise ValueError("dataset target/path counts are missing or invalid")
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
            != requested_paths - actual_paths - unplanned_paths
            or path_failure_counts.get("total")
            != sum(
                value
                for reason, value in path_failure_counts.items()
                if reason != "total"
            )
        ):
            raise ValueError("dataset failure counts do not explain missing paths")
        section_lengths = metadata.get("section_lengths")
        if not isinstance(section_lengths, list):
            raise ValueError("dataset metadata has no section_lengths")
        split_codes = metadata.get("split_codes", {})
        if split not in split_codes:
            raise ValueError(f"unknown split {split!r}; choose from {sorted(split_codes)}")

        with np.load(archive_path, allow_pickle=False) as archive:
            required = {
                "paths", "target_tips", "target_indices", "start_indices",
                "ik_indices", "split_indices", "obstacle_centres", "obstacle_offsets",
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(f"{archive_path} is missing arrays: {missing}")
            if archive["paths"].shape[0] != actual_paths:
                raise ValueError("dataset metadata and NPZ path counts differ")
            if len(np.unique(archive["target_indices"])) != actual_targets:
                raise ValueError("dataset metadata and NPZ target counts differ")
            selected = np.flatnonzero(
                archive["split_indices"] == int(split_codes[split])
            )
            centres = archive["obstacle_centres"]
            offsets = archive["obstacle_offsets"]
            selected_centres = [
                centres[offsets[index] : offsets[index + 1]] for index in selected
            ]
            selected_offsets = np.zeros(len(selected) + 1, dtype=np.int64)
            if selected_centres:
                selected_offsets[1:] = np.cumsum([len(value) for value in selected_centres])
                packed_centres = np.concatenate(selected_centres, axis=0)
            else:
                packed_centres = np.empty((0, 3), dtype=np.float32)
            return cls(
                archive["paths"][selected],
                archive["target_tips"][selected],
                archive["target_indices"][selected],
                archive["start_indices"][selected],
                archive["ik_indices"][selected],
                packed_centres,
                selected_offsets,
                tuple(section_lengths),
                float(metadata["sphere_radius"]),
                float(metadata["clearance_margin"]),
                normaliser,
            )

    def obstacles(self, index: int) -> np.ndarray:
        begin = int(self.obstacle_offsets[index])
        end = int(self.obstacle_offsets[index + 1])
        return self.obstacle_centres[begin:end]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = torch.from_numpy(self.paths[index])
        target_tip = torch.from_numpy(self.target_tips[index])
        centres = torch.from_numpy(self.obstacles(index))
        centres = self.normaliser.normalise_workspace(centres)
        radius = self.normaliser.normalise_workspace_radius(self.sphere_radius)
        return {
            "path": self.normaliser.normalise_q(path),
            "start": self.normaliser.normalise_q(path[0]),
            "target_tip": self.normaliser.normalise_workspace(target_tip),
            "obstacle_graph": obstacle_graph(centres, radius),
        }


__all__ = [
    "DATASET_FORMAT_VERSION",
    "PccDataset",
    "collate_pcc_batch",
    "directed_edge_index",
    "obstacle_graph",
]
