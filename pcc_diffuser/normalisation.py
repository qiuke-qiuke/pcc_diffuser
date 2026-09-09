"""Serializable affine transforms for PCC paths and scene geometry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch


def _record_value(record: object, name: str) -> Any:
    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def _finite_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


class PccNormaliser:
    """Normalise configurations and world geometry while preserving metric shape.

    Configuration dimensions use independent affine transforms. Target positions
    and sphere centres share one three-dimensional centre and one scalar scale,
    so Euclidean workspace directions are not distorted.
    """

    VERSION = 2

    def __init__(
        self,
        q_centre: Any,
        q_scale: Any,
        workspace_centre: Any,
        workspace_scale: float,
    ) -> None:
        self.q_centre = _finite_array(q_centre, (6,), "q_centre")
        self.q_scale = _finite_array(q_scale, (6,), "q_scale")
        self.workspace_centre = _finite_array(
            workspace_centre, (3,), "workspace_centre"
        )
        self.workspace_scale = float(workspace_scale)
        if np.any(self.q_scale <= 0):
            raise ValueError("q_scale must be positive in every dimension")
        if not np.isfinite(self.workspace_scale) or self.workspace_scale <= 0:
            raise ValueError("workspace_scale must be positive and finite")

    @classmethod
    def fit(
        cls,
        records: Iterable[object],
        *,
        minimum_scale: float = 1e-6,
    ) -> "PccNormaliser":
        """Fit from physical paths, target tips, and ragged obstacle centres."""
        records = list(records)
        if not records:
            raise ValueError("cannot fit a normaliser without records")
        if minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive")

        paths = []
        targets = []
        obstacle_centres = []
        for index, record in enumerate(records):
            path = np.asarray(
                _record_value(record, "path"), dtype=np.float64
            )
            if path.ndim != 2 or path.shape[1] != 6:
                raise ValueError(
                    f"record {index} path must have shape [H,6], "
                    f"got {path.shape}"
                )
            if not np.isfinite(path).all():
                raise ValueError(f"record {index} path contains non-finite values")
            target = _finite_array(
                _record_value(record, "target_tip"), (3,), f"record {index} target_tip"
            )
            centres = np.asarray(
                _record_value(record, "obstacle_centres"), dtype=np.float64
            )
            if centres.ndim != 2 or centres.shape[1] != 3 or not np.isfinite(centres).all():
                raise ValueError(f"record {index} obstacle_centres must be finite [K,3]")
            paths.append(path)
            targets.append(target)
            if len(centres):
                obstacle_centres.append(centres)

        all_q = np.concatenate(paths, axis=0)
        # The planner hard-conditions q=0, so always keep it representable.
        q_min = np.minimum(all_q.min(axis=0), 0.0)
        q_max = np.maximum(all_q.max(axis=0), 0.0)
        q_centre = 0.5 * (q_min + q_max)
        q_half_range = 0.5 * (q_max - q_min)
        q_scale = np.where(q_half_range >= minimum_scale, q_half_range, 1.0)

        workspace_values = [np.stack(targets, axis=0), *obstacle_centres]
        workspace = np.concatenate(workspace_values, axis=0)
        workspace_min = workspace.min(axis=0)
        workspace_max = workspace.max(axis=0)
        workspace_centre = 0.5 * (workspace_min + workspace_max)
        # A scalar preserves Euclidean geometry and lets the same transform be
        # used by FK tip targets and obstacle centres.
        workspace_scale = float(np.max(0.5 * (workspace_max - workspace_min)))
        if workspace_scale < minimum_scale:
            workspace_scale = 1.0

        return cls(
            q_centre=q_centre,
            q_scale=q_scale,
            workspace_centre=workspace_centre,
            workspace_scale=workspace_scale,
        )

    @classmethod
    def from_records(
        cls, records: Iterable[object], *, minimum_scale: float = 1e-6
    ) -> "PccNormaliser":
        """Alias for :meth:`fit` for readable dataset construction."""
        return cls.fit(records, minimum_scale=minimum_scale)

    @staticmethod
    def _constant_like(value: Any, constant: Any) -> Any:
        if torch.is_tensor(value):
            if not torch.is_floating_point(value):
                raise TypeError("normalisation inputs must use a floating-point dtype")
            return torch.as_tensor(constant, dtype=value.dtype, device=value.device)
        array = np.asarray(value)
        if not np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float32)
        return np.asarray(constant, dtype=array.dtype)

    @staticmethod
    def _check_last_dim(value: Any, size: int, name: str) -> None:
        if np.ndim(value) == 0 or value.shape[-1] != size:
            raise ValueError(f"{name} must have last dimension {size}, got {value.shape}")

    def normalise_q(self, q: Any) -> Any:
        """Map physical configurations to approximately ``[-1, 1]``."""
        self._check_last_dim(q, 6, "q")
        centre = self._constant_like(q, self.q_centre)
        scale = self._constant_like(q, self.q_scale)
        return (q - centre) / scale

    def denormalise_q(self, q: Any) -> Any:
        """Map normalised configurations back to physical coordinates."""
        self._check_last_dim(q, 6, "q")
        centre = self._constant_like(q, self.q_centre)
        scale = self._constant_like(q, self.q_scale)
        return q * scale + centre

    def normalise_workspace(self, position: Any) -> Any:
        """Normalise target positions or sphere centres with one shared scale."""
        self._check_last_dim(position, 3, "position")
        centre = self._constant_like(position, self.workspace_centre)
        scale = self._constant_like(position, self.workspace_scale)
        return (position - centre) / scale

    def denormalise_workspace(self, position: Any) -> Any:
        """Undo :meth:`normalise_workspace`."""
        self._check_last_dim(position, 3, "position")
        centre = self._constant_like(position, self.workspace_centre)
        scale = self._constant_like(position, self.workspace_scale)
        return position * scale + centre

    def normalise_workspace_radius(self, radius: Any) -> Any:
        """Scale sphere radii consistently with workspace positions."""
        scale = self._constant_like(radius, self.workspace_scale)
        return radius / scale

    def denormalise_workspace_radius(self, radius: Any) -> Any:
        """Undo :meth:`normalise_workspace_radius`."""
        scale = self._constant_like(radius, self.workspace_scale)
        return radius * scale

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable checkpoint payload."""
        return {
            "version": self.VERSION,
            "q_center": self.q_centre.tolist(),
            "q_scale": self.q_scale.tolist(),
            "workspace_center": self.workspace_centre.tolist(),
            "workspace_scale": self.workspace_scale,
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, Any]) -> "PccNormaliser":
        """Restore a normaliser produced by :meth:`to_dict`."""
        version = int(state.get("version", 0))
        if version != cls.VERSION:
            raise ValueError(f"unsupported normaliser version {version}")
        required = (
            "q_center",
            "q_scale",
            "workspace_center",
            "workspace_scale",
        )
        missing = [key for key in required if key not in state]
        if missing:
            raise ValueError(f"normaliser state is missing {missing}")
        return cls(
            q_centre=state["q_center"],
            q_scale=state["q_scale"],
            workspace_centre=state["workspace_center"],
            workspace_scale=state["workspace_scale"],
        )

    def save_json(self, path: str | Path) -> None:
        """Write normalisation parameters as portable JSON."""
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "PccNormaliser":
        """Load parameters written by :meth:`save_json`."""
        with Path(path).open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))


__all__ = [
    "PccNormaliser",
]
