from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

Voxel = Tuple[int, int, int]


@dataclass(frozen=True)
class FrameGeometry:
    """Per-pixel geometry and optional target-object mask for one frame."""

    frame_path: str
    points: np.ndarray
    confidence: np.ndarray
    semantic_mask: np.ndarray | None = None


@dataclass(frozen=True)
class SelectionResult:
    selected_frames: List[str]
    marginal_gains: List[float]
    cumulative_scores: List[float]
    covered_voxels: int
    candidate_frames: int
    total_voxels: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "selected_frames": self.selected_frames,
            "marginal_gains": self.marginal_gains,
            "cumulative_scores": self.cumulative_scores,
            "covered_voxels": self.covered_voxels,
            "candidate_frames": self.candidate_frames,
            "total_voxels": self.total_voxels,
        }


def uniform_sample_paths(paths: Sequence[str | Path], max_candidates: int) -> List[str]:
    """Choose at most ``max_candidates`` indices, including both endpoints."""

    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    ordered = [str(path) for path in paths]
    if len(ordered) <= max_candidates:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, max_candidates, dtype=int)
    return [ordered[index] for index in indices]


def _flatten_frame(frame: FrameGeometry) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(frame.points, dtype=np.float32).reshape(-1, 3)
    confidence = np.asarray(frame.confidence, dtype=np.float32).reshape(-1)
    if confidence.size != points.shape[0]:
        raise ValueError(f"confidence size does not match points for {frame.frame_path}")
    if frame.semantic_mask is None:
        semantic = np.zeros(points.shape[0], dtype=bool)
    else:
        semantic = np.asarray(frame.semantic_mask, dtype=bool).reshape(-1)
        if semantic.size != points.shape[0]:
            raise ValueError(f"semantic mask size does not match points for {frame.frame_path}")
    finite = np.isfinite(points).all(axis=1) & np.isfinite(confidence)
    return points[finite], confidence[finite], semantic[finite]


def build_weighted_voxel_sets(
    frames: Sequence[FrameGeometry],
    *,
    confidence_threshold: float = 0.1,
    voxel_resolution: float = 20.0,
    semantic_weight: float = 1.0,
    background_weight: float = 0.5,
) -> Tuple[Dict[str, set[Voxel]], Dict[Voxel, float]]:
    """Voxelize valid points and compute paper-defined mean semantic weights."""

    if not frames:
        raise ValueError("at least one frame is required")
    if voxel_resolution <= 0:
        raise ValueError("voxel_resolution must be positive")

    flattened = [_flatten_frame(frame) for frame in frames]
    valid_points = [points[confidence > confidence_threshold] for points, confidence, _ in flattened]
    valid_points = [points for points in valid_points if points.size]
    if not valid_points:
        raise ValueError("no points remain after confidence filtering")

    minimum = np.minimum.reduce([points.min(axis=0) for points in valid_points])
    maximum = np.maximum.reduce([points.max(axis=0) for points in valid_points])
    extent = maximum - minimum
    # A collapsed axis otherwise causes division by zero. It maps to voxel index 0.
    voxel_size = np.where(extent > 0, extent / voxel_resolution, 1.0)

    frame_voxels: Dict[str, set[Voxel]] = {}
    weight_sum: Dict[Voxel, float] = {}
    weight_count: Dict[Voxel, int] = {}

    for frame, (points, confidence, semantic) in zip(frames, flattened):
        valid = confidence > confidence_threshold
        points = points[valid]
        semantic = semantic[valid]
        indices = np.floor((points - minimum) / voxel_size).astype(np.int64)
        indices = np.clip(indices, 0, int(np.ceil(voxel_resolution)))
        unique_indices, inverse = np.unique(indices, axis=0, return_inverse=True)
        point_weights = np.where(semantic, semantic_weight, background_weight)
        sums = np.bincount(inverse, weights=point_weights)
        counts = np.bincount(inverse)
        voxel_set: set[Voxel] = set()
        for index, value_sum, value_count in zip(unique_indices, sums, counts):
            voxel = (int(index[0]), int(index[1]), int(index[2]))
            voxel_set.add(voxel)
            weight_sum[voxel] = weight_sum.get(voxel, 0.0) + float(value_sum)
            weight_count[voxel] = weight_count.get(voxel, 0) + int(value_count)
        frame_voxels[frame.frame_path] = voxel_set

    voxel_weights = {voxel: weight_sum[voxel] / weight_count[voxel] for voxel in weight_sum}
    return frame_voxels, voxel_weights


def greedy_maximum_coverage(
    frame_voxels: Mapping[str, set[Voxel]],
    voxel_weights: Mapping[Voxel, float],
    num_frames: int,
) -> SelectionResult:
    """Greedy weighted maximum coverage with deterministic tie breaking."""

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    remaining = dict(frame_voxels)
    covered: set[Voxel] = set()
    selected: List[str] = []
    gains: List[float] = []
    cumulative: List[float] = []
    score = 0.0

    while remaining and len(selected) < num_frames:
        best_path, best_gain = min(
            (
                (path, sum(voxel_weights.get(voxel, 1.0) for voxel in voxels - covered))
                for path, voxels in remaining.items()
            ),
            key=lambda item: (-item[1], item[0]),
        )
        covered.update(remaining.pop(best_path))
        selected.append(best_path)
        gains.append(round(float(best_gain), 6))
        score += float(best_gain)
        cumulative.append(round(score, 6))

    all_voxels = set().union(*frame_voxels.values()) if frame_voxels else set()
    return SelectionResult(
        selected_frames=selected,
        marginal_gains=gains,
        cumulative_scores=cumulative,
        covered_voxels=len(covered),
        candidate_frames=len(frame_voxels),
        total_voxels=len(all_voxels),
    )


def select_frames(
    frames: Sequence[FrameGeometry],
    num_frames: int,
    **voxel_options: float,
) -> SelectionResult:
    frame_voxels, voxel_weights = build_weighted_voxel_sets(frames, **voxel_options)
    return greedy_maximum_coverage(frame_voxels, voxel_weights, num_frames)
