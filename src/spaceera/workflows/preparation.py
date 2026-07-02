from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Dict, List, Sequence

from spaceera.scenepick.batch import (
    VIDEO_SUFFIXES,
    discover_frame_directories,
    extract_video_candidates,
    frame_paths,
)
from spaceera.scenepick.grounding import frame_semantic_mask
from spaceera.scenepick.sampling import FrameGeometry, select_frames, uniform_sample_paths
from spaceera.workflows.data import ScanForgeSample


def _candidate_paths(video_path: Path, cache_dir: Path, max_candidates: int) -> List[str]:
    if video_path.is_file():
        if video_path.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"unsupported video type: {video_path}")
        digest = hashlib.sha1(str(video_path.resolve()).encode("utf-8")).hexdigest()[:10]
        output_dir = cache_dir / "frames" / f"{video_path.stem}_{digest}_{max_candidates}"
        return extract_video_candidates(video_path, output_dir, max_candidates)
    if not video_path.is_dir():
        raise FileNotFoundError(video_path)
    direct = frame_paths(video_path)
    if direct:
        return uniform_sample_paths(direct, max_candidates)
    discovered = discover_frame_directories(video_path)
    if len(discovered) != 1:
        raise ValueError(
            f"{video_path} must directly contain frames or resolve to exactly one frame directory"
        )
    return uniform_sample_paths(frame_paths(discovered[0]), max_candidates)


def _with_semantics(
    geometry: Sequence[FrameGeometry], sample: ScanForgeSample
) -> List[FrameGeometry]:
    if not sample.grounding_boxes:
        return list(geometry)
    enriched: List[FrameGeometry] = []
    for frame in geometry:
        image_shape = frame.points.shape[:2]
        enriched.append(
            FrameGeometry(
                frame_path=frame.frame_path,
                points=frame.points,
                confidence=frame.confidence,
                semantic_mask=frame_semantic_mask(
                    frame.frame_path, image_shape, sample.grounding_boxes
                ),
            )
        )
    return enriched


def prepare_scanforgeqa_samples(
    samples: Sequence[ScanForgeSample],
    *,
    vggt_model: str,
    cache_dir: Path,
    num_frames: int = 32,
    max_candidates: int = 64,
    confidence_threshold: float = 0.1,
    voxel_resolution: float = 20.0,
    device: str | None = None,
    force: bool = False,
) -> List[ScanForgeSample]:
    """Run ScenePick once per video and cache geometry within this preparation run."""

    if num_frames > max_candidates:
        raise ValueError("num_frames cannot exceed max_candidates")
    pending = [sample for sample in samples if force or not sample.selected_frames]
    if not pending:
        return list(samples)

    from spaceera.scenepick.vggt_backend import VGGTReconstructor

    cache_dir.mkdir(parents=True, exist_ok=True)
    reconstructor = VGGTReconstructor(vggt_model, device=device)
    prepared: List[ScanForgeSample] = list(samples)
    grouped: Dict[str, List[int]] = {}
    for index, sample in enumerate(samples):
        if force or not sample.selected_frames:
            video_key = str(Path(sample.video_path).resolve())
            grouped.setdefault(video_key, []).append(index)

    # Keep only one video's dense point maps in memory at a time. All questions
    # for that video reuse the same reconstruction with different semantic masks.
    for video_key, indices in grouped.items():
        candidates = _candidate_paths(Path(video_key), cache_dir, max_candidates)
        geometry = reconstructor.reconstruct(candidates)
        for index in indices:
            sample = samples[index]
            result = select_frames(
                _with_semantics(geometry, sample),
                num_frames,
                confidence_threshold=confidence_threshold,
                voxel_resolution=voxel_resolution,
            )
            metadata = dict(sample.metadata)
            metadata["scenepick"] = result.to_dict()
            prepared[index] = replace(
                sample, selected_frames=result.selected_frames, metadata=metadata
            )
        del geometry
    return prepared
