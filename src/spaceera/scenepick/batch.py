from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from spaceera.scenepick.grounding import load_grounding_boxes
from spaceera.scenepick.sampling import uniform_sample_paths
from spaceera.scenepick.vggt_backend import VGGTReconstructor

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


def discover_frame_directories(root: Path) -> List[Path]:
    """Find leaf-like directories containing image frames at any nesting depth."""

    return sorted(
        directory
        for directory in (root, *root.rglob("*"))
        if directory.is_dir()
        and any(
            child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
            for child in directory.iterdir()
        )
    )


def frame_paths(directory: Path) -> List[str]:
    return [
        str(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]


def extract_video_candidates(
    video_path: Path, output_dir: Path, max_candidates: int
) -> List[str]:
    """Uniformly extract persistent candidate images from a video using FFmpeg."""

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("direct video input requires ffmpeg and ffprobe on PATH")
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(probe.stdout.strip())
    if duration <= 0:
        raise ValueError(f"video has invalid duration: {video_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "frame_%06d.jpg"
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps={max_candidates}/{duration}",
            "-frames:v",
            str(max_candidates),
            "-q:v",
            "2",
            str(output_pattern),
        ],
        check=True,
    )
    extracted = frame_paths(output_dir)
    if not extracted:
        raise RuntimeError(f"FFmpeg did not extract frames from {video_path}")
    return extracted[:max_candidates]


def run_batch(
    *,
    video_dir: Path,
    output_file: Path,
    model_path: str,
    num_frames: int = 32,
    max_candidates: int = 64,
    confidence_threshold: float = 0.1,
    voxel_resolution: float = 20.0,
    semantic_weight: float = 1.0,
    background_weight: float = 0.5,
    grounding_json: Path | None = None,
    device: str | None = None,
) -> List[Dict[str, object]]:
    if num_frames > max_candidates:
        raise ValueError("num_frames cannot exceed max_candidates")
    if not video_dir.exists():
        raise FileNotFoundError(video_dir)
    if video_dir.is_file() and video_dir.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError(f"unsupported video input: {video_dir}")

    reconstructor = VGGTReconstructor(model_path, device=device)
    grounding = load_grounding_boxes(grounding_json) if grounding_json else None
    records: List[Dict[str, object]] = []

    from spaceera.scenepick.sampling import select_frames

    if video_dir.is_file():
        extracted_dir = output_file.parent / f"{video_dir.stem}_{max_candidates}_candidates"
        sources = [(video_dir.stem, extract_video_candidates(video_dir, extracted_dir, max_candidates))]
    else:
        sources = [
            (
                directory.name
                if directory.relative_to(video_dir) == Path(".")
                else directory.relative_to(video_dir).as_posix(),
                uniform_sample_paths(frame_paths(directory), max_candidates),
            )
            for directory in discover_frame_directories(video_dir)
        ]

    for video_id, candidates in sources:
        geometry = reconstructor.reconstruct(candidates, grounding_boxes=grounding)
        result = select_frames(
            geometry,
            num_frames,
            confidence_threshold=confidence_threshold,
            voxel_resolution=voxel_resolution,
            semantic_weight=semantic_weight,
            background_weight=background_weight,
        )
        record = result.to_dict()
        record["video_id"] = video_id
        if video_dir.is_file():
            record["source_video"] = str(video_dir)
        records.append(record)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records
