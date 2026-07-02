"""Backward-compatible entry point for batch ScenePick selection.

Prefer ``spaceera scenepick --help`` for new usage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from spaceera.scenepick.batch import run_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ScenePick over nested frame directories.")
    parser.add_argument(
        "--video_dir", type=Path, required=True, help="Frame-directory root or video file"
    )
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--model", required=True, help="VGGT model name or local path")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--max_candidates", type=int, default=64)
    parser.add_argument("--confidence_threshold", type=float, default=0.1)
    parser.add_argument("--voxel_resolution", type=float, default=20.0)
    parser.add_argument("--grounding_json", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    args = parser.parse_args()
    run_batch(
        video_dir=args.video_dir,
        output_file=args.output_file,
        model_path=args.model,
        num_frames=args.num_frames,
        max_candidates=args.max_candidates,
        confidence_threshold=args.confidence_threshold,
        voxel_resolution=args.voxel_resolution,
        grounding_json=args.grounding_json,
        device=args.device,
    )


if __name__ == "__main__":
    main()
