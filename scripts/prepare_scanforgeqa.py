#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from spaceera.workflows.data import load_scanforgeqa, write_scanforgeqa
from spaceera.workflows.preparation import prepare_scanforgeqa_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare ScanForgeQA JSON/JSONL records with cached ScenePick frames."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--vggt-model", required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/scenepick_cache"))
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--voxel-resolution", type=float, default=20.0)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    samples = load_scanforgeqa(args.dataset, video_root=args.video_root)
    prepared = prepare_scanforgeqa_samples(
        samples,
        vggt_model=args.vggt_model,
        cache_dir=args.cache_dir,
        num_frames=args.num_frames,
        max_candidates=args.max_candidates,
        confidence_threshold=args.confidence_threshold,
        voxel_resolution=args.voxel_resolution,
        device=args.device,
        force=args.force,
    )
    write_scanforgeqa(args.output, prepared)
    print(f"Prepared {len(prepared)} samples -> {args.output}")


if __name__ == "__main__":
    main()
