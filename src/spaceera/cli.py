from __future__ import annotations

import argparse
import json
from pathlib import Path

from spaceera.scanforgeqa.qa_generation import generate_qa_pairs
from spaceera.scanforgeqa.scan_creation import build_scan_sequence, export_blender_camera_script
from spaceera.scanforgeqa.scene_construction import build_scene_graph_from_file
from spaceera.schemas import SceneGraph
from spaceera.spatialmind.pipeline import build_prompt_package


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_spatialmind(args: argparse.Namespace) -> None:
    scene = SceneGraph.from_json(Path(args.scene_graph))
    prompt_package = build_prompt_package(scene, args.question)
    _write_json(Path(args.output), prompt_package)


def run_build_scene(args: argparse.Namespace) -> None:
    build_scene_graph_from_file(Path(args.scene_spec), output_path=Path(args.output))


def run_build_scan(args: argparse.Namespace) -> None:
    scene = SceneGraph.from_json(Path(args.scene_graph))
    scan = build_scan_sequence(
        scene,
        scan_id=args.scan_id,
        orbit_radius=args.orbit_radius,
        orbit_views=args.orbit_views,
        navigation_stride=args.navigation_stride,
        fps=args.fps,
    )
    _write_json(Path(args.output), scan.to_dict())
    if args.blender_script:
        export_blender_camera_script(scan, Path(args.blender_script))


def run_generate_qa(args: argparse.Namespace) -> None:
    scene = SceneGraph.from_json(Path(args.scene_graph))
    qas = generate_qa_pairs(scene)
    payload = {"scene_id": scene.scene_id, "qa_pairs": [item.to_dict() for item in qas]}
    if args.video_path:
        payload["video_path"] = str(Path(args.video_path).resolve())
        payload["scene_graph_path"] = str(Path(args.scene_graph).resolve())
    _write_json(Path(args.output), payload)


def run_scenepick(args: argparse.Namespace) -> None:
    from spaceera.scenepick.batch import run_batch

    run_batch(
        video_dir=Path(args.video_dir),
        output_file=Path(args.output),
        model_path=args.model,
        num_frames=args.num_frames,
        max_candidates=args.max_candidates,
        confidence_threshold=args.confidence_threshold,
        voxel_resolution=args.voxel_resolution,
        grounding_json=Path(args.grounding_json) if args.grounding_json else None,
        device=args.device,
    )


def run_spacealign_reward(args: argparse.Namespace) -> None:
    from spaceera.spacealign.rewards import spacealign_reward

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    result = spacealign_reward(
        response=payload["response"],
        task_reward=float(payload["task_reward"]),
        predicted_coordinates=payload["predicted_coordinates"],
        ground_truth_coordinates=payload["ground_truth_coordinates"],
        grid_size=int(payload.get("grid_size", 10)),
        category_counts=payload.get("category_counts"),
        alpha=float(payload.get("alpha", 0.5)),
    )
    _write_json(Path(args.output), result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SpaceEra project CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    spatialmind = subparsers.add_parser("spatialmind", help="Build a SpatialMind prompt package.")
    spatialmind.add_argument("--scene-graph", required=True)
    spatialmind.add_argument("--question", required=True)
    spatialmind.add_argument("--output", required=True)
    spatialmind.set_defaults(func=run_spatialmind)

    scene_builder = subparsers.add_parser("build-scene", help="Construct a scene graph from a scene spec.")
    scene_builder.add_argument("--scene-spec", required=True)
    scene_builder.add_argument("--output", required=True)
    scene_builder.set_defaults(func=run_build_scene)

    scan_builder = subparsers.add_parser("build-scan", help="Construct a scan trajectory from a scene graph.")
    scan_builder.add_argument("--scene-graph", required=True)
    scan_builder.add_argument("--output", required=True)
    scan_builder.add_argument("--scan-id", default="scan_000")
    scan_builder.add_argument("--orbit-radius", type=float, default=1.0)
    scan_builder.add_argument("--orbit-views", type=int, default=8)
    scan_builder.add_argument("--navigation-stride", type=float, default=1.5)
    scan_builder.add_argument("--fps", type=int, default=3)
    scan_builder.add_argument("--blender-script")
    scan_builder.set_defaults(func=run_build_scan)

    qa_builder = subparsers.add_parser("generate-qa", help="Generate QA pairs from a scene graph.")
    qa_builder.add_argument("--scene-graph", required=True)
    qa_builder.add_argument("--output", required=True)
    qa_builder.add_argument(
        "--video-path",
        help="Attach a rendered scan so the output can be consumed by integrated training.",
    )
    qa_builder.set_defaults(func=run_generate_qa)

    scene_picker = subparsers.add_parser(
        "scenepick", help="Select SpaceEra++ key frames with VGGT reconstruction."
    )
    scene_picker.add_argument(
        "--video-dir", required=True, help="Frame-directory root or a single video file."
    )
    scene_picker.add_argument("--model", required=True, help="VGGT model name or local path.")
    scene_picker.add_argument("--output", required=True)
    scene_picker.add_argument("--num-frames", type=int, default=32)
    scene_picker.add_argument("--max-candidates", type=int, default=64)
    scene_picker.add_argument("--confidence-threshold", type=float, default=0.1)
    scene_picker.add_argument("--voxel-resolution", type=float, default=20.0)
    scene_picker.add_argument("--grounding-json")
    scene_picker.add_argument("--device", choices=["cpu", "cuda"])
    scene_picker.set_defaults(func=run_scenepick)

    reward = subparsers.add_parser(
        "spacealign-reward", help="Evaluate the SpaceAlign reward components from JSON."
    )
    reward.add_argument("--input", required=True)
    reward.add_argument("--output", required=True)
    reward.set_defaults(func=run_spacealign_reward)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
