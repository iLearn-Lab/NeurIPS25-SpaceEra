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
    _write_json(Path(args.output), {"scene_id": scene.scene_id, "qa_pairs": [item.to_dict() for item in qas]})


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
    qa_builder.set_defaults(func=run_generate_qa)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
