from pathlib import Path

from spaceera.scanforgeqa.scan_creation import export_blender_camera_script
from spaceera.scanforgeqa.scene_construction import build_scene_graph_from_file


def main() -> None:
    scene = build_scene_graph_from_file(Path("examples/scene_spec.json"))
    from spaceera.scanforgeqa.scan_creation import build_scan_sequence

    scan = build_scan_sequence(scene)
    export_blender_camera_script(scan, Path("generated_scan_camera.py"))


if __name__ == "__main__":
    main()

