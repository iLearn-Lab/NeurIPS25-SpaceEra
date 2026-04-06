import argparse
from pathlib import Path

from spaceera.schemas import SceneGraph
from spaceera.spatialmind.scene_decomposition import decompose_scene


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SpatialMind scene decomposition from a scene graph.")
    parser.add_argument("--scene-graph", required=True, help="Path to a scene_graph.json file.")
    args = parser.parse_args()

    scene = SceneGraph.from_json(Path(args.scene_graph))
    payload = decompose_scene(scene)
    print(payload)


if __name__ == "__main__":
    main()
