from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from spaceera.schemas import ObjectInstance, Room, SceneGraph
from spaceera.utils import dump_json, load_json


def _expand_room_objects(room_payload: Dict[str, object], room_index: int) -> Room:
    objects: List[ObjectInstance] = []
    for object_index, obj in enumerate(room_payload.get("objects", []), start=1):
        object_id = f"room{room_index}_{obj['category']}_{object_index}"
        objects.append(
            ObjectInstance(
                object_id=object_id,
                category=obj["category"],
                center=obj["center"],
                size=obj.get("size", [1.0, 1.0, 1.0]),
                room_id=room_payload["room_id"],
                attributes=obj.get("attributes", {}),
            )
        )
    return Room(
        room_id=room_payload["room_id"],
        room_type=room_payload.get("room_type", "generic"),
        size=room_payload["size"],
        origin=room_payload.get("origin", [0.0, 0.0, 0.0]),
        objects=objects,
    )


def build_scene_graph(scene_spec: Dict[str, object]) -> SceneGraph:
    rooms = [
        _expand_room_objects(room_payload, room_index=index)
        for index, room_payload in enumerate(scene_spec.get("rooms", []), start=1)
    ]
    return SceneGraph(
        scene_id=scene_spec["scene_id"],
        rooms=rooms,
        metadata=scene_spec.get("metadata", {}),
    )


def build_scene_graph_from_file(scene_spec_path: Path, output_path: Path | None = None) -> SceneGraph:
    scene = build_scene_graph(load_json(scene_spec_path))
    if output_path is not None:
        dump_json(output_path, scene.to_dict())
    return scene

