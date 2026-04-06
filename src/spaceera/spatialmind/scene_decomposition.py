from __future__ import annotations

from typing import Dict, List

from spaceera.schemas import ObjectInstance, SceneGraph
from spaceera.utils import clamp


def _room_objects(scene: SceneGraph) -> List[ObjectInstance]:
    objects: List[ObjectInstance] = []
    for room in scene.rooms:
        objects.extend(room.objects)
    return objects


def build_local_modeling(scene: SceneGraph) -> Dict[str, Dict[str, object]]:
    local_views: Dict[str, Dict[str, object]] = {}
    for room in scene.rooms:
        if not room.objects:
            continue
        anchor = room.objects[0]
        relations = []
        for obj in room.objects:
            dx = round(obj.center[0] - anchor.center[0], 2)
            dy = round(obj.center[1] - anchor.center[1], 2)
            dz = round(obj.center[2] - anchor.center[2], 2)
            relations.append(
                {
                    "object_id": obj.object_id,
                    "category": obj.category,
                    "relative_to_anchor_m": [dx, dy, dz],
                }
            )
        local_views[room.room_id] = {
            "anchor_id": anchor.object_id,
            "anchor_category": anchor.category,
            "relations": relations,
        }
    return local_views


def build_coordinate_mapping(scene: SceneGraph, grid_size: int = 10) -> Dict[str, object]:
    objects = _room_objects(scene)
    if not objects:
        return {"reference_object_id": None, "map_3d": {}, "grid_2d": {}}

    reference = objects[0]
    max_x = max(obj.center[0] for obj in objects) or 1.0
    max_y = max(obj.center[1] for obj in objects) or 1.0

    map_3d: Dict[str, List[float]] = {}
    grid_2d: Dict[str, List[float]] = {}
    for obj in objects:
        relative = [round(obj.center[index] - reference.center[index], 2) for index in range(3)]
        map_3d[obj.object_id] = relative
        grid_2d[obj.object_id] = [
            round(clamp((obj.center[0] / max_x) * (grid_size - 1), 0, grid_size - 1), 1),
            round(clamp((obj.center[1] / max_y) * (grid_size - 1), 0, grid_size - 1), 1),
        ]

    return {
        "reference_object_id": reference.object_id,
        "reference_category": reference.category,
        "map_3d": map_3d,
        "grid_2d": grid_2d,
    }


def build_cognition_generation(scene: SceneGraph) -> Dict[str, str]:
    objects = _room_objects(scene)
    if not objects:
        return {}

    reference = objects[0]
    descriptions: Dict[str, str] = {}
    for obj in objects:
        dx = round(obj.center[0] - reference.center[0], 2)
        dy = round(obj.center[1] - reference.center[1], 2)
        dz = round(obj.center[2] - reference.center[2], 2)
        descriptions[obj.object_id] = (
            f"{obj.category} is located {dx} m along x, {dy} m along y, and {dz} m along z "
            f"relative to reference object {reference.object_id}."
        )
    return descriptions


def decompose_scene(scene: SceneGraph, grid_size: int = 10) -> Dict[str, object]:
    return {
        "scene_id": scene.scene_id,
        "local_modeling": build_local_modeling(scene),
        "coordinate_mapping": build_coordinate_mapping(scene, grid_size=grid_size),
        "cognition_generation": build_cognition_generation(scene),
    }

