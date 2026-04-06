from __future__ import annotations

from typing import List

from spaceera.schemas import QuestionAnswer, SceneGraph
from spaceera.spatialmind.question_bank import QUESTION_TEMPLATES
from spaceera.utils import euclidean_distance


def _all_objects(scene: SceneGraph):
    for room in scene.rooms:
        for obj in room.objects:
            yield obj


def _humanize(label: str) -> str:
    return label.replace("_", " ")


def _build_distance_qas(scene: SceneGraph) -> List[QuestionAnswer]:
    objects = list(_all_objects(scene))
    qas: List[QuestionAnswer] = []
    for index in range(max(0, len(objects) - 1)):
        first = objects[index]
        second = objects[index + 1]
        distance = euclidean_distance(first.center, second.center)
        qas.append(
            QuestionAnswer(
                qa_id=f"{scene.scene_id}_distance_{index:03d}",
                question_type="absolute_distance",
                question=f"What is the distance between the {_humanize(first.category)} and the {_humanize(second.category)}?",
                answer=f"{distance:.2f} meters",
                evidence_object_ids=[first.object_id, second.object_id],
                reasoning_steps=QUESTION_TEMPLATES["absolute_distance"]["steps"],
            )
        )
    return qas


def _build_count_qas(scene: SceneGraph) -> List[QuestionAnswer]:
    counts = {}
    for obj in _all_objects(scene):
        counts[obj.category] = counts.get(obj.category, 0) + 1

    qas: List[QuestionAnswer] = []
    for index, (category, count) in enumerate(sorted(counts.items())):
        qas.append(
            QuestionAnswer(
                qa_id=f"{scene.scene_id}_count_{index:03d}",
                question_type="object_count",
                question=f"How many {_humanize(category)} objects are there in the scene?",
                answer=str(count),
                evidence_object_ids=[obj.object_id for obj in _all_objects(scene) if obj.category == category],
                reasoning_steps=QUESTION_TEMPLATES["object_count"]["steps"],
            )
        )
    return qas


def _build_room_area_qas(scene: SceneGraph) -> List[QuestionAnswer]:
    qas: List[QuestionAnswer] = []
    for index, room in enumerate(scene.rooms):
        area = room.size[0] * room.size[1]
        qas.append(
            QuestionAnswer(
                qa_id=f"{scene.scene_id}_room_{index:03d}",
                question_type="room_size",
                question=f"What is the area of room {room.room_id}?",
                answer=f"{area:.2f} square meters",
                evidence_object_ids=[obj.object_id for obj in room.objects],
                reasoning_steps=QUESTION_TEMPLATES["room_size"]["steps"],
            )
        )
    return qas


def generate_qa_pairs(scene: SceneGraph) -> List[QuestionAnswer]:
    qas: List[QuestionAnswer] = []
    qas.extend(_build_distance_qas(scene))
    qas.extend(_build_count_qas(scene))
    qas.extend(_build_room_area_qas(scene))
    return qas
