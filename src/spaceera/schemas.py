from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json


@dataclass
class ObjectInstance:
    object_id: str
    category: str
    center: List[float]
    size: List[float]
    room_id: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Room:
    room_id: str
    room_type: str
    size: List[float]
    origin: List[float]
    objects: List[ObjectInstance] = field(default_factory=list)


@dataclass
class SceneGraph:
    scene_id: str
    rooms: List[Room]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SceneGraph":
        rooms: List[Room] = []
        for room_payload in payload.get("rooms", []):
            objects = [ObjectInstance(**item) for item in room_payload.get("objects", [])]
            rooms.append(
                Room(
                    room_id=room_payload["room_id"],
                    room_type=room_payload["room_type"],
                    size=room_payload["size"],
                    origin=room_payload["origin"],
                    objects=objects,
                )
            )
        return cls(
            scene_id=payload["scene_id"],
            rooms=rooms,
            metadata=payload.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, path: Path) -> "SceneGraph":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


@dataclass
class VideoFrame:
    frame_id: str
    timestamp_sec: float
    visible_object_ids: List[str]
    observations: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ScanTrajectoryStep:
    step_id: str
    action: str
    position: List[float]
    yaw_deg: float
    note: str = ""


@dataclass
class ScanSequence:
    scene_id: str
    scan_id: str
    steps: List[ScanTrajectoryStep]
    frames: List[VideoFrame]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


@dataclass
class QuestionAnswer:
    qa_id: str
    question_type: str
    question: str
    answer: str
    evidence_object_ids: List[str]
    reasoning_steps: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

