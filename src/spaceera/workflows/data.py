from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class ScanForgeSample:
    """Canonical record shared by ScenePick, GRPO training, and inference."""

    sample_id: str
    video_path: str
    question: str
    answer: str
    question_type: str = "open_ended"
    selected_frames: List[str] = field(default_factory=list)
    ground_truth_coordinates: Dict[str, List[float]] = field(default_factory=dict)
    category_counts: Dict[str, int] = field(default_factory=dict)
    grounding_boxes: Dict[str, List[List[float]]] = field(default_factory=dict)
    scene_graph_path: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_selected_frames(self, frames: Sequence[str]) -> "ScanForgeSample":
        return replace(self, selected_frames=list(frames))


def _first(payload: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in payload and payload[name] is not None:
            return payload[name]
    return default


def _resolve_path(value: str | None, base_dir: Path, override_root: Path | None) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    if override_root is not None:
        return str((override_root / path).resolve())
    return str((base_dir / path).resolve())


def _normalise_record(
    payload: Mapping[str, Any],
    *,
    index: int,
    base_dir: Path,
    inherited: Mapping[str, Any] | None = None,
    video_root: Path | None = None,
) -> ScanForgeSample:
    merged = dict(inherited or {})
    merged.update(payload)
    sample_id = str(_first(merged, ("sample_id", "qa_id", "id"), f"sample_{index:06d}"))
    video = _first(merged, ("video_path", "video", "source_video", "frame_dir"))
    question = _first(merged, ("question", "problem", "query"))
    answer = _first(merged, ("answer", "solution", "ground_truth"))
    if not video:
        raise ValueError(f"{sample_id}: missing video_path/video/source_video/frame_dir")
    if question is None or answer is None:
        raise ValueError(f"{sample_id}: both question and answer are required")

    selected = [
        _resolve_path(str(path), base_dir, video_root) or str(path)
        for path in _first(merged, ("selected_frames", "frame_files", "images"), [])
    ]
    scene_graph = _resolve_path(
        _first(merged, ("scene_graph_path", "scene_graph")), base_dir, None
    )
    known = {
        "sample_id", "qa_id", "id", "video_path", "video", "source_video", "frame_dir",
        "question", "problem", "query", "answer", "solution", "ground_truth", "question_type",
        "selected_frames", "frame_files", "images", "ground_truth_coordinates", "coordinates",
        "category_counts", "grounding_boxes", "scene_graph_path", "scene_graph", "metadata",
    }
    metadata = dict(merged.get("metadata", {}))
    metadata.update({key: value for key, value in merged.items() if key not in known})
    coordinates = {
        str(key): [float(value[0]), float(value[1])]
        for key, value in _first(
            merged, ("ground_truth_coordinates", "coordinates"), {}
        ).items()
    }
    counts = {
        str(key): int(value) for key, value in merged.get("category_counts", {}).items()
    }
    if not coordinates and scene_graph and Path(scene_graph).exists():
        coordinates, counts = _coordinates_from_scene_graph(Path(scene_graph))
    return ScanForgeSample(
        sample_id=sample_id,
        video_path=_resolve_path(str(video), base_dir, video_root) or str(video),
        question=str(question),
        answer=str(answer),
        question_type=str(merged.get("question_type", "open_ended")),
        selected_frames=selected,
        ground_truth_coordinates=coordinates,
        category_counts=counts,
        grounding_boxes={
            str(key): [list(map(float, box)) for box in boxes]
            for key, boxes in merged.get("grounding_boxes", {}).items()
        },
        scene_graph_path=scene_graph,
        metadata=metadata,
    )


def _coordinates_from_scene_graph(path: Path) -> tuple[Dict[str, List[float]], Dict[str, int]]:
    from spaceera.schemas import SceneGraph
    from spaceera.spatialmind.scene_decomposition import build_coordinate_mapping

    scene = SceneGraph.from_json(path)
    mapping = build_coordinate_mapping(scene, grid_size=10)["grid_2d"]
    objects = [obj for room in scene.rooms for obj in room.objects]
    totals: Dict[str, int] = {}
    for obj in objects:
        totals[obj.category] = totals.get(obj.category, 0) + 1
    seen: Dict[str, int] = {}
    coordinates: Dict[str, List[float]] = {}
    for obj in objects:
        seen[obj.category] = seen.get(obj.category, 0) + 1
        key = obj.category if totals[obj.category] == 1 else f"{obj.category}_{seen[obj.category]}"
        coordinates[key] = list(mapping[obj.object_id])
    return coordinates, {key: 1 for key in coordinates}


def _read_payloads(path: Path) -> tuple[List[Mapping[str, Any]], Mapping[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return records, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload, {}
    if not isinstance(payload, dict):
        raise ValueError("ScanForgeQA input must be JSON/JSONL objects")
    for key in ("samples", "records", "data"):
        if isinstance(payload.get(key), list):
            inherited = {name: value for name, value in payload.items() if name != key}
            return payload[key], inherited
    if isinstance(payload.get("qa_pairs"), list):
        inherited = {name: value for name, value in payload.items() if name != "qa_pairs"}
        return payload["qa_pairs"], inherited
    return [payload], {}


def load_scanforgeqa(path: Path, video_root: Path | None = None) -> List[ScanForgeSample]:
    records, inherited = _read_payloads(path)
    return [
        _normalise_record(
            record,
            index=index,
            base_dir=path.parent.resolve(),
            inherited=inherited,
            video_root=video_root.resolve() if video_root else None,
        )
        for index, record in enumerate(records)
    ]


def write_scanforgeqa(path: Path, samples: Iterable[ScanForgeSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(sample.to_dict(), ensure_ascii=False) for sample in samples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
