from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json
import math


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def euclidean_distance(point_a: list[float], point_b: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(point_a, point_b)))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))

