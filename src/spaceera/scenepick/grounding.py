from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np


def load_grounding_boxes(path: Path) -> Dict[str, List[List[float]]]:
    """Load ``{frame name/path: [[x1, y1, x2, y2], ...]}`` detections."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("grounding JSON must be an object keyed by frame path or file name")
    return {str(key): [list(map(float, box)) for box in boxes] for key, boxes in payload.items()}


def boxes_to_mask(height: int, width: int, boxes: Sequence[Sequence[float]]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for box in boxes:
        if len(box) != 4:
            raise ValueError("each grounding box must contain x1, y1, x2, y2")
        x1, y1, x2, y2 = box
        left, right = sorted((max(0, round(x1)), min(width, round(x2))))
        top, bottom = sorted((max(0, round(y1)), min(height, round(y2))))
        mask[top:bottom, left:right] = True
    return mask


def frame_semantic_mask(
    frame_path: str,
    image_shape: Sequence[int],
    grounding_boxes: Mapping[str, Sequence[Sequence[float]]],
) -> np.ndarray:
    key = frame_path if frame_path in grounding_boxes else Path(frame_path).name
    boxes = grounding_boxes.get(key, [])
    return boxes_to_mask(int(image_shape[0]), int(image_shape[1]), boxes)
