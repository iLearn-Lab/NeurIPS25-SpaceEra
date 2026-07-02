from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Sequence

from spaceera.spacealign.rewards import (
    absolute_coordinate_reward,
    multiple_choice_reward,
    numerical_reward,
    relative_relation_reward,
)

_MAP_PATTERN = re.compile(r"<spatial_map>\s*(\{.*?\})\s*</spatial_map>", re.DOTALL)
_ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
_FULL_PATTERN = re.compile(
    r"^\s*<spatial_map>\s*\{.*?\}\s*</spatial_map>\s*"
    r"<think>.+?</think>\s*<answer>.+?</answer>\s*$",
    re.DOTALL,
)
_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")
_NUMERICAL_TYPES = {
    "absolute_distance",
    "relative_distance",
    "object_size",
    "room_size",
    "object_count",
}


def _content(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        item = completion[0]
        if isinstance(item, dict):
            return str(item.get("content", ""))
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    return str(completion)


def _value_at(value: Any, index: int, default: Any = None) -> Any:
    if isinstance(value, (list, tuple)):
        return value[index]
    return default if value is None else value


def _json_mapping(value: Any) -> Dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    return dict(value)


def extract_answer(response: str) -> str:
    match = _ANSWER_PATTERN.search(response)
    return match.group(1).strip() if match else response.strip()


def extract_spatial_map(response: str) -> Dict[str, List[float]]:
    match = _MAP_PATTERN.search(response)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(1))
        return {
            str(name): [float(coordinates[0]), float(coordinates[1])]
            for name, coordinates in payload.items()
            if isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2
        }
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def spacealign_format_reward(completions: Sequence[Any], **_: Any) -> List[float]:
    return [float(bool(_FULL_PATTERN.fullmatch(_content(item)))) for item in completions]


def spacealign_task_reward(
    completions: Sequence[Any],
    solution: Sequence[str],
    question_type: Sequence[str] | str = "open_ended",
    **_: Any,
) -> List[float]:
    rewards: List[float] = []
    for index, completion in enumerate(completions):
        predicted = extract_answer(_content(completion))
        target = str(_value_at(solution, index, ""))
        kind = str(_value_at(question_type, index, question_type))
        if kind in _NUMERICAL_TYPES:
            predicted_number = _NUMBER_PATTERN.search(predicted)
            target_number = _NUMBER_PATTERN.search(target)
            if not predicted_number or not target_number:
                rewards.append(0.0)
            else:
                rewards.append(
                    numerical_reward(float(predicted_number.group()), float(target_number.group()))
                )
        else:
            rewards.append(multiple_choice_reward(predicted, target))
    return rewards


def spacealign_absolute_reward(
    completions: Sequence[Any],
    ground_truth_coordinates_json: Sequence[str] | str,
    category_counts_json: Sequence[str] | str | None = None,
    grid_size: Sequence[int] | int = 10,
    **_: Any,
) -> List[float]:
    rewards: List[float] = []
    for index, completion in enumerate(completions):
        predicted = extract_spatial_map(_content(completion))
        target = _json_mapping(_value_at(ground_truth_coordinates_json, index, "{}"))
        counts = _json_mapping(_value_at(category_counts_json, index, "{}"))
        size = int(_value_at(grid_size, index, grid_size))
        rewards.append(absolute_coordinate_reward(predicted, target, size, counts))
    return rewards


def spacealign_relative_reward(
    completions: Sequence[Any],
    ground_truth_coordinates_json: Sequence[str] | str,
    grid_size: Sequence[int] | int = 10,
    alpha: Sequence[float] | float = 0.5,
    **_: Any,
) -> List[float]:
    rewards: List[float] = []
    for index, completion in enumerate(completions):
        predicted = extract_spatial_map(_content(completion))
        target = _json_mapping(_value_at(ground_truth_coordinates_json, index, "{}"))
        size = int(_value_at(grid_size, index, grid_size))
        balance = float(_value_at(alpha, index, alpha))
        rewards.append(relative_relation_reward(predicted, target, size, balance))
    return rewards
