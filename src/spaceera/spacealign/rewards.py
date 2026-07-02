from __future__ import annotations

import math
import re
from typing import Mapping, Sequence, Tuple

Coordinate = Tuple[float, float]
_FORMAT_PATTERN = re.compile(
    r"^\s*<think>.+?</think>\s*<answer>.+?</answer>\s*$", re.DOTALL
)


def format_reward(response: str) -> float:
    return float(bool(_FORMAT_PATTERN.fullmatch(response)))


def multiple_choice_reward(prediction: str, target: str) -> float:
    return float(prediction.strip().casefold() == target.strip().casefold())


def numerical_reward(
    prediction: float,
    target: float,
    thresholds: Sequence[float] = tuple(index / 100 for index in range(50, 100, 5)),
) -> float:
    if not thresholds:
        raise ValueError("thresholds cannot be empty")
    if target == 0:
        return float(abs(prediction) <= 1e-8)
    relative_error = abs(prediction - target) / abs(target)
    return sum(relative_error <= 1.0 - threshold for threshold in thresholds) / len(thresholds)


def _shared_objects(
    predicted: Mapping[str, Coordinate], ground_truth: Mapping[str, Coordinate]
) -> list[str]:
    return sorted(set(predicted) & set(ground_truth))


def absolute_coordinate_reward(
    predicted: Mapping[str, Coordinate],
    ground_truth: Mapping[str, Coordinate],
    grid_size: int,
    category_counts: Mapping[str, int] | None = None,
) -> float:
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    objects = sorted(ground_truth)
    if not objects:
        return 0.0
    weights = {name: int((category_counts or {}).get(name, 1)) for name in objects}
    if any(weight <= 0 for weight in weights.values()):
        raise ValueError("category counts must be positive")
    denominator = sum(weights.values())
    diagonal = math.sqrt(2.0) * grid_size
    reward = 0.0
    for name in objects:
        if name not in predicted:
            continue
        px, py = predicted[name]
        gx, gy = ground_truth[name]
        normalized_distance = math.hypot(px - gx, py - gy) / diagonal
        reward += weights[name] / denominator * (1.0 - normalized_distance)
    return reward


def relative_relation_reward(
    predicted: Mapping[str, Coordinate],
    ground_truth: Mapping[str, Coordinate],
    grid_size: int,
    alpha: float = 0.5,
    epsilon: float = 1e-8,
) -> float:
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    objects = _shared_objects(predicted, ground_truth)
    if len(objects) < 2 or len(objects) != len(ground_truth):
        return 0.0
    discrepancies = []
    for first in objects:
        for second in objects:
            if first == second:
                continue
            predicted_vector = (
                predicted[second][0] - predicted[first][0],
                predicted[second][1] - predicted[first][1],
            )
            target_vector = (
                ground_truth[second][0] - ground_truth[first][0],
                ground_truth[second][1] - ground_truth[first][1],
            )
            predicted_norm = math.hypot(*predicted_vector)
            target_norm = math.hypot(*target_vector)
            dot = sum(a * b for a, b in zip(predicted_vector, target_vector))
            cosine = dot / (predicted_norm * target_norm + epsilon)
            cosine = max(-1.0, min(1.0, cosine))
            angular_loss = 1.0 - cosine
            distance_loss = abs(predicted_norm - target_norm) / (math.sqrt(2.0) * grid_size)
            discrepancies.append(alpha * angular_loss + (1.0 - alpha) * distance_loss)
    return 1.0 - sum(discrepancies) / len(discrepancies)


def spacealign_reward(
    *,
    response: str,
    task_reward: float,
    predicted_coordinates: Mapping[str, Coordinate],
    ground_truth_coordinates: Mapping[str, Coordinate],
    grid_size: int,
    category_counts: Mapping[str, int] | None = None,
    alpha: float = 0.5,
) -> dict[str, float]:
    components = {
        "format": format_reward(response),
        "task": float(task_reward),
        "absolute": absolute_coordinate_reward(
            predicted_coordinates, ground_truth_coordinates, grid_size, category_counts
        ),
        "relative": relative_relation_reward(
            predicted_coordinates, ground_truth_coordinates, grid_size, alpha
        ),
    }
    components["total"] = sum(components.values())
    return components
