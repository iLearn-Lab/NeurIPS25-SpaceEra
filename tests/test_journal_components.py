from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np

from spaceera.scenepick.sampling import (
    FrameGeometry,
    greedy_maximum_coverage,
    select_frames,
    uniform_sample_paths,
)
from spaceera.scenepick.batch import discover_frame_directories
from spaceera.spacealign.rewards import (
    absolute_coordinate_reward,
    format_reward,
    numerical_reward,
    relative_relation_reward,
    spacealign_reward,
)


class ScenePickTest(unittest.TestCase):
    def test_discovers_nested_frame_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            frames = Path(temp_dir) / "split" / "scene"
            frames.mkdir(parents=True)
            (frames / "0001.jpg").touch()
            self.assertEqual(discover_frame_directories(Path(temp_dir)), [frames])

    def test_uniform_sampling_includes_endpoints_without_duplicates(self) -> None:
        sampled = uniform_sample_paths([str(index) for index in range(10)], 4)
        self.assertEqual(sampled, ["0", "3", "6", "9"])

    def test_weighted_semantic_coverage_changes_selection(self) -> None:
        frame_a = FrameGeometry(
            "a.jpg",
            points=np.array([[[0, 0, 0], [1, 0, 0]]], dtype=float),
            confidence=np.ones((1, 2)),
            semantic_mask=np.array([[False, False]]),
        )
        frame_b = FrameGeometry(
            "b.jpg",
            points=np.array([[[0, 0, 0], [0, 1, 0]]], dtype=float),
            confidence=np.ones((1, 2)),
            semantic_mask=np.array([[True, True]]),
        )
        result = select_frames([frame_a, frame_b], 1, voxel_resolution=2)
        self.assertEqual(result.selected_frames, ["b.jpg"])
        self.assertLess(result.covered_voxels, result.total_voxels)

    def test_greedy_ties_are_deterministic(self) -> None:
        result = greedy_maximum_coverage(
            {"b.jpg": {(0, 0, 0)}, "a.jpg": {(1, 0, 0)}},
            {(0, 0, 0): 1.0, (1, 0, 0): 1.0},
            1,
        )
        self.assertEqual(result.selected_frames, ["a.jpg"])


class SpaceAlignTest(unittest.TestCase):
    def test_perfect_layout_receives_full_spatial_rewards(self) -> None:
        coordinates = {"a": (1, 1), "b": (4, 5)}
        self.assertAlmostEqual(absolute_coordinate_reward(coordinates, coordinates, 10), 1.0)
        self.assertAlmostEqual(relative_relation_reward(coordinates, coordinates, 10), 1.0)

    def test_format_and_numerical_rewards(self) -> None:
        self.assertEqual(format_reward("<think>x</think><answer>A</answer>"), 1.0)
        self.assertEqual(format_reward("A"), 0.0)
        self.assertEqual(numerical_reward(10, 10), 1.0)

    def test_total_is_component_sum(self) -> None:
        reward = spacealign_reward(
            response="<think>x</think><answer>A</answer>",
            task_reward=1.0,
            predicted_coordinates={"a": (1, 1), "b": (2, 2)},
            ground_truth_coordinates={"a": (1, 1), "b": (2, 2)},
            grid_size=10,
        )
        self.assertAlmostEqual(reward["total"], 4.0)

    def test_missing_objects_do_not_receive_layout_credit(self) -> None:
        predicted = {"a": (1, 1)}
        target = {"a": (1, 1), "b": (2, 2)}
        self.assertAlmostEqual(absolute_coordinate_reward(predicted, target, 10), 0.5)
        self.assertEqual(relative_relation_reward(predicted, target, 10), 0.0)


if __name__ == "__main__":
    unittest.main()
