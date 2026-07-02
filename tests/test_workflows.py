from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from spaceera.spacealign.trl_rewards import (
    extract_answer,
    extract_spatial_map,
    spacealign_absolute_reward,
    spacealign_format_reward,
    spacealign_relative_reward,
    spacealign_task_reward,
)
from spaceera.scanforgeqa.scene_construction import build_scene_graph_from_file
from spaceera.workflows.data import ScanForgeSample, load_scanforgeqa, write_scanforgeqa
from spaceera.workflows.preparation import prepare_scanforgeqa_samples
from spaceera.workflows.prompts import build_inference_messages, build_training_prompt


VALID_COMPLETION = (
    '<spatial_map>{"sofa": [2, 3], "table": [4, 3]}</spatial_map>'
    '<think>The table is right of the sofa.</think><answer>2 meters</answer>'
)


class WorkflowDataTest(unittest.TestCase):
    def test_generated_qa_bundle_derives_spacealign_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            scene_path = root / "scene.json"
            build_scene_graph_from_file(Path("examples/scene_spec.json"), output_path=scene_path)
            video_path = root / "scan.mp4"
            video_path.touch()
            bundle = root / "qa.json"
            bundle.write_text(
                json.dumps(
                    {
                        "video_path": str(video_path),
                        "scene_graph_path": str(scene_path),
                        "qa_pairs": [
                            {
                                "qa_id": "qa1",
                                "question": "How many sofas are there?",
                                "answer": "1",
                                "question_type": "object_count",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            sample = load_scanforgeqa(bundle)[0]
            self.assertTrue(sample.ground_truth_coordinates)
            self.assertIn("sofa", sample.ground_truth_coordinates)

    def test_jsonl_loader_resolves_paths_and_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "samples.jsonl"
            source.write_text(
                json.dumps(
                    {
                        "qa_id": "qa1",
                        "video": "video.mp4",
                        "question": "distance?",
                        "answer": "2 meters",
                        "selected_frames": ["frame.jpg"],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            samples = load_scanforgeqa(source)
            self.assertEqual(samples[0].sample_id, "qa1")
            self.assertEqual(samples[0].video_path, str((root / "video.mp4").resolve()))
            output = root / "prepared.jsonl"
            write_scanforgeqa(output, samples)
            self.assertEqual(
                load_scanforgeqa(output)[0].selected_frames,
                [str((root / "frame.jpg").resolve())],
            )

    def test_preparation_skips_already_selected_samples_without_vggt(self) -> None:
        sample = ScanForgeSample("id", "video.mp4", "q", "a", selected_frames=["1.jpg"])
        prepared = prepare_scanforgeqa_samples(
            [sample], vggt_model="not-loaded", cache_dir=Path("unused")
        )
        self.assertEqual(prepared, [sample])


class IntegratedPromptTest(unittest.TestCase):
    def test_training_prompt_includes_supervised_object_keys(self) -> None:
        sample = ScanForgeSample(
            "id",
            "video.mp4",
            "Where is the table?",
            "right",
            ground_truth_coordinates={"table": [4, 3], "sofa": [2, 3]},
        )
        prompt = build_training_prompt(sample)
        self.assertIn("sofa, table", prompt[1]["content"])

    def test_inference_prompt_contains_images_and_spatialmind(self) -> None:
        sample = ScanForgeSample(
            "id", "video.mp4", "Where is the table?", "right", selected_frames=["1.jpg"]
        )
        messages = build_inference_messages(sample)
        self.assertEqual(messages[1]["content"][0]["type"], "image")
        self.assertIn("question_decomposition", messages[1]["content"][-1]["text"])


class TrlRewardAdapterTest(unittest.TestCase):
    def test_parsers_and_all_reward_components(self) -> None:
        completions = [[{"role": "assistant", "content": VALID_COMPLETION}]]
        coordinates = json.dumps({"sofa": [2, 3], "table": [4, 3]})
        self.assertEqual(extract_answer(VALID_COMPLETION), "2 meters")
        self.assertEqual(extract_spatial_map(VALID_COMPLETION)["table"], [4.0, 3.0])
        self.assertEqual(spacealign_format_reward(completions), [1.0])
        self.assertEqual(
            spacealign_task_reward(
                completions, solution=["2 meters"], question_type=["absolute_distance"]
            ),
            [1.0],
        )
        self.assertEqual(
            spacealign_absolute_reward(
                completions,
                ground_truth_coordinates_json=[coordinates],
                category_counts_json=[json.dumps({"sofa": 1, "table": 1})],
            ),
            [1.0],
        )
        self.assertAlmostEqual(
            spacealign_relative_reward(
                completions, ground_truth_coordinates_json=[coordinates]
            )[0],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
