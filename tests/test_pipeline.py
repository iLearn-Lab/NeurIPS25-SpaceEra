from pathlib import Path
import tempfile
import unittest

from spaceera.scanforgeqa.qa_generation import generate_qa_pairs
from spaceera.scanforgeqa.scan_creation import build_scan_sequence
from spaceera.scanforgeqa.scene_construction import build_scene_graph_from_file
from spaceera.spatialmind.pipeline import build_prompt_package


class PipelineTest(unittest.TestCase):
    def test_end_to_end(self) -> None:
        scene_spec = Path("examples/scene_spec.json")
        with tempfile.TemporaryDirectory() as temp_dir:
            scene_graph_path = Path(temp_dir) / "scene_graph.json"
            scene = build_scene_graph_from_file(scene_spec, output_path=scene_graph_path)
            prompt = build_prompt_package(scene, "What is the distance between the sofa and the coffee table?")
            scan = build_scan_sequence(scene)
            qas = generate_qa_pairs(scene)

        self.assertEqual(scene.scene_id, "living_room_demo")
        self.assertIn("scene_decomposition", prompt["user_prompt"])
        self.assertGreater(len(scan.steps), 0)
        self.assertGreater(len(qas), 0)


if __name__ == "__main__":
    unittest.main()
