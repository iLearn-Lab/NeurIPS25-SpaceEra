from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from spaceera.schemas import SceneGraph
from spaceera.spatialmind.pipeline import build_prompt_package, build_visual_prompt_package
from spaceera.workflows.data import ScanForgeSample


SPACEALIGN_SYSTEM_PROMPT = """You are SpaceEra++, a visual-spatial reasoning assistant.
First infer an object layout on a 10x10 top-down grid, then reason from that layout.
Return exactly this structure:
<spatial_map>{"object_name": [x, y]}</spatial_map>
<think>your concise spatial reasoning</think>
<answer>your final answer only</answer>"""


def build_training_prompt(sample: ScanForgeSample) -> List[Dict[str, str]]:
    object_hint = ""
    if sample.ground_truth_coordinates:
        object_hint = (
            "\nThe spatial_map must contain these object keys: "
            + ", ".join(sorted(sample.ground_truth_coordinates))
            + "."
        )
    return [
        {"role": "system", "content": SPACEALIGN_SYSTEM_PROMPT},
        {"role": "user", "content": sample.question + object_hint},
    ]


def _spatialmind_text(sample: ScanForgeSample) -> str:
    if sample.scene_graph_path and Path(sample.scene_graph_path).exists():
        scene = SceneGraph.from_json(Path(sample.scene_graph_path))
        package = build_prompt_package(scene, sample.question)
    else:
        package = build_visual_prompt_package(sample.question)
    text = json.dumps(package["user_prompt"], ensure_ascii=False, indent=2)
    if sample.ground_truth_coordinates:
        text += (
            "\nUse exactly these object keys in <spatial_map>: "
            + ", ".join(sorted(sample.ground_truth_coordinates))
            + ". Their coordinates must be inferred from the selected frames."
        )
    return text


def build_inference_messages(sample: ScanForgeSample) -> List[Dict[str, Any]]:
    if not sample.selected_frames:
        raise ValueError("ScenePick must populate selected_frames before inference")
    user_content: List[Dict[str, str]] = [
        {"type": "image", "image": frame} for frame in sample.selected_frames
    ]
    user_content.append({"type": "text", "text": _spatialmind_text(sample)})
    return [
        {"role": "system", "content": SPACEALIGN_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
