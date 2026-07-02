from __future__ import annotations

from typing import Dict

from spaceera.schemas import SceneGraph
from spaceera.spatialmind.question_decomposition import decompose_question
from spaceera.spatialmind.scene_decomposition import decompose_scene


SYSTEM_PROMPT = """You are SpatialMind, a structured spatial reasoning assistant.
Use scene decomposition first, then solve the question with explicit reasoning steps.
Do not invent objects that are not present in the scene representation."""


def build_visual_prompt_package(question: str) -> Dict[str, object]:
    """Build SpatialMind instructions when only selected video frames are available."""

    question_payload = decompose_question(question)
    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": {
            "instruction": (
                "Use all selected frames as one scanning sequence. First perform scene decomposition: "
                "(1) identify objects and local relations in each view, (2) align repeated objects into "
                "a global 3D layout and a 10x10 top-down grid, and (3) write a concise spatial cognition "
                "summary. Then follow the question decomposition and answer from the aligned layout."
            ),
            "scene_decomposition": {
                "local_modeling": "infer from the selected frames",
                "coordinate_mapping": "construct a shared 3D map and 10x10 grid",
                "cognition_generation": "summarize grounded object relations",
            },
            "question_decomposition": question_payload,
        },
    }


def build_prompt_package(scene: SceneGraph, question: str) -> Dict[str, object]:
    scene_payload = decompose_scene(scene)
    question_payload = decompose_question(question)
    user_prompt = {
        "instruction": (
            "Solve the question with the provided scene decomposition. "
            "Use the 3D map, 2D grid, and textual cognition summary jointly."
        ),
        "scene_decomposition": scene_payload,
        "question_decomposition": question_payload,
    }
    return {
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
    }
