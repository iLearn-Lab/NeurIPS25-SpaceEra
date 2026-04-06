from __future__ import annotations

from typing import Dict

from spaceera.schemas import SceneGraph
from spaceera.spatialmind.question_decomposition import decompose_question
from spaceera.spatialmind.scene_decomposition import decompose_scene


SYSTEM_PROMPT = """You are SpatialMind, a structured spatial reasoning assistant.
Use scene decomposition first, then solve the question with explicit reasoning steps.
Do not invent objects that are not present in the scene representation."""


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

