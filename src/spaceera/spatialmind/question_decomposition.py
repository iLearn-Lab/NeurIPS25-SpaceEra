from __future__ import annotations

from difflib import SequenceMatcher
from typing import Dict, List, Tuple

from spaceera.spatialmind.question_bank import QUESTION_TEMPLATES


def classify_question(question: str) -> Tuple[str, float]:
    lowered = question.strip().lower()
    best_name = "open_ended"
    best_score = 0.0
    for name, payload in QUESTION_TEMPLATES.items():
        score = SequenceMatcher(None, lowered, payload["template"].lower()).ratio()
        if score > best_score:
            best_name = name
            best_score = score
    return best_name, best_score


def build_generic_steps(question: str) -> List[str]:
    return [
        "Identify the entities, quantities, or room regions referenced in the question.",
        "Ground those entities in the decomposed scene representations.",
        "Perform the required spatial comparison, counting, or path reasoning.",
        "Return a concise answer supported by the derived spatial evidence.",
    ]


def decompose_question(question: str, threshold: float = 0.33) -> Dict[str, object]:
    question_type, confidence = classify_question(question)
    if confidence >= threshold and question_type in QUESTION_TEMPLATES:
        steps = QUESTION_TEMPLATES[question_type]["steps"]
    else:
        question_type = "open_ended"
        steps = build_generic_steps(question)
    return {
        "question": question,
        "question_type": question_type,
        "confidence": round(confidence, 3),
        "reasoning_steps": steps,
    }

