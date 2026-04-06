from __future__ import annotations

QUESTION_TEMPLATES = {
    "relative_distance": {
        "template": "Measuring from the closest point of each object, which object is closest to the reference object?",
        "steps": [
            "Identify the reference object and the candidate target objects.",
            "Estimate object coordinates or relative positions from the scan.",
            "Compute the distance from each candidate to the reference object.",
            "Select the smallest distance as the answer.",
        ],
    },
    "object_count": {
        "template": "How many instances of a category are visible in the scene?",
        "steps": [
            "Identify the target category in the question.",
            "Scan the whole scene representation for all matching instances.",
            "Count distinct instances and exclude duplicates across frames.",
            "Return the total count.",
        ],
    },
    "appearance_order": {
        "template": "What is the first appearance order of categories in the video?",
        "steps": [
            "Identify the categories mentioned in the question.",
            "Locate the first timestamp where each category becomes visible.",
            "Sort categories by timestamp.",
            "Return the ordered list.",
        ],
    },
    "relative_direction": {
        "template": "If I am at object A and facing object B, where is object C?",
        "steps": [
            "Identify the anchor object, facing object, and target object.",
            "Build a local coordinate frame from the anchor to the facing object.",
            "Project the target object into the local frame.",
            "Classify the target as left, right, front, or back.",
        ],
    },
    "object_size": {
        "template": "What is the longest dimension of object A?",
        "steps": [
            "Identify the target object.",
            "Read its length, width, and height from the scene representation.",
            "Compare the three dimensions.",
            "Return the maximum value.",
        ],
    },
    "absolute_distance": {
        "template": "What is the distance between object A and object B?",
        "steps": [
            "Identify the two objects mentioned in the question.",
            "Read or estimate their coordinates from the scene map.",
            "Compute the Euclidean distance between them.",
            "Return the distance with units.",
        ],
    },
    "room_size": {
        "template": "What is the area of the room?",
        "steps": [
            "Identify which room or rooms are referenced.",
            "Read the room dimensions from the scene graph.",
            "Compute each room area and sum when needed.",
            "Return the total area.",
        ],
    },
    "route_plan": {
        "template": "Navigate from object A to object C while facing object B initially.",
        "steps": [
            "Identify the starting object, facing object, and destination.",
            "Use the scene layout to infer a collision-free route.",
            "Convert the route into ordered actions such as move, turn left, and turn right.",
            "Return the completed navigation plan.",
        ],
    },
}

