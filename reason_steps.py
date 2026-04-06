from spaceera.spatialmind.question_decomposition import decompose_question


def get_solution_steps(question: str) -> list[str]:
    return decompose_question(question)["reasoning_steps"]


if __name__ == "__main__":
    input_question = "How far is the desk from the chair?"
    steps = get_solution_steps(input_question)
    print("Solution Steps:")
    for index, step in enumerate(steps, start=1):
        print(f"Step{index}: {step}")
