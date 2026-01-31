from typing import TypedDict


class DifficultyConfig(TypedDict):
    mcts_simulations: int
    temperature: float


DIFFICULTY_SETTINGS: dict[str, DifficultyConfig] = {
    "easy": {"mcts_simulations": 50, "temperature": 1.2},
    "medium": {"mcts_simulations": 200, "temperature": 0.8},
    "hard": {"mcts_simulations": 1200, "temperature": 0.2},
}


def get_difficulty_config(difficulty: str) -> DifficultyConfig:
    if difficulty not in DIFFICULTY_SETTINGS:
        raise ValueError(f"Invalid difficulty: {difficulty}")
    return DIFFICULTY_SETTINGS[difficulty]
