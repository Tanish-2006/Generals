from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NetworkConfig:
    input_channels: int = 17
    board_size: int = 10
    action_dim: int = 10003
    hidden_channels: int = 196
    num_res_blocks: int = 7


@dataclass(frozen=True)
class TrainingConfig:
    games_per_iter: int = 32
    mcts_simulations: int = 100
    temperature_threshold: int = 10
    train_epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    max_replay_batches: int = 20


@dataclass(frozen=True)
class EvalConfig:
    eval_games: int = 20
    mcts_simulations: int = 100
    acceptance_threshold: float = 0.55


@dataclass
class PathConfig:
    def __init__(self, project_root: Path = None):
        if project_root is None:
            project_root = Path(__file__).parent.resolve()
        self.project_root = project_root
        self.checkpoint_dir = project_root / "data" / "checkpoints"
        self.replay_dir = project_root / "data" / "replay"

    def ensure_dirs(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.replay_dir.mkdir(parents=True, exist_ok=True)


NETWORK = NetworkConfig()
TRAINING = TrainingConfig()
EVAL = EvalConfig()
PATHS = PathConfig()
