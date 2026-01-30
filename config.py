import os
import contextlib
import multiprocessing
import torch
from dataclasses import dataclass, field
from pathlib import Path


def get_optimal_workers():
    try:
        if os.name == "nt":
            return 0
        cpu_count = multiprocessing.cpu_count()
        workers = max(1, cpu_count - 3)
        return min(8, workers)
    except Exception:
        return 1


def get_optimal_batch_size():
    if torch.cuda.is_available():
        with contextlib.suppress(Exception):
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            if vram_gb > 14:  # T4 has ~15-16GB
                return 128
            else:  # 8GB cards
                return 64
    return 64


@dataclass(frozen=True)
class NetworkConfig:
    input_channels: int = 17
    board_size: int = 10
    action_dim: int = 10003
    hidden_channels: int = 196
    num_res_blocks: int = 7


@dataclass(frozen=True)
class TrainingConfig:
    games_per_iter: int = 128
    mcts_simulations: int = 100
    temperature_threshold: int = 10
    train_epochs: int = 3
    batch_size: int = field(default_factory=get_optimal_batch_size)
    num_workers: int = field(default_factory=get_optimal_workers)
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
