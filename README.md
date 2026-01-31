# GENERAL: Strategic Game AI System

A production-grade deep reinforcement learning system for training AI agents to play strategic board games. Built on the AlphaZero paradigm, combining Monte Carlo Tree Search with deep neural networks.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Directory Structure](#directory-structure)
4. [Component Specifications](#component-specifications)
5. [Data Flow](#data-flow)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Training Pipeline](#training-pipeline)
9. [Evaluation](#evaluation)
10. [Cloud Deployment](#cloud-deployment)
11. [Configuration](#configuration)
12. [Performance Optimization](#performance-optimization)
13. [Testing](#testing)
14. [Troubleshooting](#troubleshooting)

---

## Overview

GENERAL implements an AlphaZero-style learning system that masters strategic games through self-play reinforcement learning. The system learns entirely from scratch without human game knowledge, using only the game rules.

### Core Principles

- **Self-Play Learning**: The agent plays against itself to generate training data
- **Neural Network Guidance**: A deep CNN evaluates positions and suggests moves
- **Monte Carlo Tree Search**: MCTS explores the game tree guided by the neural network
- **Iterative Improvement**: Each training iteration produces a stronger model

### Why AlphaZero?

| Feature | Benefit |
|---------|---------|
| No human knowledge required | Learns optimal strategies from first principles |
| MCTS + Neural Network | Combines deep learning with principled search |
| Self-play curriculum | Automatically generates harder training examples |
| Proven scalability | Demonstrated superhuman performance in Chess, Go, Shogi |

---

## Architecture

### System Overview

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|   Self-Play      |---->|  Replay Buffer   |---->|    Trainer       |
|   Engine         |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
        |                                                  |
        v                                                  v
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|   MCTS Engine    |<----|  Inference       |<----|  Neural Network  |
|                  |     |  Server          |     |                  |
+------------------+     +------------------+     +------------------+
        |                                                  |
        v                                                  v
+------------------+                              +------------------+
|                  |                              |                  |
|   Environment    |                              |      Arena       |
|                  |                              |   (Evaluation)   |
+------------------+                              +------------------+
```

### Training Loop

1. **Self-Play Generation**: Multiple games are played concurrently using MCTS guided by the current neural network
2. **Data Storage**: Game states, action probabilities, and outcomes are stored in the replay buffer
3. **Network Training**: The neural network is trained on accumulated self-play data
4. **Arena Evaluation**: The new model competes against the previous best model
5. **Model Selection**: If the new model achieves a win rate above threshold, it becomes the current best

---

## Directory Structure

```
Generals/
|
|-- main.py                     # Training orchestration entry point
|-- config.py                   # Centralized configuration parameters
|-- README.md                   # This document
|
|-- env/
|   |-- __init__.py
|   +-- generals_env.py         # Game environment implementation
|
|-- model/
|   |-- __init__.py
|   +-- network.py              # Neural network architecture (GeneralsNet)
|
|-- mcts/
|   |-- __init__.py
|   +-- mcts.py                 # Monte Carlo Tree Search (AsyncMCTS)
|
|-- training/
|   |-- __init__.py
|   |-- train.py                # Network trainer with AMP support
|   +-- replay_buffer.py        # Experience storage and management
|
|-- selfplay/
|   |-- __init__.py
|   +-- selfplay.py             # Concurrent self-play game generation
|
|-- evaluate/
|   |-- __init__.py
|   +-- evaluate.py             # Arena for model comparison
|
|-- utils/
|   |-- __init__.py
|   +-- batched_inference.py    # GPU batching for inference
|
|-- data/
|   |-- checkpoints/            # Model weight files (.pth)
|   +-- replay/                 # Training data files (.npz)
|
|-- notebooks/
|   +-- colab.ipynb             # Google Colab training notebook
|
+-- tests/
    |-- __init__.py
    |-- test_arena.py           # Arena evaluation tests
    |-- test_env.py             # Environment tests
    |-- test_mcts.py            # MCTS tests
    |-- test_replay.py          # Replay buffer tests
    |-- test_train.py           # Trainer tests
    |-- test_inference_reload.py # Inference server tests
    +-- test_conversion_rule.py # Game rule tests
```

---

## Component Specifications

### GeneralsEnv (env/generals_env.py)

The game environment implementing the Generals board game rules.

**Specifications:**
- Board Size: 10x10
- Action Space: 10004 discrete actions
- State Encoding: 9 channels of 10x10 spatial features
- Asymmetric Roles: Attacker vs Defender (determined by toss)

**Key Methods:**
- `reset()`: Initialize new game, return initial state
- `step(action_id)`: Execute action, return (next_state, reward, done, info)
- `get_legal_actions()`: Return list of valid actions
- `encode_state()`: Convert game state to neural network input
- `save_state()` / `restore_state()`: State checkpointing for MCTS

**State Encoding Format (9 Channels):**

| Channel | Description |
|---------|--------------|
| 0 | My units (binary) |
| 1 | Opponent units (binary) |
| 2 | General location |
| 3 | My role (1.0 if Attacker) |
| 4 | Opponent role (1.0 if Attacker) |
| 5 | Board ownership sign |
| 6 | Dice value (normalized) |
| 7 | General hits (normalized) |
| 8 | Moat cells |

### GeneralsNet (model/network.py)

Dual-head convolutional neural network for policy and value prediction.

**Architecture:**
- Input: 9 x 10 x 10 tensor
- Backbone: 7 residual blocks with 196 channels
- Policy Head: Outputs 10004 action logits
- Value Head: Outputs scalar value in [-1, 1]

**Parameter Count:** Approximately 8.5 million

**Network Structure:**
```
Input (9x10x10)
    |
Conv2d(9, 196) + BatchNorm + ReLU
    |
[ResidualBlock x 7]
    |
    +---> Policy Head: Conv(196,2) + FC(200, 10004)
    |
    +---> Value Head: Conv(196,1) + FC(100, 64) + FC(64, 1) + Tanh
```

### AsyncMCTS (mcts/mcts.py)

Asynchronous Monte Carlo Tree Search for action selection.

**Algorithm:**
1. Selection: Traverse tree using PUCT criterion
2. Expansion: Add new nodes using neural network priors
3. Evaluation: Get value estimate from neural network
4. Backpropagation: Update visit counts and values

**Key Parameters:**
- `c_puct`: Exploration constant (default: 1.0)
- `n_sims`: Number of simulations per move

**PUCT Formula:**
```
U(s,a) = c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))
score(s,a) = Q(s,a) + U(s,a)
```

### Trainer (training/train.py)

Handles neural network optimization with mixed precision support.

**Features:**
- Automatic Mixed Precision (AMP) when CUDA available
- KL Divergence loss for policy
- MSE loss for value
- Adam optimizer with weight decay

**Training Process:**
1. Load data from replay buffer
2. Shuffle and batch data
3. Forward pass with AMP autocast
4. Compute combined policy + value loss
5. Backward pass with gradient scaling
6. Save checkpoint

### ReplayBuffer (training/replay_buffer.py)

Manages storage and retrieval of training data.

**Features:**
- Persistent storage as compressed NPZ files
- Configurable maximum batch limit
- Automatic cleanup of old batches

**Data Format:**
- `states`: float32 array of shape (N, 9, 10, 10)
- `policies`: float32 array of shape (N, 10004)
- `values`: float32 array of shape (N,)

### SelfPlay (selfplay/selfplay.py)

Generates training data through self-play games.

**Features:**
- Concurrent game execution with asyncio
- Temperature-based action selection
- Configurable MCTS simulation count

**Temperature Schedule:**
- Moves 1-10: temperature = 1.0 (exploration)
- Moves 11+: temperature = 0.1 (exploitation)

### Arena (evaluate/evaluate.py)

Evaluates two models through head-to-head competition.

**Evaluation Protocol:**
- Models alternate playing first
- Each game plays to completion
- Win rate calculated as Model A wins / total games

### InferenceServer (utils/batched_inference.py)

Batches neural network inference requests for throughput.

**Features:**
- Async queue-based batching
- Configurable batch size and timeout
- Hot-reload of model weights during training

---

## Data Flow

### Training Iteration Flow

```
1. main.py initiates iteration
         |
         v
2. SelfPlay.play_iteration()
   - Creates N concurrent games
   - Each game runs MCTS at each move
   - MCTS queries InferenceServer
   - InferenceServer batches to GPU
   - GeneralsNet returns (policy, value)
   - Game continues until terminal
   - Returns (states, policies, values)
         |
         v
3. ReplayBuffer.add_game()
   - Saves to data/replay/batch_XXXX.npz
         |
         v
4. ReplayBuffer.load_all()
   - Concatenates all batches
         |
         v
5. Trainer.train()
   - Trains network on all data
   - Saves to data/checkpoints/model_latest.pth
         |
         v
6. InferenceServer.reload_model()
   - Loads new weights for next iteration
         |
         v
7. Arena.run()
   - Compares model_latest vs model_old
   - Returns win rate
         |
         v
8. Model Selection
   - If win_rate > threshold: copy latest -> old
   - Else: keep old as best
```

---

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- NumPy
- CUDA toolkit (optional, for GPU acceleration)

### Local Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Generals-AI.git
cd Generals-AI/Generals

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install torch numpy
```

### Verify Installation

```bash
python -c "from env import GeneralsEnv; print('Environment OK')"
python -c "from model import GeneralsNet; print('Network OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Usage

### Start Training

```bash
python main.py
```

The training loop will:
1. Generate self-play games
2. Train the neural network
3. Evaluate against previous model
4. Repeat indefinitely until interrupted

### Run Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_env.py
```

### Evaluate Models

```bash
python tests/test_arena.py
```

---

## Training Pipeline

### Default Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| games_per_iter | 32 | Games per self-play iteration |
| mcts_simulations | 100 | MCTS simulations per move |
| train_epochs | 3 | Training epochs per iteration |
| batch_size | 32 | Training batch size |
| learning_rate | 5e-4 | Adam learning rate |
| eval_games | 20 | Arena evaluation games |
| acceptance_threshold | 0.55 | Win rate to accept new model |

### Training Stages

**Stage 1: Initial Bootstrap (Iterations 1-10)**
- Network learns basic game mechanics
- Random-like play with gradual improvement

**Stage 2: Strategy Development (Iterations 10-50)**
- Emerging tactical patterns
- Improved position evaluation

**Stage 3: Refinement (Iterations 50+)**
- Fine-tuning of strategies
- Slower but consistent improvement

### Checkpoints

Models are saved to `data/checkpoints/`:
- `model_latest.pth`: Most recently trained model
- `model_old.pth`: Best model (used for evaluation)

---

## Evaluation

### Arena Evaluation

The Arena class conducts head-to-head matches between two models.

**Protocol:**
1. Load both models
2. Play N games, alternating first player
3. Count wins for each model
4. Report win rate

**Usage:**
```python
from evaluate import Arena

arena = Arena(
    model_A_path="data/checkpoints/model_latest.pth",
    model_B_path="data/checkpoints/model_old.pth",
    games=20,
    mcts_simulations=100
)
win_rate = await arena.run()
```

---

## Cloud Deployment

### Google Colab

1. Open `notebooks/colab.ipynb` in Google Colab
2. Set runtime to GPU (Runtime -> Change runtime type)
3. Run all cells sequentially

### Configuration for Cloud

Adjust parameters for cloud hardware:

```python
# For T4 GPU (Colab free tier)
TRAINING_CONFIG = {
    "games_per_iter": 64,
    "batch_size": 64,
    "mcts_simulations": 100
}

# For V100/A100 (Colab Pro)
TRAINING_CONFIG = {
    "games_per_iter": 128,
    "batch_size": 128,
    "mcts_simulations": 200
}
```

### Download Trained Model

After training in Colab:
```python
from google.colab import files
files.download("data/checkpoints/model_latest.pth")
```

---

## Configuration

### config.py Reference

```python
from config import NETWORK, TRAINING, EVAL, PATHS

# Network configuration
NETWORK.input_channels    # 9
NETWORK.board_size        # 10
NETWORK.action_dim        # 10004
NETWORK.hidden_channels   # 196
NETWORK.num_res_blocks    # 7

# Training configuration
TRAINING.games_per_iter          # 32
TRAINING.mcts_simulations        # 100
TRAINING.temperature_threshold   # 10
TRAINING.train_epochs            # 3
TRAINING.batch_size              # 32
TRAINING.learning_rate           # 5e-4
TRAINING.weight_decay            # 1e-4
TRAINING.max_replay_batches      # 20

# Evaluation configuration
EVAL.eval_games              # 20
EVAL.mcts_simulations        # 100
EVAL.acceptance_threshold    # 0.55

# Path configuration
PATHS.project_root      # Project root directory
PATHS.checkpoint_dir    # data/checkpoints
PATHS.replay_dir        # data/replay
```

---

## Performance Optimization

### Current Optimizations

| Feature | Implementation | Speedup |
|---------|---------------|---------|
| Mixed Precision Training | torch.amp.autocast | 2x |
| Batched Inference | InferenceServer | 5-10x |
| Concurrent Self-Play | asyncio.gather | Nx (N = games) |
| cuDNN Benchmark | cudnn.benchmark = True | 1.2x |

### Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1060 6GB | RTX 3080 or higher |
| CPU | 4 cores | 8+ cores |
| RAM | 8 GB | 16+ GB |
| Storage | 10 GB SSD | 50+ GB SSD |

### Scaling Guidelines

- **More Games**: Increase `games_per_iter` for faster data collection
- **Better Search**: Increase `mcts_simulations` for higher quality moves
- **Faster Training**: Increase `batch_size` (limited by GPU memory)
- **Longer Training**: Run more iterations for stronger models

---

## Testing

### Test Suite

| Test File | Coverage |
|-----------|----------|
| test_env.py | Environment state, actions, step logic |
| test_mcts.py | Tree search, action selection |
| test_train.py | Network training, checkpointing |
| test_replay.py | Data storage, loading |
| test_arena.py | Model evaluation |
| test_inference_reload.py | Hot model reload |
| test_conversion_rule.py | Game-specific rules |

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_env.py -v

# With coverage
python -m pytest tests/ --cov=. --cov-report=html
```

---

## Troubleshooting

### Common Issues

**Issue: CUDA out of memory**
```
Solution: Reduce batch_size or games_per_iter
```

**Issue: Training loss not decreasing**
```
Solution: 
1. Verify replay buffer has data
2. Check learning rate (try 1e-4)
3. Ensure model is in training mode
```

**Issue: Model not improving in Arena**
```
Solution:
1. Run more training iterations
2. Increase mcts_simulations
3. Increase eval_games for reliable statistics
```

**Issue: FileNotFoundError for checkpoints**
```
Solution: Run tests/test_train.py to create initial model
```

### Debug Commands

```bash
# Check GPU status
python -c "import torch; print(torch.cuda.is_available())"

# Verify model architecture
python -c "from model import GeneralsNet; m = GeneralsNet(); print(sum(p.numel() for p in m.parameters()))"

# Test environment
python tests/test_env.py
```

---

## License

MIT License. See LICENSE file for details.

---

## Acknowledgments

This project implements concepts from:
- Silver, D., et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
- Silver, D., et al. (2016). "Mastering the game of Go with deep neural networks and tree search"
