"""Training loop for the MCTS value/policy network.

Provides TrainingConfig, TrainingStats, train_value_network(),
SelfPlayConfig, save_checkpoint, load_checkpoint, and self_play_loop()
for the AlphaZero-style training loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from monopoly_engine.actions import EndTurn
from monopoly_engine.game import MonopolyGame
from monopoly_gym.action_space import ActionEncoder

from .data import ReplayBuffer, generate_training_data
from .features import get_feature_size
from .network import ValueNetwork
from .search import MCTSConfig, MCTSSearch, _auto_roll_dice


@dataclass
class TrainingConfig:
    """Configuration for value network training.

    Attributes:
        learning_rate: Initial learning rate for Adam.
        lr_decay: Multiplicative LR decay per epoch.
        batch_size: Training batch size.
        num_epochs: Epochs per training iteration.
        value_loss_weight: Weight for value loss (MSE).
        policy_loss_weight: Weight for policy loss (CrossEntropy).
        weight_decay: L2 regularization.
        validation_fraction: Fraction of data for validation.
        device: Torch device ("cpu" or "cuda").
    """

    learning_rate: float = 1e-3
    lr_decay: float = 0.99
    batch_size: int = 256
    num_epochs: int = 10
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    weight_decay: float = 1e-4
    validation_fraction: float = 0.1
    device: str = "cpu"


@dataclass
class TrainingStats:
    """Statistics from a training run.

    Attributes:
        epoch_losses: List of (value_loss, policy_loss, total_loss) per epoch.
        val_losses: List of (value_loss, policy_loss, total_loss) per epoch.
    """

    epoch_losses: list[tuple[float, float, float]] = field(default_factory=list)
    val_losses: list[tuple[float, float, float]] = field(default_factory=list)


def _build_datasets(
    buffer: ReplayBuffer,
    validation_fraction: float,
) -> tuple[TensorDataset, TensorDataset | None]:
    """Convert replay buffer to PyTorch datasets with train/val split.

    Args:
        buffer: Replay buffer with training examples.
        validation_fraction: Fraction of data for validation.

    Returns:
        Tuple of (train_dataset, val_dataset). val_dataset is None if
        the buffer is too small for a meaningful split.
    """
    examples = buffer.sample(len(buffer))

    features = np.array([e.features for e in examples], dtype=np.float32)
    policies = np.array([e.mcts_policy for e in examples], dtype=np.float32)
    outcomes = np.array([e.outcome for e in examples], dtype=np.float32)

    features_t = torch.from_numpy(features)
    policies_t = torch.from_numpy(policies)
    outcomes_t = torch.from_numpy(outcomes)

    n = len(examples)
    val_size = int(n * validation_fraction)

    if val_size < 1 or n - val_size < 1:
        # Not enough data for a validation split
        return TensorDataset(features_t, policies_t, outcomes_t), None

    # Shuffle indices for split
    perm = torch.randperm(n)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    train_ds = TensorDataset(features_t[train_idx], policies_t[train_idx], outcomes_t[train_idx])
    val_ds = TensorDataset(features_t[val_idx], policies_t[val_idx], outcomes_t[val_idx])

    return train_ds, val_ds


def _compute_loss(
    network: ValueNetwork,
    features: torch.Tensor,
    target_policies: torch.Tensor,
    target_outcomes: torch.Tensor,
    config: TrainingConfig,
) -> tuple[torch.Tensor, float, float]:
    """Compute combined value + policy loss.

    Args:
        network: The value network.
        features: Input features (batch, input_size).
        target_policies: MCTS visit-count distributions (batch, 149).
        target_outcomes: Actual game outcomes (batch, num_players).
        config: Training configuration.

    Returns:
        Tuple of (total_loss, value_loss_scalar, policy_loss_scalar).
    """
    values, policy_logits = network(features)

    # Value loss: MSE between predicted and actual outcomes
    value_loss = nn.functional.mse_loss(values, target_outcomes)

    # Policy loss: cross-entropy with MCTS policy as target distribution
    # target_policies is a probability distribution (sums to ~1)
    log_probs = nn.functional.log_softmax(policy_logits, dim=-1)
    policy_loss = -(target_policies * log_probs).sum(dim=-1).mean()

    total_loss = config.value_loss_weight * value_loss + config.policy_loss_weight * policy_loss

    return total_loss, value_loss.item(), policy_loss.item()


def _evaluate_validation(
    network: ValueNetwork,
    val_loader: DataLoader[tuple[torch.Tensor, ...]],
    config: TrainingConfig,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate on validation set.

    Args:
        network: The value network.
        val_loader: DataLoader for validation data.
        config: Training configuration.
        device: Torch device.

    Returns:
        Tuple of (value_loss, policy_loss, total_loss) averaged over batches.
    """
    network.eval()
    total_val = 0.0
    total_vloss = 0.0
    total_ploss = 0.0
    count = 0

    with torch.no_grad():
        for batch in val_loader:
            feat, pol, out = batch
            feat = feat.to(device)
            pol = pol.to(device)
            out = out.to(device)

            loss, vloss, ploss = _compute_loss(network, feat, pol, out, config)
            total_val += loss.item()
            total_vloss += vloss
            total_ploss += ploss
            count += 1

    network.train()

    if count == 0:
        return 0.0, 0.0, 0.0

    return total_vloss / count, total_ploss / count, total_val / count


def train_value_network(
    network: ValueNetwork,
    buffer: ReplayBuffer,
    config: TrainingConfig | None = None,
    tensorboard_dir: str | Path | None = None,
) -> TrainingStats:
    """Train the value network on collected self-play data.

    Loss = value_loss_weight * MSE(value, outcome)
         + policy_loss_weight * CrossEntropy(policy, mcts_policy)

    Args:
        network: The value network to train (modified in-place).
        buffer: Replay buffer with training examples.
        config: Training configuration. Uses defaults if None.
        tensorboard_dir: Optional directory for TensorBoard logs.

    Returns:
        Training statistics with per-epoch losses.

    Raises:
        ValueError: If the replay buffer is empty.
    """
    if len(buffer) == 0:
        raise ValueError("Cannot train on an empty replay buffer.")

    if config is None:
        config = TrainingConfig()

    device = torch.device(config.device)
    network.to(device)
    network.train()

    # Build datasets
    train_ds, val_ds = _build_datasets(buffer, config.validation_fraction)

    train_loader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False
    )
    val_loader: DataLoader[tuple[torch.Tensor, ...]] | None = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size, shuffle=False, drop_last=False
        )

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)

    # Optional TensorBoard writer
    writer = None
    if tensorboard_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(tensorboard_dir))

    stats = TrainingStats()

    for epoch in range(config.num_epochs):
        epoch_vloss = 0.0
        epoch_ploss = 0.0
        epoch_total = 0.0
        batch_count = 0

        for batch in train_loader:
            feat, pol, out = batch
            feat = feat.to(device)
            pol = pol.to(device)
            out = out.to(device)

            optimizer.zero_grad()
            loss, vloss, ploss = _compute_loss(network, feat, pol, out, config)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()

            epoch_vloss += vloss
            epoch_ploss += ploss
            epoch_total += loss.item()
            batch_count += 1

        scheduler.step()

        # Average over batches
        avg_vloss = epoch_vloss / max(batch_count, 1)
        avg_ploss = epoch_ploss / max(batch_count, 1)
        avg_total = epoch_total / max(batch_count, 1)
        stats.epoch_losses.append((avg_vloss, avg_ploss, avg_total))

        # Validation
        if val_loader is not None:
            val_vloss, val_ploss, val_total = _evaluate_validation(
                network, val_loader, config, device
            )
            stats.val_losses.append((val_vloss, val_ploss, val_total))

            if writer is not None:
                writer.add_scalar("val/value_loss", val_vloss, epoch)
                writer.add_scalar("val/policy_loss", val_ploss, epoch)
                writer.add_scalar("val/total_loss", val_total, epoch)

        # TensorBoard logging
        if writer is not None:
            writer.add_scalar("train/value_loss", avg_vloss, epoch)
            writer.add_scalar("train/policy_loss", avg_ploss, epoch)
            writer.add_scalar("train/total_loss", avg_total, epoch)
            writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], epoch)

    if writer is not None:
        writer.close()

    return stats


# ---------------------------------------------------------------------------
# B5: Self-Play Iteration (AlphaZero Loop) + B6: Checkpoint Management
# ---------------------------------------------------------------------------


@dataclass
class SelfPlayConfig:
    """Configuration for the AlphaZero-style self-play training loop.

    Attributes:
        num_iterations: Number of self-play iterations.
        games_per_iteration: Games to play per iteration.
        mcts_simulations: MCTS simulations per move during self-play.
        num_players: Players per game.
        temperature: MCTS temperature for data generation.
        eval_games: Games for evaluation after each iteration.
        eval_frequency: Evaluate every N iterations.
        acceptance_threshold: Win rate vs previous model to accept new model.
        max_turns_per_game: Max turns before truncating.
        training_config: Configuration for network training.
    """

    num_iterations: int = 50
    games_per_iteration: int = 100
    mcts_simulations: int = 100
    num_players: int = 4
    temperature: float = 1.0
    eval_games: int = 50
    eval_frequency: int = 5
    acceptance_threshold: float = 0.55
    max_turns_per_game: int = 500
    training_config: TrainingConfig = field(default_factory=TrainingConfig)


def save_checkpoint(
    path: str | Path,
    network: ValueNetwork,
    buffer: ReplayBuffer,
    iteration: int,
    stats: dict[str, Any],
) -> None:
    """Save a training checkpoint to a directory.

    Creates:
        path/network.pt     -- model weights + config (via ValueNetwork.save)
        path/buffer.npz     -- replay buffer (via ReplayBuffer.save)
        path/metadata.json  -- iteration, stats, network config

    Args:
        path: Directory to save checkpoint in (created if absent).
        network: The value network.
        buffer: The replay buffer.
        iteration: Current iteration number.
        stats: Arbitrary stats dict (win rates, loss history, etc.).
    """
    ckpt_path = Path(path)
    ckpt_path.mkdir(parents=True, exist_ok=True)

    network.save(ckpt_path / "network.pt")
    buffer.save(ckpt_path / "buffer.npz")

    metadata: dict[str, Any] = {
        "iteration": iteration,
        "stats": stats,
        "network_config": {
            "input_size": network.input_size,
            "num_players": network.num_players,
            "action_size": network.action_size,
            "hidden_size": network.hidden_size,
            "trunk_layers": network.trunk_layers,
        },
    }
    with open(ckpt_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def load_checkpoint(
    path: str | Path,
) -> tuple[ValueNetwork, ReplayBuffer, int, dict[str, Any]]:
    """Load a training checkpoint from a directory.

    Args:
        path: Directory containing network.pt, buffer.npz, metadata.json.

    Returns:
        Tuple of (network, buffer, iteration, stats).
    """
    ckpt_path = Path(path)
    network = ValueNetwork.load(ckpt_path / "network.pt")
    buffer = ReplayBuffer.load(ckpt_path / "buffer.npz")

    with open(ckpt_path / "metadata.json") as f:
        metadata: dict[str, Any] = json.load(f)

    iteration = int(metadata["iteration"])
    stats: dict[str, Any] = metadata.get("stats", {})

    return network, buffer, iteration, stats


def _run_game_with_networks(
    new_network: ValueNetwork,
    old_network: ValueNetwork | None,
    config: SelfPlayConfig,
    seed: int | None = None,
) -> int | None:
    """Play a single evaluation game and return the winning player id.

    Player 0 uses MCTSSearch with new_network (greedy, temperature=0).
    Player 1 uses MCTSSearch with old_network when provided.
    All other players use rule_based policy via MCTSSearch._get_opponent_action.

    Args:
        new_network: The candidate network (plays as player 0).
        old_network: The reference network (plays as player 1), or None.
        config: Self-play config (num_players, mcts_simulations, max_turns).
        seed: Random seed for game creation.

    Returns:
        Winning player id, or None if game was truncated.
    """
    game = MonopolyGame(num_players=config.num_players, seed=seed)
    encoder = ActionEncoder(enable_trades=False)

    eval_mcts_cfg = MCTSConfig(
        num_simulations=config.mcts_simulations,
        use_value_network=True,
        temperature=0.0,
        opponent_policy="rule_based",
    )
    mcts_new = MCTSSearch(eval_mcts_cfg, value_network=new_network)
    mcts_old: MCTSSearch | None = None
    if old_network is not None:
        mcts_old = MCTSSearch(eval_mcts_cfg, value_network=old_network)

    _auto_roll_dice(game)
    turn = 0
    while not game.game_over and turn < config.max_turns_per_game:
        pid = game.decision_player

        if game.players[pid].bankrupt:
            end = EndTurn(player_id=pid)
            valid, _ = end.validate(game)
            if valid:
                game.apply_action(end.player_id, end)
                _auto_roll_dice(game)
            turn += 1
            continue

        if pid == 0:
            visit_counts = mcts_new.search(game, player_id=0)
            if not visit_counts:
                turn += 1
                continue
            action_idx = mcts_new.select_action(visit_counts, temperature=0.0)
        elif pid == 1 and mcts_old is not None:
            visit_counts = mcts_old.search(game, player_id=1)
            if not visit_counts:
                turn += 1
                continue
            action_idx = mcts_old.select_action(visit_counts, temperature=0.0)
        else:
            action_idx = mcts_new._get_opponent_action(game, pid)

        action = encoder.decode(action_idx, pid, game)
        game.apply_action(action.player_id, action)

        if game.decision_player != pid:
            _auto_roll_dice(game)

        turn += 1

    return game.winner


def self_play_loop(
    config: SelfPlayConfig,
    save_dir: str | Path,
    resume_from: str | Path | None = None,
) -> None:
    """Run the full AlphaZero-style self-play training loop.

    Per iteration:
      1. Generate training data via MCTS self-play
      2. Add examples to replay buffer
      3. Train network on replay buffer (if buffer is non-empty)
      4. Every eval_frequency iterations:
         - Play new network vs previous best (head-to-head evaluation)
         - Accept as new best if win rate >= acceptance_threshold
      5. Save iteration checkpoint

    Args:
        config: Self-play configuration.
        save_dir: Directory for checkpoints and TensorBoard logs.
        resume_from: Optional checkpoint directory to resume training from.
    """
    root_dir = Path(save_dir)
    root_dir.mkdir(parents=True, exist_ok=True)

    # Initialize or resume
    feature_size = get_feature_size(config.num_players)
    if resume_from is not None:
        network, buffer, start_iter, _ = load_checkpoint(resume_from)
        start_iter += 1
    else:
        network = ValueNetwork(
            input_size=feature_size,
            num_players=config.num_players,
        )
        buffer = ReplayBuffer()
        start_iter = 0

    best_dir = root_dir / "mcts_best"

    for iteration in range(start_iter, start_iter + config.num_iterations):
        # 1. Generate self-play data
        new_data = generate_training_data(
            num_games=config.games_per_iteration,
            num_players=config.num_players,
            mcts_simulations=config.mcts_simulations,
            temperature=config.temperature,
            max_turns=config.max_turns_per_game,
            value_network=network,
            opponent_policy="rule_based",
            seed=iteration,
        )
        for example in new_data.sample(len(new_data)):
            buffer.add(example)

        # 2. Train network
        tb_dir = root_dir / "tensorboard" / f"iter_{iteration}"
        train_stats: TrainingStats | None = None
        if len(buffer) > 0:
            train_stats = train_value_network(
                network,
                buffer,
                config.training_config,
                tensorboard_dir=tb_dir,
            )

        # 3. Evaluate and accept/reject
        if (iteration + 1) % config.eval_frequency == 0:
            if best_dir.exists():
                # Head-to-head: new network (player 0) vs best (player 1)
                old_network, _, _, _ = load_checkpoint(best_dir)
                wins = 0
                for g in range(config.eval_games):
                    winner = _run_game_with_networks(
                        network,
                        old_network,
                        config,
                        seed=10000 + iteration * 100 + g,
                    )
                    if winner == 0:
                        wins += 1
                win_rate = wins / config.eval_games

                if win_rate >= config.acceptance_threshold:
                    save_checkpoint(
                        best_dir,
                        network,
                        buffer,
                        iteration,
                        {"win_rate_vs_old": win_rate, "iteration": iteration},
                    )
            else:
                # First evaluation: accept immediately as baseline
                save_checkpoint(
                    best_dir,
                    network,
                    buffer,
                    iteration,
                    {"win_rate_vs_old": 1.0, "iteration": iteration},
                )

        # 4. Save iteration checkpoint
        epoch_losses: list[tuple[float, float, float]] = (
            train_stats.epoch_losses if train_stats is not None else []
        )
        save_checkpoint(
            root_dir / f"mcts_iter_{iteration}",
            network,
            buffer,
            iteration,
            {"epoch_losses": epoch_losses},
        )
