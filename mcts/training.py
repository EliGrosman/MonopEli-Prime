"""Training loop for the MCTS value/policy network.

Provides TrainingConfig, TrainingStats, and train_value_network() for the
AlphaZero-style training loop: MSE value loss + cross-entropy policy loss.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .data import ReplayBuffer
from .network import ValueNetwork


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

    train_ds = TensorDataset(
        features_t[train_idx], policies_t[train_idx], outcomes_t[train_idx]
    )
    val_ds = TensorDataset(
        features_t[val_idx], policies_t[val_idx], outcomes_t[val_idx]
    )

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

    total_loss = (
        config.value_loss_weight * value_loss
        + config.policy_loss_weight * policy_loss
    )

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
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.lr_decay
    )

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
