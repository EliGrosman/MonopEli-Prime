"""Behavioral cloning for Monopoly: supervised pre-training on expert demonstrations.

Trains a MaskablePPO-compatible policy network to imitate expert actions via
cross-entropy loss. The resulting model can be loaded directly by MaskablePPO
for RL fine-tuning (TASK-203).

Usage:
    from training.behavioral_cloning import BCTrainer, BCConfig

    config = BCConfig(data_dir=Path("data/expert/test"), epochs=10)
    trainer = BCTrainer(config)
    model_path = trainer.train()
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium import spaces
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


@dataclass
class BCConfig:
    """Configuration for behavioral cloning training."""

    data_dir: Path = Path("data/expert/test")
    save_dir: Path = Path("models/bc")
    epochs: int = 10
    batch_size: int = 512
    learning_rate: float = 1e-3
    patience: int = 5  # Early stopping patience (epochs without val_loss improvement)
    num_players: int = 4  # Determines observation size
    policy_kwargs: dict[str, Any] | None = None  # Network architecture
    max_samples: int | None = None  # Cap training set size (None = use all)
    filter_winners: bool = False  # Only train on winning players' experiences
    seed: int = 42
    verbose: bool = True


class DummyMonopolyEnv(gym.Env):
    """Minimal env for MaskablePPO initialization (defines obs/action spaces only)."""

    def __init__(self, obs_size: int = 428, action_size: int = 149):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(action_size)

    def action_masks(self) -> NDArray[np.bool_]:
        return np.ones(self.action_space.n, dtype=bool)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {
            "action_mask": self.action_masks()
        }

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        return (
            np.zeros(self.observation_space.shape, dtype=np.float32),
            0.0,
            True,
            False,
            {"action_mask": self.action_masks()},
        )


class ExpertDataset(Dataset):
    """In-memory PyTorch dataset loaded from HDF5 expert demonstrations.

    Preloads data into memory on init (sequential HDF5 read is fast; random
    per-item access on gzip-compressed HDF5 is prohibitively slow). Memory
    usage for the full 7.3M training set is ~13GB; use max_samples to limit.

    Args:
        h5_path: Path to HDF5 file (train.h5 or val.h5).
        max_samples: Optional cap on dataset size (limits memory usage).
        filter_winners: If True, only include experiences from winning players.
    """

    def __init__(
        self,
        h5_path: str | Path,
        max_samples: int | None = None,
        filter_winners: bool = False,
    ):
        h5_path = Path(h5_path)

        with h5py.File(h5_path, "r") as f:
            total = len(f["actions"])
            n = min(total, max_samples) if max_samples else total

            if filter_winners:
                # Read metadata to build winner mask, then load only matching rows
                player_ids = f["player_ids"][:n]
                winners = f["winners"][:n]
                keep = (player_ids == winners) & (winners >= 0)
                indices = np.nonzero(keep)[0]

                self.observations = torch.as_tensor(
                    f["observation"][:n][indices], dtype=torch.float32
                )
                self.actions = torch.as_tensor(
                    f["actions"][:n][indices], dtype=torch.long
                )
                self.masks = torch.as_tensor(
                    f["action_masks"][:n][indices], dtype=torch.bool
                )
            else:
                # Sequential slice read (fast even with gzip compression)
                self.observations = torch.as_tensor(
                    f["observation"][:n], dtype=torch.float32
                )
                self.actions = torch.as_tensor(
                    f["actions"][:n], dtype=torch.long
                )
                self.masks = torch.as_tensor(
                    f["action_masks"][:n], dtype=torch.bool
                )

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.observations[idx], self.actions[idx], self.masks[idx]


class BCTrainer:
    """Behavioral cloning trainer.

    Creates a MaskablePPO model and trains its policy network (mlp_extractor.policy_net
    + action_net) via cross-entropy loss on expert demonstrations. The value network
    is left untrained (will be learned during RL fine-tuning).

    The trained model is saved in SB3 format and can be loaded directly by
    MaskablePPO.load() for fine-tuning.
    """

    def __init__(self, config: BCConfig):
        self.config = config

    def train(self) -> Path:
        """Run behavioral cloning training. Returns path to saved model."""
        config = self.config
        config.save_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if config.verbose:
            print(f"Device: {device}")
            print(f"Data directory: {config.data_dir}")

        # 1. Compute observation size from num_players
        from monopoly_gym.observation import get_flat_observation_size
        from monopoly_gym import GAMEPLAY_ACTION_SPACE_SIZE

        obs_size = get_flat_observation_size(config.num_players)
        action_size = GAMEPLAY_ACTION_SPACE_SIZE

        if config.verbose:
            print(f"Observation size: {obs_size} ({config.num_players} players)")
            print(f"Action size: {action_size}")

        # 2. Create MaskablePPO with target architecture
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        dummy_env = DummyVecEnv([lambda: DummyMonopolyEnv(obs_size, action_size)])
        model = MaskablePPO(
            "MlpPolicy",
            dummy_env,
            policy_kwargs=config.policy_kwargs,
            seed=config.seed,
            verbose=0,
        )
        policy = model.policy.to(device)

        if config.verbose:
            # Count parameters
            pi_params = sum(
                p.numel() for p in policy.mlp_extractor.policy_net.parameters()
            ) + sum(p.numel() for p in policy.action_net.parameters())
            vf_params = sum(
                p.numel() for p in policy.mlp_extractor.value_net.parameters()
            ) + sum(p.numel() for p in policy.value_net.parameters())
            print(f"Policy network parameters: {pi_params:,} (training)")
            print(f"Value network parameters: {vf_params:,} (frozen)")
            print(f"Architecture: {config.policy_kwargs}")

        # 3. Set up optimizer for policy path only
        policy_params = list(policy.mlp_extractor.policy_net.parameters()) + list(
            policy.action_net.parameters()
        )
        optimizer = torch.optim.Adam(policy_params, lr=config.learning_rate)

        # 4. Load datasets
        if config.verbose:
            print(f"\nLoading datasets...")

        train_dataset = ExpertDataset(
            config.data_dir / "train.h5",
            max_samples=config.max_samples,
            filter_winners=config.filter_winners,
        )
        val_dataset = ExpertDataset(
            config.data_dir / "val.h5",
            filter_winners=config.filter_winners,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )

        if config.verbose:
            print(f"Train samples: {len(train_dataset):,}")
            print(f"Val samples: {len(val_dataset):,}")
            print(f"Batches per epoch: {len(train_loader):,}")
            if config.filter_winners:
                print("Filtering: winning players only")

        # 5. TensorBoard writer
        writer = SummaryWriter(log_dir=str(config.save_dir / "tensorboard"))

        # 6. Training loop with early stopping
        best_val_acc = 0.0
        best_val_loss = float("inf")
        patience_counter = 0
        best_epoch = 0

        if config.verbose:
            print(f"\nStarting BC training for {config.epochs} epochs...")
            print(f"Learning rate: {config.learning_rate}")
            print(f"Batch size: {config.batch_size}")
            print(f"Early stopping patience: {config.patience}")
            print()

        for epoch in range(config.epochs):
            epoch_start = time.time()

            # Train
            train_loss, train_acc, train_top5 = self._train_epoch(
                policy, train_loader, optimizer, device
            )

            # Validate
            val_loss, val_acc, val_top5 = self._validate(policy, val_loader, device)

            epoch_time = time.time() - epoch_start

            # Log to TensorBoard
            writer.add_scalar("bc/train_loss", train_loss, epoch)
            writer.add_scalar("bc/val_loss", val_loss, epoch)
            writer.add_scalar("bc/train_accuracy", train_acc, epoch)
            writer.add_scalar("bc/val_accuracy", val_acc, epoch)
            writer.add_scalar("bc/train_top5_accuracy", train_top5, epoch)
            writer.add_scalar("bc/val_top5_accuracy", val_top5, epoch)
            writer.add_scalar("bc/learning_rate", config.learning_rate, epoch)

            if config.verbose:
                print(
                    f"Epoch {epoch + 1}/{config.epochs} ({epoch_time:.1f}s) | "
                    f"Train: loss={train_loss:.4f} acc={train_acc:.1%} top5={train_top5:.1%} | "
                    f"Val: loss={val_loss:.4f} acc={val_acc:.1%} top5={val_top5:.1%}"
                )

            # Early stopping on val_loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch + 1
                patience_counter = 0

                # Save best model (move policy back to CPU for SB3 save)
                policy.to("cpu")
                model.save(str(config.save_dir / "bc_model"))
                policy.to(device)

                if config.verbose:
                    print(f"  -> New best model saved (val_acc={val_acc:.1%})")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    if config.verbose:
                        print(
                            f"\nEarly stopping at epoch {epoch + 1} "
                            f"(no improvement for {config.patience} epochs)"
                        )
                    break

        writer.close()
        dummy_env.close()

        model_path = config.save_dir / "bc_model"

        if config.verbose:
            print(f"\n{'=' * 50}")
            print("BC TRAINING COMPLETE")
            print(f"{'=' * 50}")
            print(f"Best epoch: {best_epoch}/{config.epochs}")
            print(f"Best val accuracy: {best_val_acc:.1%}")
            print(f"Best val loss: {best_val_loss:.4f}")
            print(f"Model saved to: {model_path}")

        return model_path

    def _forward_policy(
        self, policy: Any, obs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through policy network with action masking.

        Returns masked logits (invalid actions set to -1e8).
        """
        features = policy.extract_features(obs, policy.pi_features_extractor)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        logits = policy.action_net(latent_pi)

        # Mask invalid actions
        logits = logits.clone()
        logits[~mask] = -1e8

        return logits

    def _train_epoch(
        self,
        policy: Any,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> tuple[float, float, float]:
        """Train for one epoch. Returns (loss, top1_acc, top5_acc)."""
        policy.set_training_mode(True)

        total_loss = 0.0
        total_correct = 0
        total_top5_correct = 0
        total_samples = 0

        for obs, actions, masks in loader:
            obs = obs.to(device)
            actions = actions.to(device)
            masks = masks.to(device)

            logits = self._forward_policy(policy, obs, masks)
            loss = F.cross_entropy(logits, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            batch_size = actions.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == actions).sum().item()
            _, top5 = logits.topk(5, dim=1)
            total_top5_correct += (top5 == actions.unsqueeze(1)).any(dim=1).sum().item()
            total_samples += batch_size

        return (
            total_loss / total_samples,
            total_correct / total_samples,
            total_top5_correct / total_samples,
        )

    @torch.no_grad()
    def _validate(
        self,
        policy: Any,
        loader: DataLoader,
        device: torch.device,
    ) -> tuple[float, float, float]:
        """Validate on dataset. Returns (loss, top1_acc, top5_acc)."""
        policy.set_training_mode(False)

        total_loss = 0.0
        total_correct = 0
        total_top5_correct = 0
        total_samples = 0

        for obs, actions, masks in loader:
            obs = obs.to(device)
            actions = actions.to(device)
            masks = masks.to(device)

            logits = self._forward_policy(policy, obs, masks)
            loss = F.cross_entropy(logits, actions)

            batch_size = actions.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == actions).sum().item()
            _, top5 = logits.topk(5, dim=1)
            total_top5_correct += (top5 == actions.unsqueeze(1)).any(dim=1).sum().item()
            total_samples += batch_size

        return (
            total_loss / total_samples,
            total_correct / total_samples,
            total_top5_correct / total_samples,
        )
