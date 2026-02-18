"""Value and policy network for MCTS.

Provides the ValueNetwork class (PyTorch nn.Module) used for leaf evaluation
and policy guidance during MCTS search. Architecture follows AlphaZero:
shared trunk with separate value and policy heads.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


class ValueNetwork(nn.Module):
    """Combined value and policy network for MCTS.

    Architecture:
        Input features -> SharedTrunk -> ValueHead (per-player values)
                                      -> PolicyHead (action logits)

    The shared trunk learns a representation of the game state.
    The value head predicts the expected outcome for each player.
    The policy head predicts a distribution over actions to guide MCTS.

    Attributes:
        input_size: Dimension of input feature vector.
        num_players: Number of players (for value head output).
        action_size: Number of possible actions (149).
        hidden_size: Hidden layer dimension in the trunk.
        trunk_layers: Number of FC layers in the shared trunk.
    """

    def __init__(
        self,
        input_size: int,
        num_players: int = 4,
        action_size: int = 149,
        hidden_size: int = 256,
        trunk_layers: int = 3,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.num_players = num_players
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.trunk_layers = trunk_layers

        # Build shared trunk: FC layers with ReLU + BatchNorm
        trunk_modules: list[nn.Module] = []
        in_dim = input_size
        for i in range(trunk_layers):
            out_dim = hidden_size if i < trunk_layers - 1 else hidden_size // 2
            trunk_modules.append(nn.Linear(in_dim, out_dim))
            trunk_modules.append(nn.BatchNorm1d(out_dim))
            trunk_modules.append(nn.ReLU())
            in_dim = out_dim

        self.trunk = nn.Sequential(*trunk_modules)
        trunk_out = hidden_size // 2  # 128 with default hidden_size=256

        # Value head: trunk_out -> 64 -> num_players (tanh)
        self.value_head = nn.Sequential(
            nn.Linear(trunk_out, 64),
            nn.ReLU(),
            nn.Linear(64, num_players),
            nn.Tanh(),
        )

        # Policy head: trunk_out -> action_size (raw logits)
        self.policy_head = nn.Sequential(
            nn.Linear(trunk_out, action_size),
        )

    def forward(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input features of shape (batch, input_size).
            action_mask: Optional boolean mask of shape (batch, action_size).
                        True = valid action. Invalid actions get -inf logits.

        Returns:
            Tuple of:
                values: Shape (batch, num_players), tanh-activated in [-1, 1].
                policy_logits: Shape (batch, action_size), masked if mask provided.
        """
        shared = self.trunk(x)

        values: torch.Tensor = self.value_head(shared)
        policy_logits: torch.Tensor = self.policy_head(shared)

        if action_mask is not None:
            # Set invalid action logits to -inf so softmax gives ~0
            policy_logits = policy_logits.masked_fill(~action_mask.bool(), float("-inf"))

        return values, policy_logits

    @torch.no_grad()
    def predict(
        self,
        features: NDArray[np.float32],
        action_mask: NDArray[np.bool_] | None = None,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Numpy-based prediction for use during MCTS search.

        Handles tensor conversion and moves to/from the correct device.
        No gradient computation. Uses eval mode for BatchNorm.

        Args:
            features: Input features of shape (input_size,) or (batch, input_size).
            action_mask: Optional mask of shape (action_size,) or (batch, action_size).

        Returns:
            Tuple of (values, policy_probs) as numpy arrays.
            values: shape (num_players,) or (batch, num_players).
            policy_probs: shape (action_size,) or (batch, action_size).
        """
        was_training = self.training
        self.eval()

        device = next(self.parameters()).device
        single = features.ndim == 1
        if single:
            features = features[np.newaxis, :]  # (1, input_size)

        x = torch.from_numpy(features).to(device)

        mask_tensor: torch.Tensor | None = None
        if action_mask is not None:
            if action_mask.ndim == 1:
                action_mask = action_mask[np.newaxis, :]
            mask_tensor = torch.from_numpy(action_mask).to(device)

        values, policy_logits = self.forward(x, mask_tensor)

        # Convert policy logits to probabilities
        policy_probs = torch.nn.functional.softmax(policy_logits, dim=-1)

        values_np = values.cpu().numpy().astype(np.float32)
        policy_np = policy_probs.cpu().numpy().astype(np.float32)

        if single:
            values_np = values_np[0]
            policy_np = policy_np[0]

        if was_training:
            self.train()

        return values_np, policy_np

    def save(self, path: str | Path) -> None:
        """Save model weights and config.

        Saves both the state dict and the constructor arguments so the
        model can be fully reconstructed from the checkpoint.

        Args:
            path: File path to save the checkpoint to.
        """
        checkpoint: dict[str, Any] = {
            "state_dict": self.state_dict(),
            "config": {
                "input_size": self.input_size,
                "num_players": self.num_players,
                "action_size": self.action_size,
                "hidden_size": self.hidden_size,
                "trunk_layers": self.trunk_layers,
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path) -> ValueNetwork:
        """Load model from checkpoint.

        Args:
            path: File path to load the checkpoint from.

        Returns:
            A ValueNetwork instance with loaded weights.
        """
        checkpoint = torch.load(path, weights_only=False)
        config = checkpoint["config"]
        network = cls(**config)
        network.load_state_dict(checkpoint["state_dict"])
        return network
