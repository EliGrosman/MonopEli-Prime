"""Phasic Policy Gradient (PPG) callback for Monopoly training.

PPG alternates between policy phases (normal PPO) and auxiliary phases
(extra value function training on replay data). This gives the value
function more training iterations than the policy, addressing value
collapse in multi-agent settings.

References:
    Cobbe et al., "Phasic Policy Gradient" (2021)
    https://arxiv.org/abs/2009.04416
"""

from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback


class PPGCallback(BaseCallback):
    """Callback that implements PPG auxiliary phase on top of MaskablePPO.

    This callback counts rollout collections. Every n_pi rollouts, it:
    1. Creates a frozen snapshot of the current policy
    2. Runs n_aux_epochs of value-only training on buffered rollouts
    3. Adds a KL penalty to prevent the policy from drifting

    This is an approximation of full PPG — it reuses PPO's rollout buffer
    rather than maintaining a separate replay buffer.
    """

    def __init__(
        self,
        n_pi: int = 32,
        n_aux_epochs: int = 6,
        beta_clone: float = 1.0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.n_pi = n_pi
        self.n_aux_epochs = n_aux_epochs
        self.beta_clone = beta_clone
        self._rollout_count = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout collection."""
        self._rollout_count += 1

        if self._rollout_count % self.n_pi == 0:
            self._run_auxiliary_phase()

    def _run_auxiliary_phase(self) -> None:
        """Run auxiliary value training phase.

        Steps:
        1. Deep-copy the policy into a frozen snapshot (no in-place mutations)
        2. Train value function for n_aux_epochs on current rollout data
        3. Add KL penalty between snapshot and evolving policy to prevent drift
        """
        if self.verbose >= 1:
            print(f"[PPG] Auxiliary phase at rollout {self._rollout_count}")

        policy = self.model.policy
        rollout_buffer = self.model.rollout_buffer

        # 1. Create a frozen copy of the policy for old-logit computation.
        #    This avoids in-place state_dict swaps that break autograd.
        old_policy = copy.deepcopy(policy)
        old_policy.eval()
        for param in old_policy.parameters():
            param.requires_grad_(False)

        value_loss_val = 0.0
        kl_loss_val = 0.0

        # 2. Extra value training epochs
        for epoch in range(self.n_aux_epochs):
            for batch in rollout_buffer.get(self.model.batch_size):
                obs = batch.observations
                returns = batch.returns

                if isinstance(obs, dict):
                    obs_tensor: Any = {
                        k: torch.as_tensor(v).to(self.model.device)
                        for k, v in obs.items()
                    }
                else:
                    obs_tensor = torch.as_tensor(obs).to(self.model.device)

                returns_tensor = torch.as_tensor(returns).to(self.model.device)

                # Value loss (from current, evolving policy)
                values = policy.predict_values(obs_tensor)
                value_loss = F.mse_loss(values.flatten(), returns_tensor.flatten())

                # Old logits from frozen snapshot (no graph needed)
                with torch.no_grad():
                    old_dist = old_policy.get_distribution(obs_tensor)
                    old_logits = old_dist.distribution.logits

                # New logits from current policy (needs gradient)
                new_dist = policy.get_distribution(obs_tensor)
                new_logits = new_dist.distribution.logits

                # KL divergence as preservation penalty
                kl_loss = F.kl_div(
                    F.log_softmax(new_logits, dim=-1),
                    F.softmax(old_logits, dim=-1),
                    reduction="batchmean",
                )

                # Combined loss
                total_loss = value_loss + self.beta_clone * kl_loss

                policy.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), self.model.max_grad_norm
                )
                policy.optimizer.step()

                value_loss_val = value_loss.item()
                kl_loss_val = kl_loss.item()

            if self.verbose >= 1:
                print(
                    f"  Epoch {epoch + 1}/{self.n_aux_epochs}: "
                    f"value_loss={value_loss_val:.6f}, "
                    f"kl_loss={kl_loss_val:.6f}"
                )

        # Free the snapshot
        del old_policy

        # Log auxiliary phase metrics
        self.logger.record("ppg/aux_value_loss", value_loss_val)
        self.logger.record("ppg/aux_kl_loss", kl_loss_val)
        self.logger.record("ppg/rollout_count", self._rollout_count)
