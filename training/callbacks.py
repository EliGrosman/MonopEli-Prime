"""Custom callbacks for diagnostic logging during training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


class SeparateValueLRCallback(BaseCallback):
    """Sets a separate (higher) learning rate for the value function.

    After model initialization, splits the optimizer into two parameter groups:
    - Policy parameters: use the base learning rate
    - Value function parameters: use vf_lr_multiplier * base_lr

    Args:
        vf_lr_multiplier: Multiplier for value function LR (e.g., 3.0 means 3x policy LR)
    """

    def __init__(self, vf_lr_multiplier: float = 3.0, verbose: int = 0):
        super().__init__(verbose)
        self.vf_lr_multiplier = vf_lr_multiplier
        self._initialized = False

    def _on_step(self) -> bool:
        if not self._initialized:
            self._setup_separate_lr()
            self._initialized = True
        return True

    def _setup_separate_lr(self) -> None:
        """Rebuild the optimizer with separate parameter groups."""
        policy = self.model.policy
        optimizer = policy.optimizer

        # Get current base learning rate
        base_lr = optimizer.param_groups[0]["lr"]
        vf_lr = base_lr * self.vf_lr_multiplier

        # Separate parameters into policy and value groups
        policy_params = []
        value_params = []

        for name, param in policy.named_parameters():
            if "value" in name or "vf" in name:
                value_params.append(param)
            else:
                policy_params.append(param)

        # Create new optimizer with separate groups
        new_optimizer = torch.optim.Adam([
            {"params": policy_params, "lr": base_lr},
            {"params": value_params, "lr": vf_lr},
        ], eps=1e-5)

        # Replace the optimizer
        policy.optimizer = new_optimizer

        if self.verbose >= 1:
            print(f"[SeparateValueLR] Policy LR: {base_lr:.2e}, "
                  f"Value LR: {vf_lr:.2e} ({self.vf_lr_multiplier}x)")
            print(f"  Policy params: {len(policy_params)}, Value params: {len(value_params)}")

    def _on_rollout_start(self) -> None:
        """Update value LR when base LR changes (for LR schedules)."""
        if not self._initialized:
            return

        optimizer = self.model.policy.optimizer
        if len(optimizer.param_groups) >= 2:
            base_lr = optimizer.param_groups[0]["lr"]
            optimizer.param_groups[1]["lr"] = base_lr * self.vf_lr_multiplier


class DiagnosticCallback(BaseCallback):
    """
    Logs detailed diagnostics for debugging training instability.

    Logs every `log_freq` steps:
    - Value function statistics (mean, std, min, max)
    - Gradient norms for each network layer
    - Policy entropy (from last rollout buffer)
    - Action distribution statistics
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        """
        Args:
            log_freq: How often to log diagnostics (in env steps)
            verbose: Verbosity level (0 = no prints, 1 = info, 2 = debug)
        """
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        """
        Called after every step in the environment.
        Returns True to continue training, False to stop.
        """
        # Only log every log_freq steps
        if self.n_calls % self.log_freq != 0:
            return True

        # 1. Value Function Statistics
        self._log_value_stats()

        # 2. Gradient Norms
        self._log_gradient_norms()

        # 3. Policy Entropy (from rollout buffer)
        self._log_policy_entropy()

        # 4. Action Distribution
        self._log_action_distribution()

        return True  # Continue training

    def _log_value_stats(self) -> None:
        """Log value function predictions (mean, std, min, max)."""
        try:
            # Get last observations from rollout buffer
            if self.model.rollout_buffer.buffer_size > 0:
                # Get observations from buffer
                # Handle different SB3 versions - try different attributes
                if hasattr(self.model.rollout_buffer, 'observations'):
                    obs = self.model.rollout_buffer.observations
                elif hasattr(self.model.rollout_buffer, 'obs'):
                    obs = self.model.rollout_buffer.obs
                else:
                    if self.verbose >= 1:
                        print("[DiagnosticCallback] Could not find observations in rollout buffer")
                    return

                # Get current position in buffer
                pos = getattr(self.model.rollout_buffer, 'pos', 0)
                if pos == 0:
                    # Buffer not yet filled, skip
                    return

                # Get valid observations (up to current position)
                if isinstance(obs, dict):
                    # Dict observations - flatten and process
                    obs_arrays = []
                    for key in obs.keys():
                        obs_arr = obs[key]
                        # Shape is typically [n_steps, n_envs, ...]
                        if len(obs_arr.shape) >= 2:
                            # Take up to current position
                            valid_obs = obs_arr[:pos]
                            obs_arrays.append(valid_obs.reshape(-1))
                    # Stack all observations
                    if obs_arrays:
                        obs_flat = np.concatenate(obs_arrays)
                    else:
                        return
                else:
                    # numpy array observations
                    # Shape is typically [n_steps, n_envs, obs_dim]
                    if len(obs.shape) >= 2:
                        obs_flat = obs[:pos].reshape(-1, obs.shape[-1])
                    else:
                        obs_flat = obs.reshape(-1)

                # Convert to tensor and move to device
                obs_tensor = torch.as_tensor(obs_flat, dtype=torch.float32).to(self.model.device)

                # Predict values (no gradient needed)
                with torch.no_grad():
                    values = self.model.policy.predict_values(obs_tensor)

                # Log statistics
                values_np = values.cpu().numpy().flatten()
                self.logger.record("diagnostics/value_mean", float(np.mean(values_np)))
                self.logger.record("diagnostics/value_std", float(np.std(values_np)))
                self.logger.record("diagnostics/value_min", float(np.min(values_np)))
                self.logger.record("diagnostics/value_max", float(np.max(values_np)))

                if self.verbose >= 2:
                    print(f"[DiagnosticCallback] Value stats: "
                          f"mean={np.mean(values_np):.4f}, std={np.std(values_np):.4f}")

        except Exception as e:
            if self.verbose >= 1:
                print(f"[DiagnosticCallback] Error logging value stats: {e}")

    def _log_gradient_norms(self) -> None:
        """Log gradient norms for each network layer."""
        try:
            # Only log if gradients exist (after backward pass)
            total_norm = 0.0
            param_count = 0

            for name, param in self.model.policy.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2).item()  # L2 norm
                    # Sanitize name for TensorBoard (replace dots with slashes)
                    sanitized_name = name.replace(".", "/")
                    self.logger.record(f"gradients/{sanitized_name}_norm", param_norm)
                    total_norm += param_norm ** 2
                    param_count += 1

            if param_count > 0:
                total_norm = total_norm ** 0.5  # sqrt of sum of squares
                self.logger.record("gradients/total_norm", total_norm)

                if self.verbose >= 2:
                    print(f"[DiagnosticCallback] Total gradient norm: {total_norm:.4f}")

        except Exception as e:
            if self.verbose >= 1:
                print(f"[DiagnosticCallback] Error logging gradient norms: {e}")

    def _log_policy_entropy(self) -> None:
        """Log policy entropy from rollout buffer."""
        try:
            rollout_buffer = self.model.rollout_buffer

            # Try different attribute names for log_probs
            log_probs = None
            if hasattr(rollout_buffer, 'log_probs'):
                log_probs = rollout_buffer.log_probs
            elif hasattr(rollout_buffer, 'logp'):
                log_probs = rollout_buffer.logp

            if log_probs is not None:
                pos = getattr(rollout_buffer, 'pos', 0)
                if pos > 0:
                    # Get valid log probabilities
                    valid_log_probs = log_probs[:pos]

                    # Entropy ≈ -mean(log_prob) for the sampled actions
                    # This is an approximation of the policy's entropy at visited states
                    entropy_approx = -valid_log_probs.mean().item()

                    self.logger.record("diagnostics/policy_entropy_approx", entropy_approx)

                    if self.verbose >= 2:
                        print(f"[DiagnosticCallback] Policy entropy (approx): {entropy_approx:.4f}")

        except Exception as e:
            if self.verbose >= 1:
                print(f"[DiagnosticCallback] Error logging policy entropy: {e}")

    def _log_action_distribution(self) -> None:
        """Log action distribution statistics."""
        try:
            rollout_buffer = self.model.rollout_buffer

            # Try different attribute names for actions
            actions = None
            if hasattr(rollout_buffer, 'actions'):
                actions = rollout_buffer.actions
            elif hasattr(rollout_buffer, 'acts'):
                actions = rollout_buffer.acts

            if actions is not None:
                pos = getattr(rollout_buffer, 'pos', 0)
                if pos > 0:
                    # Get valid actions (shape: [n_steps, n_envs])
                    valid_actions = actions[:pos]
                    actions_flat = valid_actions.flatten()

                    # Log statistics
                    unique_actions = len(np.unique(actions_flat))
                    self.logger.record("diagnostics/unique_actions_per_rollout", unique_actions)

                    # Most common action
                    action_counts = np.bincount(actions_flat.astype(int), minlength=149)
                    most_common_action = int(np.argmax(action_counts))
                    most_common_freq = float(action_counts[most_common_action] / len(actions_flat))

                    self.logger.record("diagnostics/most_common_action", most_common_action)
                    self.logger.record("diagnostics/most_common_action_freq", most_common_freq)

                    if self.verbose >= 2:
                        print(f"[DiagnosticCallback] Unique actions: {unique_actions}, "
                              f"Most common: {most_common_action} ({most_common_freq:.2%})")

        except Exception as e:
            if self.verbose >= 1:
                print(f"[DiagnosticCallback] Error logging action distribution: {e}")
