"""Self-play training for Monopoly using PettingZoo's native API.

This module provides proper multi-agent self-play training where all agents
share the same policy network. The key insight is that in self-play, we want
the agent to learn from playing against itself, so all players use identical
(or near-identical) policies.

Architecture:
- Uses MonopolyEnv (PettingZoo AEC) directly
- Wraps it as a Gymnasium env for SB3 compatibility
- All opponent actions come from the same policy (true self-play)
- Supports curriculum: start vs random/rule-based, then self-play

Usage:
    from training.pettingzoo_selfplay import SelfPlayTrainer, SelfPlayConfig

    config = SelfPlayConfig(
        total_timesteps=1_000_000,
        opponent_type="self",  # or "random", "rule_based"
    )
    trainer = SelfPlayTrainer(config)
    trainer.train()
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from monopoly_gym.single_agent_env import SingleAgentMonopolyEnv


@dataclass
class SelfPlayConfig:
    """Configuration for self-play training."""

    # Training
    total_timesteps: int = 1_000_000
    num_envs: int = 16
    learning_rate: float = 3e-4
    lr_schedule: str = "linear"  # "linear" or "constant"
    n_steps: int = 2048
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr_min: float = 0.0  # Minimum LR floor for linear schedule
    policy_kwargs: dict[str, Any] | None = None  # Custom network architecture

    # Environment
    num_players: int = 4
    max_turns: int = 1000
    reward_type: str = "sparse"  # "sparse" or "dense"
    terminal_win_reward: float = 1.0  # Fixed foundation terminal win reward
    terminal_loss_reward: float = -1.0  # Fixed foundation elimination reward
    worth_scale: float | None = None  # Deprecated shaping configuration

    # Self-play
    opponent_type: str = "self"  # "self", "random", "rule_based", "mixed"
    update_opponent_freq: int = 10_000  # Update opponent policy every N steps
    opponent_pool: list[str] | None = None  # Pool of opponent types to sample from
    opponent_weights: list[float] | None = None  # Sampling weights (uniform if None)

    # Evaluation
    eval_freq: int = 25_000
    eval_episodes: int = 200

    # Checkpointing
    save_freq: int = 50_000
    save_dir: str = "models/selfplay"
    checkpoint_min_win_rate: float = 0.0  # Min win rate vs random to save (0=disabled)
    collapse_detection: bool = True  # Stop training if win rate collapses to 0%
    load_model: str | None = None  # Path to pre-trained model to load

    # Vectorization
    use_subproc: bool = True  # Use SubprocVecEnv for true parallelism (auto-disabled for self-play)

    # Misc
    seed: int = 42
    verbose: bool = True
    diagnostic_logging: bool = True  # Enable diagnostic callbacks


@dataclass
class TrainingMetrics:
    """Metrics tracked during training."""

    total_steps: int = 0
    games_completed: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_game_length: int = 0

    # Evaluation results
    eval_results: list[dict[str, float]] = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        total = self.wins + self.losses + self.draws
        return self.wins / total if total > 0 else 0.0

    @property
    def avg_game_length(self) -> float:
        return self.total_game_length / self.games_completed if self.games_completed > 0 else 0.0




class SelfPlayEnv(SingleAgentMonopolyEnv):
    """Compatibility configuration for the shared learner adapter."""

    def __init__(
        self,
        num_players=4,
        max_turns=1000,
        opponent_type="random",
        reward_type="sparse",
        terminal_win_reward=1.0,
        terminal_loss_reward=-1.0,
        worth_scale=None,
        render_mode=None,
        learner_seat=0,
        seed=None,
    ):
        if (terminal_win_reward, terminal_loss_reward, worth_scale) != (1.0, -1.0, None):
            raise ValueError("foundation-v1 fixes terminal rewards at +1/-1; shaping is deferred")
        super().__init__(
            num_players=num_players,
            max_turns=max_turns,
            opponent_type=opponent_type,
            reward_type=reward_type,
            render_mode=render_mode,
            learner_seat=learner_seat,
            seed=seed,
        )


class SelfPlayTrainer:
    """Trainer for self-play Monopoly using MaskablePPO."""

    def __init__(self, config: SelfPlayConfig):
        self.config = config
        self.metrics = TrainingMetrics()
        self._model = None
        self._vec_env = None

        # Set up opponent pool for mixed training
        if config.opponent_pool is not None and len(config.opponent_pool) > 0:
            self._opponent_pool = config.opponent_pool
            self._opponent_weights = config.opponent_weights
            if config.verbose:
                print(f"Mixed opponent training: {self._opponent_pool}")
                if self._opponent_weights:
                    print(f"  Weights: {self._opponent_weights}")
                else:
                    print("  Weights: uniform")
        else:
            # Single opponent (backward compatible)
            self._opponent_pool = [config.opponent_type]
            self._opponent_weights = None
            if config.verbose:
                print(f"Single opponent training: {config.opponent_type}")

    def train(self) -> Any:
        """Run training and return the trained model."""
        try:
            from sb3_contrib import MaskablePPO
            from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        except ImportError as e:
            raise ImportError(
                f"Training requires sb3-contrib: {e}\nInstall with: uv add sb3-contrib"
            )

        config = self.config
        save_path = Path(config.save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if config.verbose:
            print(f"Creating {config.num_envs} parallel environments...")
            # Show opponent info (pool or single type)
            if config.opponent_pool is not None and len(config.opponent_pool) > 0:
                pool_str = ", ".join(config.opponent_pool)
                if len(config.opponent_pool) > 1:
                    if config.opponent_weights:
                        weights_str = ", ".join(f"{w:.1f}" for w in config.opponent_weights)
                        print(f"Opponent pool: [{pool_str}] (weights: [{weights_str}])")
                    else:
                        print(f"Opponent pool: [{pool_str}] (uniform weights)")
                else:
                    print(f"Opponent type: {pool_str}")
            else:
                print(f"Opponent type: {config.opponent_type}")
            print(f"Reward type: {config.reward_type}")
            print(f"Players: {config.num_players}, Max turns: {config.max_turns}")
            print(
                f"n_steps: {config.n_steps}, batch_size: {config.batch_size}, "
                f"n_epochs: {config.n_epochs}"
            )
            print(
                f"ent_coef: {config.ent_coef}, lr: {config.learning_rate}, "
                f"lr_schedule: {config.lr_schedule}"
            )
            if config.lr_min > 0:
                print(f"lr_min: {config.lr_min}")
            if config.vf_coef != 0.5:
                print(f"vf_coef: {config.vf_coef}")
            if config.gamma != 0.99:
                print(f"gamma: {config.gamma}")
            if config.gae_lambda != 0.95:
                print(f"gae_lambda: {config.gae_lambda}")
            if config.policy_kwargs is not None:
                print(f"policy_kwargs: {config.policy_kwargs}")
            if config.worth_scale is not None:
                print(f"worth_scale: {config.worth_scale}")
            if config.checkpoint_min_win_rate > 0:
                print(f"Checkpoint min win rate: {config.checkpoint_min_win_rate:.0%}")

        # Create vectorized environment
        def make_env(rank: int) -> Callable[[], SelfPlayEnv]:
            """Create a single environment instance.

            Args:
                rank: Environment index (for seeding)

            Returns:
                Callable that creates and returns a SelfPlayEnv
            """

            def _init() -> SelfPlayEnv:
                worker_stream = np.random.SeedSequence([config.seed, rank, 314159])
                choice_seed, episode_seed = worker_stream.spawn(2)
                opponent_rng = np.random.default_rng(choice_seed)
                # Sample opponent type from pool
                if len(self._opponent_pool) > 1:
                    opponent_type = opponent_rng.choice(
                        self._opponent_pool,
                        p=self._opponent_weights,  # None = uniform
                    )
                else:
                    opponent_type = self._opponent_pool[0]

                env = SelfPlayEnv(
                    num_players=config.num_players,
                    max_turns=config.max_turns,
                    opponent_type=opponent_type,  # Sampled!
                    reward_type=config.reward_type,
                    terminal_win_reward=config.terminal_win_reward,
                    terminal_loss_reward=config.terminal_loss_reward,
                    worth_scale=config.worth_scale,
                )
                env.reset(seed=int(episode_seed.generate_state(1)[0]))
                return env

            return _init

        # Use SubprocVecEnv for true parallelism when possible.
        # Self-play mode needs direct env access for policy updates, so it
        # must use DummyVecEnv. All other modes benefit from SubprocVecEnv.
        use_subproc = config.use_subproc and config.num_envs > 1 and config.opponent_type != "self"
        env_fns = [make_env(i) for i in range(config.num_envs)]
        if use_subproc:
            self._vec_env = SubprocVecEnv(env_fns)
            if config.verbose:
                print(f"Using SubprocVecEnv ({config.num_envs} processes)")
        else:
            self._vec_env = DummyVecEnv(env_fns)
            if config.verbose:
                reason = "self-play" if config.opponent_type == "self" else "single env or disabled"
                print(f"Using DummyVecEnv ({reason})")

        # Resolve learning rate schedule
        if config.lr_schedule == "linear":
            _base_lr = config.learning_rate
            _lr_min = config.lr_min

            def lr_schedule_fn(progress_remaining: float) -> float:
                return max(progress_remaining * _base_lr, _lr_min)

            lr_value: float | Callable[[float], float] = lr_schedule_fn
        elif config.lr_schedule == "cosine":
            _base_lr = config.learning_rate
            _lr_min = config.lr_min
            import math

            def lr_schedule_fn_cosine(progress_remaining: float) -> float:
                return _lr_min + 0.5 * (_base_lr - _lr_min) * (
                    1 + math.cos(math.pi * (1 - progress_remaining))
                )

            lr_value = lr_schedule_fn_cosine
        else:
            lr_value = config.learning_rate

        if config.verbose:
            print("Creating MaskablePPO model...")

        self._model = MaskablePPO(
            "MlpPolicy",
            self._vec_env,
            learning_rate=lr_value,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            policy_kwargs=config.policy_kwargs,
            verbose=0,
            tensorboard_log=str(save_path / "tensorboard"),
            seed=config.seed,
        )

        # Load pre-trained model if specified
        if config.load_model:
            self._load_pretrained(Path(config.load_model))

        # Create main self-play callback
        main_callback = SelfPlayCallback(
            trainer=self,
            eval_freq=config.eval_freq,
            eval_episodes=config.eval_episodes,
            save_freq=config.save_freq,
            save_path=save_path,
            update_opponent_freq=config.update_opponent_freq,
            verbose=config.verbose,
        )

        # Create callback list (using SB3's CallbackList for proper lifecycle)
        from stable_baselines3.common.callbacks import CallbackList

        callbacks: list[Any] = [CallableCallbackAdapter(main_callback)]

        # Add diagnostic logging if enabled
        if config.diagnostic_logging:
            from training.callbacks import DiagnosticCallback

            callbacks.append(
                DiagnosticCallback(
                    log_freq=1000,  # Log every 1000 steps
                    verbose=1 if config.verbose else 0,
                )
            )
            if config.verbose:
                print("Diagnostic logging enabled (every 1000 steps)")

        callback = CallbackList(callbacks)

        if config.verbose:
            print(f"\nStarting training for {config.total_timesteps:,} timesteps...")
            print(f"TensorBoard: tensorboard --logdir {save_path / 'tensorboard'}")
            print()

        start_time = time.time()
        self._model.learn(
            total_timesteps=config.total_timesteps,
            callback=callback,
            progress_bar=config.verbose,
        )
        training_time = time.time() - start_time

        # Save final model
        final_path = save_path / "final_model"
        self._model.save(str(final_path))

        if config.verbose:
            print(f"\n{'=' * 60}")
            print("TRAINING COMPLETE")
            print(f"{'=' * 60}")
            print(f"Training time: {training_time / 60:.1f} minutes")
            print(f"Final model: {final_path}")
            print(f"Games completed: {self.metrics.games_completed}")
            print(f"Win rate: {self.metrics.win_rate:.1%}")
            if self.metrics.eval_results:
                last_eval = self.metrics.eval_results[-1]
                print(f"Last eval vs random: {last_eval.get('vs_random', 0):.1%}")
                print(f"Last eval vs rule_based: {last_eval.get('vs_rule_based', 0):.1%}")

        return self._model

    def update_opponent_policies(self) -> None:
        """Update opponent policies to use current model (for self-play)."""
        if self._model is None or self.config.opponent_type != "self":
            return

        def policy_fn(obs: NDArray, mask: NDArray) -> int:
            """Policy function that uses the current model."""
            action, _ = self._model.predict(
                obs.reshape(1, -1),
                action_masks=mask.reshape(1, -1),
                deterministic=False,
            )
            return int(action[0])

        # Update policy in all environments
        for env in self._vec_env.envs:
            env.set_opponent_policy(policy_fn)

    def _load_pretrained(self, path: Path) -> None:
        from sb3_contrib import MaskablePPO

        self._model = MaskablePPO.load(str(path), env=self._vec_env)

    def evaluate(self, opponent_type: str, n_episodes: int = 200) -> float:
        from evaluation.runner import evaluate_policy, summarize

        if self._model is None:
            raise RuntimeError("No policy to evaluate")
        rows = evaluate_policy(
            self._model,
            opponent_type,
            n_episodes,
            self.config.num_players,
            self.config.max_turns,
            self.config.seed + 11000000,
        )
        report = summarize(rows)
        if report["counts"].get("error", 0) or report["counts"].get("stalled", 0):
            raise RuntimeError(f"Invalid evaluation: {report}")
        return float(report["win_rate"])


class SelfPlayCallback:
    """Callback for self-play training with evaluation and opponent updates.

    This callback is a plain callable (not a BaseCallback) because it needs
    to access the trainer object directly, which SB3's callback system doesn't
    support natively. It will be wrapped in a BaseCallbackAdapter.
    """

    def __init__(
        self,
        trainer: SelfPlayTrainer,
        eval_freq: int,
        eval_episodes: int,
        save_freq: int,
        save_path: Path,
        update_opponent_freq: int,
        verbose: bool,
    ):
        self.trainer = trainer
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        self.save_freq = save_freq
        self.save_path = save_path
        self.update_opponent_freq = update_opponent_freq
        self.verbose = verbose

        self.num_timesteps = 0
        self.last_eval = 0
        self.last_save = 0
        self.last_opponent_update = 0
        self.start_time = time.time()
        self.consecutive_zero_evals: int = 0

    def __call__(self, locals_dict: dict, globals_dict: dict) -> bool:
        """Called at each training step."""
        # Get model from locals
        model = locals_dict.get("self")
        if model is None:
            return True

        self.num_timesteps = model.num_timesteps

        # Update opponent policies for self-play
        if (
            self.num_timesteps - self.last_opponent_update >= self.update_opponent_freq
            and self.trainer.config.opponent_type == "self"
        ):
            self.trainer.update_opponent_policies()
            self.last_opponent_update = self.num_timesteps
            if self.verbose:
                print(f"[{self.num_timesteps:,}] Updated opponent policies")

        # Evaluation (less frequent to speed up training)
        if self.num_timesteps - self.last_eval >= self.eval_freq:
            self._evaluate()
            self.last_eval = self.num_timesteps

            # Collapse detection: check if win rate has dropped to 0%
            if self.trainer.metrics.eval_results:
                last_eval = self.trainer.metrics.eval_results[-1]
                vs_random = last_eval.get("vs_random", 0.0)
                if vs_random == 0.0 and self.num_timesteps >= 100_000:
                    self.consecutive_zero_evals += 1
                else:
                    self.consecutive_zero_evals = 0

                if self.consecutive_zero_evals >= 3 and self.trainer.config.collapse_detection:
                    print(f"\n{'!' * 60}")
                    print(f"COLLAPSE DETECTED at step {self.num_timesteps:,}")
                    print(
                        f"Win rate vs random has been 0% for "
                        f"{self.consecutive_zero_evals} consecutive evals"
                    )
                    best_path = self.save_path / "best_model.zip"
                    if best_path.exists():
                        print(f"Loading best model from {best_path}")
                        self.trainer._model = type(self.trainer._model).load(
                            str(self.save_path / "best_model"),
                            env=self.trainer._vec_env,
                        )
                    print("Stopping training early.")
                    print(f"{'!' * 60}\n")
                    return False

        # Save checkpoint
        if self.num_timesteps - self.last_save >= self.save_freq:
            self._save_checkpoint()
            self.last_save = self.num_timesteps

        return True

    def _evaluate(self) -> None:
        """Run evaluation against different opponent types."""
        results = {}

        eval_eps = self.eval_episodes

        for opp_type in ["random", "rule_based"]:
            win_rate = self.trainer.evaluate(opp_type, eval_eps)
            results[f"vs_{opp_type}"] = win_rate

        self.trainer.metrics.eval_results.append(results)

        if self.verbose:
            import sys

            elapsed = time.time() - self.start_time
            steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0
            print(f"\n{'=' * 60}")
            print(
                f"Step {self.num_timesteps:,} | {elapsed / 60:.1f}min | {steps_per_sec:.0f} steps/s"
            )
            print(f"Win rate vs random: {results['vs_random']:.1%}")
            print(f"Win rate vs rule_based: {results['vs_rule_based']:.1%}")
            print(f"{'=' * 60}\n")
            sys.stdout.flush()

        # Log to tensorboard
        if self.trainer._model is not None:
            for key, value in results.items():
                self.trainer._model.logger.record(f"eval/{key}", value)
            self.trainer._model.logger.dump(self.num_timesteps)

    def _save_checkpoint(self) -> None:
        """Save a checkpoint, gated on minimum win rate if configured."""
        if self.trainer._model is None:
            return

        min_wr = self.trainer.config.checkpoint_min_win_rate
        if min_wr > 0 and self.trainer.metrics.eval_results:
            last_eval = self.trainer.metrics.eval_results[-1]
            vs_random = last_eval.get("vs_random", 0.0)
            if vs_random < min_wr:
                if self.verbose:
                    print(
                        f"[{self.num_timesteps:,}] Checkpoint skipped "
                        f"(win rate {vs_random:.0%} < {min_wr:.0%})"
                    )
                return

        checkpoint_path = self.save_path / f"checkpoint_{self.num_timesteps}"
        self.trainer._model.save(str(checkpoint_path))

        # Track best model
        if self.trainer.metrics.eval_results:
            last_eval = self.trainer.metrics.eval_results[-1]
            vs_random = last_eval.get("vs_random", 0.0)
            if not hasattr(self, "_best_win_rate"):
                self._best_win_rate = 0.0
            if vs_random > self._best_win_rate:
                self._best_win_rate = vs_random
                best_path = self.save_path / "best_model"
                self.trainer._model.save(str(best_path))
                if self.verbose:
                    print(f"[{self.num_timesteps:,}] New best model! ({vs_random:.0%} vs random)")

        if self.verbose:
            print(f"[{self.num_timesteps:,}] Saved checkpoint: {checkpoint_path}")


def make_selfplay_env(
    num_players: int = 4,
    max_turns: int = 1000,
    opponent_type: str = "random",
    reward_type: str = "sparse",
    terminal_win_reward: float = 1.0,
    terminal_loss_reward: float = -1.0,
    worth_scale: float | None = None,
    seed: int | None = None,
) -> SelfPlayEnv:
    """Factory function to create a self-play environment."""
    env = SelfPlayEnv(
        num_players=num_players,
        max_turns=max_turns,
        opponent_type=opponent_type,
        reward_type=reward_type,
        terminal_win_reward=terminal_win_reward,
        terminal_loss_reward=terminal_loss_reward,
        worth_scale=worth_scale,
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


class CallableCallbackAdapter:
    """Adapter that wraps a plain callable callback into SB3's BaseCallback protocol.

    This allows legacy callable callbacks (like SelfPlayCallback) to work with
    SB3's CallbackList, which properly handles the callback lifecycle including
    init_callback, on_training_start, on_step, etc.
    """

    def __init__(self, callable_callback: Any):
        """Initialize with a callable callback.

        Args:
            callable_callback: A callable that takes (locals_dict, globals_dict)
                              and returns bool to continue/stop training.
        """
        self.callback = callable_callback
        self.locals: dict[str, Any] = {}
        self.globals: dict[str, Any] = {}
        self.n_calls = 0
        self.model = None
        self.logger = None

    def init_callback(self, model: Any) -> None:
        """Initialize the callback with the model.

        The callable callback doesn't need model initialization, so we just
        store the reference for potential use.
        """
        self.model = model
        self.logger = model.logger if hasattr(model, "logger") else None

    def on_training_start(self, locals_dict: dict, globals_dict: dict) -> None:
        """Called at the start of training."""
        self.locals = locals_dict
        self.globals = globals_dict

    def on_rollout_start(self) -> None:
        """Called at the start of each rollout."""
        pass

    def on_step(self) -> bool:
        """Called at each training step.

        Returns True to continue training, False to stop.
        """
        self.n_calls += 1
        return self.callback(self.locals, self.globals)

    def on_training_end(self) -> None:
        """Called at the end of training."""
        pass

    def on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        pass

    def update_locals(self, locals_dict: dict[str, Any]) -> None:
        """Update locals dict (called by SB3 during training)."""
        self.locals = locals_dict
