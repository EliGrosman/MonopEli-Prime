"""Curriculum learning for Monopoly RL agents.

This module implements progressive training where agents start against
easier opponents (random) and graduate to harder ones (rule-based, aggressive).

Usage:
    from training import CurriculumTrainer

    trainer = CurriculumTrainer()
    model = trainer.train()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CurriculumStage:
    """A stage in the curriculum.

    Attributes:
        name: Stage name
        opponent_type: Type of opponent for this stage
        timesteps: Training steps for this stage
        win_rate_threshold: Win rate required to advance (0-1)
        eval_games: Number of games for evaluation
    """
    name: str
    opponent_type: str
    timesteps: int
    win_rate_threshold: float
    eval_games: int = 50


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning.

    Attributes:
        stages: List of curriculum stages
        max_attempts_per_stage: Max training attempts before forcing advance
        eval_frequency: Evaluate every N timesteps
        save_checkpoints: Whether to save stage checkpoints
    """
    stages: list[CurriculumStage] = field(default_factory=lambda: [
        CurriculumStage(
            name="Stage 1: Random",
            opponent_type="random",
            timesteps=500_000,
            win_rate_threshold=0.80,
            eval_games=50,
        ),
        CurriculumStage(
            name="Stage 2: Conservative",
            opponent_type="conservative",
            timesteps=500_000,
            win_rate_threshold=0.70,
            eval_games=50,
        ),
        CurriculumStage(
            name="Stage 3: Rule-Based",
            opponent_type="rule_based",
            timesteps=500_000,
            win_rate_threshold=0.60,
            eval_games=50,
        ),
        CurriculumStage(
            name="Stage 4: Aggressive",
            opponent_type="aggressive",
            timesteps=500_000,
            win_rate_threshold=0.50,
            eval_games=50,
        ),
    ])
    max_attempts_per_stage: int = 3
    eval_frequency: int = 50_000
    save_checkpoints: bool = True


@dataclass
class StageResult:
    """Result of completing a curriculum stage.

    Attributes:
        stage: The stage that was completed
        final_win_rate: Win rate achieved
        timesteps_used: Actual timesteps used
        passed: Whether threshold was met
        attempts: Number of training attempts
    """
    stage: CurriculumStage
    final_win_rate: float
    timesteps_used: int
    passed: bool
    attempts: int


class CurriculumTrainer:
    """Trainer that implements curriculum learning.

    The agent progresses through stages of increasing difficulty,
    only advancing when it achieves the required win rate.
    """

    def __init__(
        self,
        config: CurriculumConfig | None = None,
        base_save_path: str | Path = "models/curriculum",
        tensorboard_log: str | Path | None = "logs/curriculum",
        seed: int | None = 42,
        verbose: bool = True,
    ) -> None:
        """Initialize curriculum trainer.

        Args:
            config: Curriculum configuration
            base_save_path: Base path for saving models
            tensorboard_log: Path for TensorBoard logs
            seed: Random seed
            verbose: Print progress
        """
        self.config = config or CurriculumConfig()
        self.base_save_path = Path(base_save_path)
        self.tensorboard_log = Path(tensorboard_log) if tensorboard_log else None
        self.seed = seed
        self.verbose = verbose

        self.stage_results: list[StageResult] = []
        self.current_model: Any = None

    def _log(self, message: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(message)

    def _evaluate_stage(
        self,
        model: Any,
        stage: CurriculumStage,
    ) -> float:
        """Evaluate model performance for a stage.

        Args:
            model: Model to evaluate
            stage: Current stage

        Returns:
            Win rate against stage opponent
        """
        from .evaluate import evaluate_against_opponent

        result = evaluate_against_opponent(
            model,
            opponent_type=stage.opponent_type,
            num_games=stage.eval_games,
            deterministic=True,
            verbose=False,
        )
        return result.win_rate

    def _train_stage(
        self,
        stage: CurriculumStage,
        initial_model: Any | None = None,
    ) -> tuple[Any, float, int]:
        """Train for a single stage.

        Args:
            stage: Stage to train
            initial_model: Model to continue training (or None for new)

        Returns:
            Tuple of (model, final_win_rate, timesteps_used)
        """
        from .train import (
            EnvironmentConfig,
            TrainingConfig,
            create_model,
            create_vectorized_env,
        )

        env_config = EnvironmentConfig(
            num_players=2,
            opponent_type=stage.opponent_type,
            reward_type="dense",  # Use dense rewards for curriculum
        )

        training_config = TrainingConfig(
            total_timesteps=stage.timesteps,
            seed=self.seed,
        )

        # Create environment
        env = create_vectorized_env(
            env_config,
            num_envs=training_config.num_envs,
            seed=self.seed,
        )

        # Create or continue model
        if initial_model is not None:
            model = initial_model
            model.set_env(env)
        else:
            tb_log = None
            if self.tensorboard_log:
                stage_name = stage.name.replace(" ", "_").replace(":", "")
                tb_log = str(self.tensorboard_log / stage_name)
            model = create_model(env, training_config, tb_log)

        # Training with periodic evaluation
        timesteps_per_eval = self.config.eval_frequency
        total_trained = 0
        best_win_rate = 0.0

        while total_trained < stage.timesteps:
            # Train for a chunk
            chunk_size = min(timesteps_per_eval, stage.timesteps - total_trained)
            model.learn(total_timesteps=chunk_size, reset_num_timesteps=False, progress_bar=False)
            total_trained += chunk_size

            # Evaluate
            win_rate = self._evaluate_stage(model, stage)
            best_win_rate = max(best_win_rate, win_rate)

            self._log(f"  {total_trained:,}/{stage.timesteps:,} steps - Win rate: {win_rate:.1%}")

            # Check if threshold met
            if win_rate >= stage.win_rate_threshold:
                self._log(f"  Threshold {stage.win_rate_threshold:.0%} reached!")
                break

        env.close()
        return model, best_win_rate, total_trained

    def train_stage(
        self,
        stage: CurriculumStage,
        initial_model: Any | None = None,
    ) -> StageResult:
        """Train a stage with multiple attempts if needed.

        Args:
            stage: Stage to train
            initial_model: Optional model to continue from

        Returns:
            Stage result
        """
        self._log(f"\n{'='*50}")
        self._log(f"Starting {stage.name}")
        self._log(f"Opponent: {stage.opponent_type}")
        self._log(f"Target: {stage.win_rate_threshold:.0%} win rate")
        self._log(f"{'='*50}")

        model = initial_model
        best_win_rate = 0.0
        total_timesteps = 0
        attempts = 0
        passed = False

        for attempt in range(self.config.max_attempts_per_stage):
            attempts += 1
            self._log(f"\nAttempt {attempt + 1}/{self.config.max_attempts_per_stage}")

            model, win_rate, timesteps = self._train_stage(stage, model)
            total_timesteps += timesteps
            best_win_rate = max(best_win_rate, win_rate)

            if win_rate >= stage.win_rate_threshold:
                passed = True
                break

            self._log("  Did not reach threshold, retrying...")

        # Save checkpoint
        if self.config.save_checkpoints and model is not None:
            stage_name = stage.name.replace(" ", "_").replace(":", "")
            checkpoint_path = self.base_save_path / f"stage_{stage_name}"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(checkpoint_path))
            self._log(f"  Saved checkpoint: {checkpoint_path}")

        self.current_model = model

        result = StageResult(
            stage=stage,
            final_win_rate=best_win_rate,
            timesteps_used=total_timesteps,
            passed=passed,
            attempts=attempts,
        )

        self._log(f"\nStage complete: {'PASSED' if passed else 'FAILED'}")
        self._log(f"  Win rate: {best_win_rate:.1%} (target: {stage.win_rate_threshold:.0%})")
        self._log(f"  Timesteps: {total_timesteps:,}")

        return result

    def train(self) -> Any:
        """Run full curriculum training.

        Returns:
            Final trained model
        """
        self._log("\n" + "="*60)
        self._log("CURRICULUM TRAINING")
        self._log(f"Stages: {len(self.config.stages)}")
        self._log("="*60)

        self.stage_results = []
        model = None

        for i, stage in enumerate(self.config.stages):
            result = self.train_stage(stage, model)
            self.stage_results.append(result)
            model = self.current_model

            if not result.passed and i < len(self.config.stages) - 1:
                self._log(f"\nWarning: Stage {i+1} not passed, continuing anyway...")

        # Save final model
        if model is None:
            raise RuntimeError("No model trained - config has no stages")
        final_path = self.base_save_path / "final_model"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(final_path))
        self._log(f"\nFinal model saved: {final_path}")

        # Print summary
        self._log("\n" + "="*60)
        self._log("TRAINING COMPLETE")
        self._log("="*60)
        for result in self.stage_results:
            status = "PASSED" if result.passed else "FAILED"
            self._log(f"  {status} {result.stage.name}: {result.final_win_rate:.1%}")

        return model

    def get_summary(self) -> str:
        """Get training summary as string."""
        lines = ["Curriculum Training Summary", ""]
        for result in self.stage_results:
            status = "PASSED" if result.passed else "FAILED"
            lines.append(
                f"{result.stage.name}: {status} "
                f"({result.final_win_rate:.1%}, "
                f"{result.timesteps_used:,} steps, "
                f"{result.attempts} attempts)"
            )
        return "\n".join(lines)
