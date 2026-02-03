"""Tests for the training infrastructure.

These tests verify the training, evaluation, and curriculum learning
infrastructure. Tests that require sb3 dependencies are skipped if
those packages are not installed.
"""

from __future__ import annotations

import pytest
from pathlib import Path


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self) -> None:
        from training import TrainingConfig

        config = TrainingConfig()
        assert config.total_timesteps == 1_000_000
        assert config.learning_rate == 3e-4
        assert config.n_steps == 2048
        assert config.batch_size == 64
        assert config.n_epochs == 10
        assert config.gamma == 0.99
        assert config.seed == 42

    def test_custom_values(self) -> None:
        from training import TrainingConfig

        config = TrainingConfig(
            total_timesteps=500_000,
            learning_rate=1e-4,
            num_envs=4,
        )
        assert config.total_timesteps == 500_000
        assert config.learning_rate == 1e-4
        assert config.num_envs == 4

    def test_all_hyperparameters(self) -> None:
        """Test all hyperparameter fields are accessible."""
        from training import TrainingConfig

        config = TrainingConfig(
            total_timesteps=100_000,
            learning_rate=1e-3,
            n_steps=1024,
            batch_size=32,
            n_epochs=5,
            gamma=0.95,
            gae_lambda=0.9,
            clip_range=0.1,
            ent_coef=0.02,
            vf_coef=0.6,
            max_grad_norm=1.0,
            num_envs=4,
            seed=123,
        )
        assert config.gae_lambda == 0.9
        assert config.clip_range == 0.1
        assert config.ent_coef == 0.02
        assert config.vf_coef == 0.6
        assert config.max_grad_norm == 1.0

    def test_seed_can_be_none(self) -> None:
        """Test that seed can be None."""
        from training import TrainingConfig

        config = TrainingConfig(seed=None)
        assert config.seed is None


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig dataclass."""

    def test_default_values(self) -> None:
        from training import EnvironmentConfig

        config = EnvironmentConfig()
        assert config.num_players == 2
        assert config.opponent_type == "random"
        assert config.max_turns == 1000
        assert config.reward_type == "sparse"

    def test_custom_values(self) -> None:
        from training import EnvironmentConfig

        config = EnvironmentConfig(
            num_players=4,
            opponent_type="rule_based",
            reward_type="dense",
        )
        assert config.num_players == 4
        assert config.opponent_type == "rule_based"
        assert config.reward_type == "dense"

    def test_all_opponent_types(self) -> None:
        """Test various opponent type values."""
        from training import EnvironmentConfig

        for opponent in ["random", "rule_based", "aggressive", "conservative"]:
            config = EnvironmentConfig(opponent_type=opponent)
            assert config.opponent_type == opponent

    def test_reward_types(self) -> None:
        """Test both reward type values."""
        from training import EnvironmentConfig

        sparse = EnvironmentConfig(reward_type="sparse")
        assert sparse.reward_type == "sparse"

        dense = EnvironmentConfig(reward_type="dense")
        assert dense.reward_type == "dense"


class TestEvaluationResult:
    """Tests for EvaluationResult dataclass."""

    def test_creation(self) -> None:
        from training import EvaluationResult

        result = EvaluationResult(
            opponent_type="random",
            num_games=100,
            wins=75,
            losses=20,
            draws=5,
            win_rate=0.75,
            avg_game_length=150.0,
            avg_reward=0.55,
            games_per_second=10.0,
        )
        assert result.opponent_type == "random"
        assert result.num_games == 100
        assert result.wins == 75
        assert result.losses == 20
        assert result.draws == 5
        assert result.win_rate == 0.75
        assert result.avg_game_length == 150.0
        assert result.avg_reward == 0.55
        assert result.games_per_second == 10.0

    def test_str_representation(self) -> None:
        from training import EvaluationResult

        result = EvaluationResult(
            opponent_type="random",
            num_games=100,
            wins=80,
            losses=15,
            draws=5,
            win_rate=0.80,
            avg_game_length=120.0,
            avg_reward=0.6,
            games_per_second=15.0,
        )
        s = str(result)
        assert "random" in s
        assert "80.0%" in s
        assert "80W" in s
        assert "15L" in s
        assert "5D" in s
        assert "120.0 turns" in s

    def test_str_with_different_win_rate(self) -> None:
        """Test string representation with different win rates."""
        from training import EvaluationResult

        result = EvaluationResult(
            opponent_type="rule_based",
            num_games=50,
            wins=30,
            losses=18,
            draws=2,
            win_rate=0.60,
            avg_game_length=200.5,
            avg_reward=0.2,
            games_per_second=5.0,
        )
        s = str(result)
        assert "rule_based" in s
        assert "60.0%" in s


class TestEvaluationSummary:
    """Tests for EvaluationSummary dataclass."""

    def test_empty_summary(self) -> None:
        from training import EvaluationSummary

        summary = EvaluationSummary(model_path="test/model")
        assert summary.model_path == "test/model"
        assert summary.total_games == 0
        assert summary.total_wins == 0
        assert summary.overall_win_rate == 0.0

    def test_with_single_result(self) -> None:
        from training import EvaluationSummary, EvaluationResult

        summary = EvaluationSummary(model_path="test/model")
        summary.results["random"] = EvaluationResult(
            opponent_type="random",
            num_games=100,
            wins=90,
            losses=10,
            draws=0,
            win_rate=0.90,
            avg_game_length=100.0,
            avg_reward=0.8,
            games_per_second=20.0,
        )

        assert summary.total_games == 100
        assert summary.total_wins == 90
        assert summary.overall_win_rate == 0.90

    def test_with_multiple_results(self) -> None:
        from training import EvaluationSummary, EvaluationResult

        summary = EvaluationSummary(model_path="test/model")
        summary.results["random"] = EvaluationResult(
            opponent_type="random",
            num_games=100,
            wins=90,
            losses=10,
            draws=0,
            win_rate=0.90,
            avg_game_length=100.0,
            avg_reward=0.8,
            games_per_second=20.0,
        )
        summary.results["rule_based"] = EvaluationResult(
            opponent_type="rule_based",
            num_games=100,
            wins=70,
            losses=30,
            draws=0,
            win_rate=0.70,
            avg_game_length=150.0,
            avg_reward=0.4,
            games_per_second=15.0,
        )

        assert summary.total_games == 200
        assert summary.total_wins == 160
        assert summary.overall_win_rate == 0.80

    def test_str_representation(self) -> None:
        from training import EvaluationSummary, EvaluationResult

        summary = EvaluationSummary(model_path="models/test")
        summary.results["random"] = EvaluationResult(
            opponent_type="random",
            num_games=100,
            wins=80,
            losses=20,
            draws=0,
            win_rate=0.80,
            avg_game_length=100.0,
            avg_reward=0.6,
            games_per_second=10.0,
        )

        s = str(summary)
        assert "models/test" in s
        assert "80.0%" in s


class TestCurriculumStage:
    """Tests for CurriculumStage dataclass."""

    def test_creation(self) -> None:
        from training import CurriculumStage

        stage = CurriculumStage(
            name="Test Stage",
            opponent_type="random",
            timesteps=100_000,
            win_rate_threshold=0.75,
        )
        assert stage.name == "Test Stage"
        assert stage.opponent_type == "random"
        assert stage.timesteps == 100_000
        assert stage.win_rate_threshold == 0.75
        assert stage.eval_games == 50  # default

    def test_custom_eval_games(self) -> None:
        from training import CurriculumStage

        stage = CurriculumStage(
            name="Test",
            opponent_type="random",
            timesteps=50_000,
            win_rate_threshold=0.80,
            eval_games=100,
        )
        assert stage.eval_games == 100


class TestCurriculumConfig:
    """Tests for curriculum learning configuration."""

    def test_default_stages(self) -> None:
        from training import CurriculumConfig

        config = CurriculumConfig()
        assert len(config.stages) == 4
        assert config.stages[0].opponent_type == "random"
        assert config.stages[1].opponent_type == "conservative"
        assert config.stages[2].opponent_type == "rule_based"
        assert config.stages[3].opponent_type == "aggressive"

    def test_default_stage_thresholds(self) -> None:
        """Test that stages have decreasing thresholds."""
        from training import CurriculumConfig

        config = CurriculumConfig()
        thresholds = [s.win_rate_threshold for s in config.stages]
        assert thresholds == [0.80, 0.70, 0.60, 0.50]

    def test_default_config_values(self) -> None:
        from training import CurriculumConfig

        config = CurriculumConfig()
        assert config.max_attempts_per_stage == 3
        assert config.eval_frequency == 50_000
        assert config.save_checkpoints is True

    def test_custom_stages(self) -> None:
        from training import CurriculumConfig, CurriculumStage

        custom_stages = [
            CurriculumStage(
                name="Easy",
                opponent_type="random",
                timesteps=200_000,
                win_rate_threshold=0.90,
            ),
            CurriculumStage(
                name="Hard",
                opponent_type="rule_based",
                timesteps=300_000,
                win_rate_threshold=0.60,
            ),
        ]
        config = CurriculumConfig(stages=custom_stages)
        assert len(config.stages) == 2
        assert config.stages[0].name == "Easy"
        assert config.stages[1].name == "Hard"


class TestStageResult:
    """Tests for StageResult dataclass."""

    def test_creation(self) -> None:
        from training import CurriculumStage, StageResult

        stage = CurriculumStage(
            name="Test",
            opponent_type="random",
            timesteps=100_000,
            win_rate_threshold=0.8,
        )
        result = StageResult(
            stage=stage,
            final_win_rate=0.85,
            timesteps_used=80_000,
            passed=True,
            attempts=1,
        )
        assert result.stage == stage
        assert result.final_win_rate == 0.85
        assert result.timesteps_used == 80_000
        assert result.passed is True
        assert result.attempts == 1

    def test_failed_stage(self) -> None:
        from training import CurriculumStage, StageResult

        stage = CurriculumStage(
            name="Hard",
            opponent_type="aggressive",
            timesteps=500_000,
            win_rate_threshold=0.70,
        )
        result = StageResult(
            stage=stage,
            final_win_rate=0.45,
            timesteps_used=1_500_000,  # 3 attempts
            passed=False,
            attempts=3,
        )
        assert result.passed is False
        assert result.attempts == 3


class TestMakeEnv:
    """Tests for environment factory function."""

    def test_make_env_creates_callable(self) -> None:
        from training import make_env, EnvironmentConfig

        config = EnvironmentConfig()
        factory = make_env(config, seed=42)
        assert callable(factory)

    def test_make_env_creates_valid_env(self) -> None:
        from training import make_env, EnvironmentConfig

        config = EnvironmentConfig(num_players=2)
        factory = make_env(config, seed=42)
        env = factory()

        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')

        env.close()

    def test_make_env_with_different_configs(self) -> None:
        """Test make_env with various environment configurations."""
        from training import make_env, EnvironmentConfig

        # Test with different num_players
        for num_players in [2, 3, 4]:
            config = EnvironmentConfig(num_players=num_players)
            factory = make_env(config, seed=42)
            env = factory()
            assert env.num_players == num_players
            env.close()

    def test_make_env_with_seed_none(self) -> None:
        """Test make_env with no seed."""
        from training import make_env, EnvironmentConfig

        config = EnvironmentConfig()
        factory = make_env(config, seed=None)
        env = factory()
        assert env is not None
        env.close()


class TestCurriculumTrainer:
    """Tests for CurriculumTrainer class."""

    def test_trainer_initialization_default(self) -> None:
        from training import CurriculumTrainer

        trainer = CurriculumTrainer()
        assert trainer.config is not None
        assert len(trainer.config.stages) == 4
        assert trainer.verbose is True
        assert trainer.seed == 42

    def test_trainer_initialization_custom(self) -> None:
        from training import CurriculumTrainer, CurriculumConfig

        config = CurriculumConfig(max_attempts_per_stage=5)
        trainer = CurriculumTrainer(
            config=config,
            base_save_path="custom/path",
            tensorboard_log=None,
            seed=123,
            verbose=False,
        )
        assert trainer.config.max_attempts_per_stage == 5
        assert trainer.base_save_path == Path("custom/path")
        assert trainer.tensorboard_log is None
        assert trainer.seed == 123
        assert trainer.verbose is False

    def test_trainer_log_silent_when_not_verbose(self, capsys) -> None:  # type: ignore[no-untyped-def]
        from training import CurriculumTrainer

        trainer = CurriculumTrainer(verbose=False)
        trainer._log("This should not appear")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_trainer_log_prints_when_verbose(self, capsys) -> None:  # type: ignore[no-untyped-def]
        from training import CurriculumTrainer

        trainer = CurriculumTrainer(verbose=True)
        trainer._log("Test message")
        captured = capsys.readouterr()
        assert "Test message" in captured.out

    def test_trainer_stage_results_initially_empty(self) -> None:
        from training import CurriculumTrainer

        trainer = CurriculumTrainer()
        assert trainer.stage_results == []
        assert trainer.current_model is None

    def test_trainer_get_summary_empty(self) -> None:
        from training import CurriculumTrainer

        trainer = CurriculumTrainer()
        summary = trainer.get_summary()
        assert "Curriculum Training Summary" in summary


# Tests that require sb3 - skip if not installed
class TestSB3Integration:
    """Tests requiring stable-baselines3."""

    @pytest.fixture
    def sb3(self):  # type: ignore[no-untyped-def]
        """Import sb3 or skip."""
        return pytest.importorskip("sb3_contrib")

    def test_import_maskable_ppo(self, sb3) -> None:  # type: ignore[no-untyped-def]
        """Test that MaskablePPO can be imported."""
        from sb3_contrib import MaskablePPO
        assert MaskablePPO is not None

    def test_create_model_function_exists(self, sb3) -> None:  # type: ignore[no-untyped-def]
        """Test that create_model function is available."""
        from training import create_model
        assert callable(create_model)

    def test_create_vectorized_env_function_exists(self, sb3) -> None:  # type: ignore[no-untyped-def]
        """Test that create_vectorized_env is available."""
        from training import create_vectorized_env
        assert callable(create_vectorized_env)

    def test_train_agent_function_exists(self, sb3) -> None:  # type: ignore[no-untyped-def]
        """Test that train_agent function is available."""
        from training import train_agent
        assert callable(train_agent)

    def test_load_model_function_exists(self, sb3) -> None:  # type: ignore[no-untyped-def]
        """Test that load_model function is available."""
        from training import load_model
        assert callable(load_model)


class TestPlayGame:
    """Tests for play_game function (requires sb3)."""

    @pytest.fixture
    def sb3(self):  # type: ignore[no-untyped-def]
        return pytest.importorskip("sb3_contrib")

    def test_play_game_import(self, sb3) -> None:  # type: ignore[no-untyped-def]
        from training import play_game
        assert callable(play_game)


class TestEvaluationFunctions:
    """Tests for evaluation functions (requires sb3)."""

    @pytest.fixture
    def sb3(self):  # type: ignore[no-untyped-def]
        return pytest.importorskip("sb3_contrib")

    def test_evaluate_against_opponent_import(self, sb3) -> None:  # type: ignore[no-untyped-def]
        from training import evaluate_against_opponent
        assert callable(evaluate_against_opponent)

    def test_evaluate_agent_import(self, sb3) -> None:  # type: ignore[no-untyped-def]
        from training import evaluate_agent
        assert callable(evaluate_agent)

    def test_quick_evaluate_import(self, sb3) -> None:  # type: ignore[no-untyped-def]
        from training import quick_evaluate
        assert callable(quick_evaluate)


class TestTrainingModuleExports:
    """Test that the training module exports all expected items."""

    def test_all_exports_present(self) -> None:
        """Test that __all__ contains expected items."""
        import training

        expected_exports = [
            # Training
            "TrainingConfig",
            "EnvironmentConfig",
            "train_agent",
            "load_model",
            "create_vectorized_env",
            "create_model",
            "make_env",
            # Evaluation
            "EvaluationResult",
            "EvaluationSummary",
            "evaluate_agent",
            "evaluate_against_opponent",
            "quick_evaluate",
            "play_game",
            # Curriculum
            "CurriculumStage",
            "CurriculumConfig",
            "CurriculumTrainer",
            "StageResult",
        ]

        for export in expected_exports:
            assert export in training.__all__, f"Missing export: {export}"

    def test_configs_importable_without_sb3(self) -> None:
        """Test that config classes can be imported without sb3."""
        # These imports should work without sb3
        from training import (
            TrainingConfig,
            EnvironmentConfig,
            EvaluationResult,
            EvaluationSummary,
            CurriculumStage,
            CurriculumConfig,
            StageResult,
        )

        # Verify they are the expected types
        assert TrainingConfig is not None
        assert EnvironmentConfig is not None
        assert EvaluationResult is not None
        assert EvaluationSummary is not None
        assert CurriculumStage is not None
        assert CurriculumConfig is not None
        assert StageResult is not None


class TestConfigDataclassFeatures:
    """Test dataclass features of config classes."""

    def test_training_config_is_dataclass(self) -> None:
        from dataclasses import is_dataclass
        from training import TrainingConfig

        assert is_dataclass(TrainingConfig)

    def test_environment_config_is_dataclass(self) -> None:
        from dataclasses import is_dataclass
        from training import EnvironmentConfig

        assert is_dataclass(EnvironmentConfig)

    def test_evaluation_result_is_dataclass(self) -> None:
        from dataclasses import is_dataclass
        from training import EvaluationResult

        assert is_dataclass(EvaluationResult)

    def test_curriculum_stage_is_dataclass(self) -> None:
        from dataclasses import is_dataclass
        from training import CurriculumStage

        assert is_dataclass(CurriculumStage)

    def test_stage_result_is_dataclass(self) -> None:
        from dataclasses import is_dataclass
        from training import StageResult

        assert is_dataclass(StageResult)

    def test_curriculum_config_is_dataclass(self) -> None:
        from dataclasses import is_dataclass
        from training import CurriculumConfig

        assert is_dataclass(CurriculumConfig)

    def test_evaluation_summary_is_dataclass(self) -> None:
        from dataclasses import is_dataclass
        from training import EvaluationSummary

        assert is_dataclass(EvaluationSummary)
