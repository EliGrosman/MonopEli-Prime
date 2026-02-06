# Monopoly Engine

A pure Python implementation of the Monopoly board game engine, designed for AI/RL training, web applications, and game simulations.

## Overview

**Monopoly Engine** is a framework-agnostic game engine that implements the complete ruleset of Monopoly. It's designed with clean architecture principles: pure game logic with zero I/O dependencies, full type safety, and JSON-serializable state.

### Key Features

- **Pure Logic**: No I/O operations (no GUI, networking, or file dependencies)
- **Type Safe**: Full type hints with mypy strict mode compliance
- **Fast**: Optimized for AI training (target: >1000 games/second)
- **Serializable**: Complete game state converts to/from JSON
- **Well-Tested**: Comprehensive test suite with high coverage
- **Deterministic**: Reproducible games with random seed support

### Use Cases

- **AI/RL Training**: Clean Gymnasium wrapper for reinforcement learning
- **Web Games**: FastAPI backend with WebSocket support
- **Game Simulations**: Run thousands of games for analysis
- **Rule Validation**: Test house rules and variants
- **Educational**: Learn game theory and probability

## Quick Start (Docker)

The fastest way to run MonopEli:

```bash
cp .env.example .env
docker compose up -d

# Backend: http://localhost:8000
# Frontend: http://localhost:3000
```

For production (single port, Nginx reverse proxy):

```bash
docker compose -f docker-compose.prod.yml up -d --build
# Everything served on http://localhost
```

See `docs/05_DEPLOYMENT_GUIDE.md` for full deployment instructions.

## Quick Start (Development)

### Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable package management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
cd /path/to/MonopEli-Prime

# Install dependencies
uv sync

# Run tests
uv run python -m pytest

# Run type checker
uv run python -m mypy monopoly_engine --strict
```

### Basic Usage

```python
from monopoly_engine import MonopolyGame, GameConfig

# Create a new game
config = GameConfig(num_players=4, starting_money=1500)
game = MonopolyGame(config)

# Play a turn
game.roll_dice()
game.move_current_player()
game.handle_landing()

# Check game state
print(f"Current player: {game.current_player.name}")
print(f"Position: {game.current_player.position}")
print(f"Money: ${game.current_player.money}")

# Serialize state to JSON
state_json = game.to_json()
```

## Project Structure

```
MonopEli-Prime/
├── monopoly_engine/           # Core game engine package (~1,800 lines)
│   ├── __init__.py           # Package exports
│   ├── types.py              # Enums, TypedDicts, Protocols (145 lines)
│   ├── board.py              # Board definition with 40 spaces (100 lines)
│   ├── property.py           # Property state management (153 lines)
│   ├── exceptions.py         # Custom exception types (24 lines)
│   ├── player.py             # Player state and actions (80 lines)
│   ├── cards.py              # Chance/Community Chest (98 lines)
│   ├── rules.py              # Game rules and validation (225 lines)
│   ├── actions.py            # All 15 action types (407 lines)
│   ├── state.py              # Game state management (74 lines)
│   └── game.py               # Main game orchestrator (482 lines)
│
├── monopoly_gym/              # PettingZoo RL environment (~1,000 lines)
│   ├── __init__.py           # Package exports
│   ├── env.py                # PettingZoo AEC environment (210 lines)
│   ├── action_space.py       # 149-dim action encoding (202 lines)
│   ├── observation.py        # Dict observation space (118 lines)
│   ├── rewards.py            # Configurable reward functions (210 lines) ✨ NEW
│   └── single_agent_env.py   # Gymnasium wrapper for SB3 (296 lines) ✨ NEW
│
├── agents/                    # Agent framework (~300 lines)
│   ├── __init__.py           # Package exports
│   ├── base.py               # Agent ABC
│   ├── random_agent.py       # Random baseline agent
│   └── rule_based.py         # Heuristic agents (RuleBased, Aggressive, Conservative)
│
├── training/                  # Training infrastructure (~1,000 lines) ✨ NEW
│   ├── __init__.py           # Package exports
│   ├── train.py              # MaskablePPO training (278 lines)
│   ├── evaluate.py           # Evaluation harness (305 lines)
│   └── curriculum.py         # Curriculum learning (354 lines)
│
├── tests/                     # Test suite (~9,400 lines, 1120 tests)
│   ├── conftest.py           # Shared fixtures
│   ├── test_types.py         # Type validation tests (27 tests)
│   ├── test_board.py         # Board structure tests (32 tests)
│   ├── test_property.py      # Property management tests (35 tests)
│   ├── test_player.py        # Player state tests (29 tests)
│   ├── test_cards.py         # Card effect tests (42 tests)
│   ├── test_rules.py         # Rules validation tests (42 tests)
│   ├── test_actions.py       # Action validation tests (74 tests)
│   ├── test_state.py         # State management tests (30 tests)
│   ├── test_game.py          # Game orchestration tests (34 tests)
│   ├── test_integration.py   # Integration tests (24 tests)
│   ├── test_monopoly_gym.py  # Environment tests (33 tests)
│   ├── test_action_space.py  # Action encoding tests (265 tests)
│   ├── test_observation.py   # Observation encoding tests (111 tests)
│   ├── test_agents.py        # Agent tests (72 tests)
│   ├── test_rewards.py       # Reward function tests (65 tests)
│   ├── test_single_agent_env.py # Single-agent env tests (90 tests)
│   └── test_training.py      # Training infrastructure tests (51 tests) ✨ NEW
│
├── scripts/                   # Utility scripts (~550 lines)
│   ├── cli_runner.py         # Interactive CLI runner (68 lines)
│   ├── benchmark.py          # Performance benchmarks (135 lines)
│   ├── train_ppo.py          # Training CLI (174 lines) ✨ NEW
│   └── evaluate_agent.py     # Evaluation CLI (137 lines) ✨ NEW
│   └── random_bot.py         # Random action bot (33 lines)
│
├── pyproject.toml            # Project configuration
├── uv.lock                   # Locked dependencies
└── README.md                 # This file
```

## Tools

### CLI Runner

Interactive command-line interface for playing games and testing:

```bash
# Play an interactive game
uv run python scripts/cli_runner.py

# Watch a game with random bots
uv run python scripts/cli_runner.py --players 4 --watch

# Run a specific number of games
uv run python scripts/cli_runner.py --games 10
```

Features:
- Interactive command-line gameplay
- Watch mode for observing bot games
- JSON state inspection
- Event log viewing
- Multi-player support (2-8 players)

### Benchmark Suite

Performance measurement and profiling:

```bash
# Run standard benchmark (1000 games)
uv run python scripts/benchmark.py

# Quick benchmark
uv run python scripts/benchmark.py --games 100

# With profiling
uv run python scripts/benchmark.py --profile

# Detailed statistics
uv run python scripts/benchmark.py --verbose
```

Metrics tracked:
- Games per second throughput
- Average game length (turns)
- Winner distribution
- Memory usage
- Action execution times

### Random Bot

Automated player that makes random valid moves:

```python
from scripts.random_bot import RandomBot
from monopoly_engine import MonopolyGame

game = MonopolyGame(num_players=4)
bot = RandomBot(game)

# Get random valid action for current player
action = bot.get_action()
```

Used for:
- Automated testing
- Performance benchmarking
- Game simulation
- Baseline AI comparison

## Development

### Requirements

- Python 3.11+
- uv (package manager)

### Development Tools

- **pytest**: Testing framework with coverage reporting
- **mypy**: Static type checker (strict mode)
- **ruff**: Fast Python linter
- **black**: Code formatter
- **hypothesis**: Property-based testing

### Development Workflow

```bash
# Run all tests
uv run python -m pytest

# Run tests with coverage report
uv run python -m pytest --cov=monopoly_engine --cov-report=html

# Type check
uv run python -m mypy monopoly_engine --strict

# Lint code
uv run python -m ruff check monopoly_engine

# Format code
uv run python -m black monopoly_engine

# Run all quality checks
uv run python -m pytest && \
  uv run python -m mypy monopoly_engine --strict && \
  uv run python -m ruff check monopoly_engine
```

### Adding Dependencies

```bash
# Add a runtime dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Sync dependencies
uv sync
```

## Architecture

### Module Overview

| Module | Purpose | Lines | Coverage | Status |
|--------|---------|-------|----------|--------|
| `types.py` | Core type definitions (enums, TypedDicts, protocols) | 145 | 100% | ✅ Complete |
| `board.py` | Immutable board definition with 40 spaces | 100 | 98% | ✅ Complete |
| `property.py` | Property ownership, mortgages, buildings | 153 | 86% | ✅ Complete |
| `exceptions.py` | Custom exception hierarchy | 24 | 100% | ✅ Complete |
| `player.py` | Player state, money, position, inventory | 80 | 99% | ✅ Complete |
| `cards.py` | Chance/Community Chest card definitions | 98 | 90% | ✅ Complete |
| `rules.py` | Rent calculation, building rules, game rules | 225 | 88% | ✅ Complete |
| `actions.py` | All 15 action types with validation/execution | 407 | 87% | ✅ Complete |
| `state.py` | Game state container with serialization | 74 | 99% | ✅ Complete |
| `game.py` | Main game orchestrator and controller | 483 | 55% | ✅ Complete |

**Totals**: ~1,800 lines of production code, 400 tests, 91% overall coverage

### Design Principles

1. **Separation of Concerns**: Game logic is independent of I/O
2. **Immutable Definitions**: Board, cards, and rules are frozen dataclasses
3. **Mutable State**: Player positions, money, ownership can change
4. **Validation First**: All actions are validated before execution
5. **Type Safety**: Full type hints, passes `mypy --strict`
6. **Testability**: Pure functions make testing straightforward
7. **Determinism**: Same seed produces identical games

### Key Design Decisions

- **No Global State**: Game state is encapsulated in objects
- **JSON Serialization**: State can be saved/loaded without pickle
- **Protocol-Based**: Duck typing with structural subtyping
- **Action Validation**: Separate validation from execution
- **Event-Driven**: Clear hooks for extending behavior

## Testing

### Test Coverage

The project maintains high test coverage:

```bash
# Run tests with coverage
uv run python -m pytest --cov=monopoly_engine --cov-report=term-missing

# Generate HTML coverage report
uv run python -m pytest --cov=monopoly_engine --cov-report=html
open htmlcov/index.html
```

### Test Categories

1. **Unit Tests**: Individual function/class behavior
2. **Integration Tests**: Module interactions
3. **Property Tests**: Invariants with hypothesis
4. **Edge Cases**: Boundary conditions and error handling

### Example Test

```python
def test_rent_calculation_with_monopoly():
    """Test that rent doubles when a player owns a complete monopoly."""
    game = create_test_game()

    # Give player 0 both Mediterranean and Baltic (brown monopoly)
    game.property_manager.properties[1].owner = 0
    game.property_manager.properties[3].owner = 0

    rent = calculate_rent(game, position=1, owner=0, dice_roll=7)

    # Base rent is $2, doubled for monopoly
    assert rent == 4
```

## Game Rules Implemented

### Complete Monopoly Ruleset

- ✅ 40-space board with accurate property values
- ✅ Property purchase, rent, and mortgage
- ✅ Building houses and hotels (even building requirement)
- ✅ Monopoly detection and rent doubling
- ✅ Chance and Community Chest cards (16 each)
- ✅ Jail mechanics (doubles, pay $50, use card)
- ✅ Utilities (4× or 10× dice roll)
- ✅ Railroads ($25-$200 rent based on ownership)
- ✅ Free Parking, Luxury Tax, Income Tax
- ✅ Bankruptcy and player elimination
- ✅ Winner determination

### Board Layout

The standard Monopoly board with 40 spaces:

- **Properties**: 22 purchasable properties in 8 color groups
- **Railroads**: 4 railroads ($200 each)
- **Utilities**: 2 utilities ($150 each)
- **Special**: GO, Jail, Free Parking, Go To Jail
- **Tax**: Income Tax ($200), Luxury Tax ($100)
- **Cards**: Chance (×3), Community Chest (×3)

See `monopoly_engine/board.py` for complete board definition.

### Action System

All game actions are implemented in `actions.py` with comprehensive validation:

1. **RollDice** - Roll dice to move around the board
2. **BuyProperty** - Purchase an unowned property
3. **BuildHouse** - Build a house on a monopoly property
4. **BuildHotel** - Build a hotel (upgrade from 4 houses)
5. **SellHouse** - Sell house back to the bank
6. **SellHotel** - Sell hotel back to the bank (returns to 4 houses)
7. **MortgageProperty** - Mortgage property for 50% of cost
8. **UnmortgageProperty** - Pay 110% of mortgage value to unmortgage
9. **ProposeTrade** - Propose trade with another player
10. **AcceptTrade** - Accept a pending trade offer
11. **RejectTrade** - Reject a pending trade offer
12. **PayJailFine** - Pay $50 to get out of jail
13. **UseJailCard** - Use Get Out of Jail Free card
14. **DeclareBankruptcy** - Declare bankruptcy and exit game
15. **EndTurn** - End turn and advance to next player

Each action follows the validate-then-execute pattern, ensuring all game rules are enforced before state changes.

## Performance

### Achieved Metrics (Week 5)

- **Throughput**: 1,234 games/second (single-threaded) ✅
- **Latency**: <1ms per action validation ✅
- **Memory**: ~8MB per game state ✅
- **Test Performance**: 369 tests in 0.52s

### Benchmarking

Run the benchmark suite to measure performance on your system:

```bash
# Run full benchmark suite
uv run python scripts/benchmark.py

# Quick benchmark (100 games)
uv run python scripts/benchmark.py --games 100

# Detailed profiling
uv run python scripts/benchmark.py --profile
```

### Optimization Strategy

- Immutable board/card definitions (shared across games)
- Efficient property lookup with dictionaries
- Minimal allocations in hot paths
- Type hints enable potential Cython/mypyc compilation
- Random bot for consistent benchmarking

## Roadmap

### Phase 1: Core Engine ✅ COMPLETE

**All Weeks Complete (1-6)**:
- ✅ Pure Python game engine foundation
- ✅ Complete board definition with 40 spaces
- ✅ All property types and ownership tracking
- ✅ Player state management
- ✅ 32 Chance/Community Chest cards
- ✅ Complete game rules (rent, building, mortgage)
- ✅ All 15 action types with validation
- ✅ Game state serialization (state.py)
- ✅ Main game orchestrator with centralized state management (game.py)
- ✅ Event logging infrastructure
- ✅ Trade management system
- ✅ Bankruptcy handling
- ✅ 400 tests, 91% coverage
- ✅ Full type safety (mypy strict mode)
- ✅ CLI runner for interactive testing (cli_runner.py)
- ✅ Performance benchmarking (benchmark.py, 1,234 games/sec)
- ✅ Random bot for automated testing (random_bot.py)
- ✅ Integration tests validating full game scenarios
- ✅ Legacy compatibility tests
- ✅ CI/CD pipeline (GitHub Actions)
- ✅ All linting issues resolved (ruff)
- ✅ Comprehensive documentation

**Status**: 10/10 core modules complete, ~1,800 lines of production code, ~5,000 lines of tests, all success criteria met

### Phase 2: PettingZoo RL Environment ✅ COMPLETE

**All Weeks Complete (1-6)**:
- ✅ PettingZoo AEC environment (`monopoly_gym/env.py`)
- ✅ 149-dimensional action space encoding
- ✅ Dict observation space
- ✅ Agent framework (Random, RuleBased, Aggressive, Conservative)
- ✅ Configurable reward functions
- ✅ Training infrastructure with MaskablePPO
- ✅ Curriculum learning (4-stage progressive difficulty)
- ✅ Self-play training infrastructure
- ✅ 94% win rate vs random baseline (exceeds 90% target)
- ✅ 1,144 tests, 92% coverage

### Phase 3: FastAPI Web Backend ✅ COMPLETE

**All Weeks Complete (1-6)**:
- ✅ FastAPI REST API with OpenAPI documentation
- ✅ WebSocket for real-time game updates
- ✅ Lobby system with matchmaking
- ✅ AI opponent integration (4 agent types)
- ✅ Player session management
- ✅ Request logging with request ID tracking
- ✅ Global error handling with standard format
- ✅ Rate limiting (token bucket algorithm)
- ✅ Locust load testing
- ✅ 199 API tests, zero mypy errors

### Phase 4: Frontend 🚧 NEXT
- React/TypeScript UI
- Real-time game board
- Player dashboard
- Spectator mode

### Phase 5: Deployment
- Docker containerization
- Kubernetes deployment
- Monitoring and logging
- Performance tuning

## Contributing

### Code Style

- Follow PEP 8 conventions
- Use type hints on all functions
- Write docstrings for public APIs
- Keep functions small and focused
- Prefer immutability where possible

### Pull Request Process

1. Create a feature branch
2. Write tests for new functionality
3. Ensure all tests pass (`uv run python -m pytest`)
4. Run type checker (`uv run python -m mypy monopoly_engine --strict`)
5. Format code (`uv run python -m black monopoly_engine`)
6. Lint code (`uv run python -m ruff check monopoly_engine`)
7. Submit PR with clear description

## License

[Add license information]

## Credits

Built as part of the MonopEli project - a multi-phase implementation of Monopoly for AI training and web gaming.

### Related Projects

- **MonopEli-Server**: Legacy implementation (reference only)
- **MonopEli-Client**: Legacy Pygame client (reference only)
- **MonopEli-Prime**: This project (Phase 1 - Core Engine)

## Contact

[Add contact information]

---

**Status**: Phase 3 - COMPLETE
**Version**: 3.0.0-dev (Phase 3 Complete)
**Last Updated**: 2026-02-03

## API Server

**Key Features**:
- Full REST API with OpenAPI documentation
- WebSocket for real-time game updates
- Lobby system with matchmaking
- AI opponents (4 agent types)
- Player session management
- Request ID tracking
- Standardized error responses
- Token bucket rate limiting

**Metrics**:
- 199 API tests passing
- ~3,300 lines of production code
- ~3,600 lines of test code
- Zero mypy errors (strict mode)
- Zero ruff linting issues

### Running the API Server

```bash
# Development (with auto-reload)
uv run uvicorn api.main:app --reload --port 8000

# Production (with multiple workers)
uv run uvicorn api.main:create_app --factory --workers 4 --port 8000

# View API documentation
# Open http://localhost:8000/api/docs
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api` | GET | API info and stats |
| `/api/games` | GET/POST | List/create games |
| `/api/games/{id}` | GET/DELETE | Get/delete game |
| `/api/lobbies` | GET/POST | List/create lobbies |
| `/api/lobbies/{id}/join` | POST | Join a lobby |
| `/api/lobbies/{id}/start` | POST | Start game from lobby |
| `/api/players/session` | POST | Create player session |
| `/ws/games/{id}` | WS | Real-time game connection |

### Environment Variables

```bash
# Server config
MONOPELI_HOST=0.0.0.0
MONOPELI_PORT=8000
MONOPELI_DEBUG=true
MONOPELI_LOG_LEVEL=INFO

# Rate limiting
MONOPELI_RATE_LIMIT_PER_MINUTE=60
MONOPELI_RATE_LIMIT_BURST=10

# CORS origins
MONOPELI_CORS_ORIGINS=["http://localhost:3000"]
```

### Load Testing

```bash
# Install locust
uv add --dev locust

# Run load test (web UI)
uv run locust -f scripts/load_test.py --host http://localhost:8000

# Run headless load test
uv run locust -f scripts/load_test.py --host http://localhost:8000 \
  --headless -u 100 -r 10 -t 60s
```

## Phase 2: RL Training Environment

### Quick Start - Training an Agent

```bash
# Install training dependencies
uv sync --extra training

# Train a PPO agent (100k steps)
uv run python scripts/train_ppo.py --timesteps 100000

# Evaluate trained agent
uv run python scripts/evaluate_agent.py --model models/ppo_monopoly --games 100

# Self-play training
uv run python scripts/train_self_play.py --timesteps 500000
```

### Using the RL Environment

```python
# Multi-agent environment (PettingZoo)
from monopoly_gym import MonopolyEnv

env = MonopolyEnv(num_players=4, reward_type="sparse")
env.reset(seed=42)

for agent in env.agent_iter():
    obs, reward, term, trunc, info = env.last()
    if term or trunc:
        action = None
    else:
        action_mask = info["action_mask"]
        action = select_action(obs, action_mask)
    env.step(action)

# Single-agent environment (Gymnasium - for SB3)
from monopoly_gym import SingleAgentMonopolyEnv

env = SingleAgentMonopolyEnv(
    num_players=2,
    opponent_type="random",  # or "rule_based", "aggressive", "conservative"
    reward_type="dense",
)
obs, info = env.reset()

while True:
    action_mask = info["action_mask"]
    action = model.predict(obs, action_masks=action_mask)
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        break
```

### Training with MaskablePPO

```python
from training import train_agent, TrainingConfig, EnvironmentConfig

config = TrainingConfig(
    total_timesteps=1_000_000,
    num_envs=8,
    learning_rate=3e-4,
)
env_config = EnvironmentConfig(
    num_players=2,
    opponent_type="random",
    reward_type="dense",
)

model = train_agent(config, env_config, save_path="models/my_agent")
```

### Self-Play Training

```python
from training import SelfPlayConfig, SelfPlayTrainer

config = SelfPlayConfig(
    total_timesteps=1_000_000,
    checkpoint_freq=50_000,
    past_version_prob=0.5,
)
trainer = SelfPlayTrainer(config)
model = trainer.train()
```

### Curriculum Learning

```python
from training import CurriculumConfig, CurriculumTrainer

config = CurriculumConfig(
    stages=[
        ("random", 100_000, 0.8),      # Train vs random until 80% win rate
        ("rule_based", 200_000, 0.6),  # Then vs rule_based until 60%
        ("aggressive", 300_000, 0.5),  # Then vs aggressive until 50%
    ],
)
trainer = CurriculumTrainer(config)
model = trainer.train()
```

### Phase 2 Achievements

| Metric | Target | Achieved |
|--------|--------|----------|
| Win rate vs random | >90% | **94%** ✅ |
| Win rate vs rule_based | >70% | 22% (2.8x baseline) |
| Training throughput | - | ~1,000 steps/sec |
| Tests | 100+ | **1144** ✅ |
| Coverage | 80%+ | **92%** ✅ |

### Key Features

- **PettingZoo AEC API**: Full multi-agent support
- **149-dim Action Space**: All game actions except trades
- **Action Masking**: Only valid actions can be selected
- **Configurable Rewards**: Sparse (+1/-1) or dense (net worth changes)
- **Agent Framework**: Random, RuleBased, Aggressive, Conservative
- **Training Infrastructure**: MaskablePPO, curriculum learning, self-play
- **Evaluation Harness**: Automated win rate testing

### Next: Phase 2.5 - Trade Actions

Trade actions are deferred to Phase 2.5 to reduce action space complexity:
- Property-for-property trades
- Cash-for-property trades
- Trade negotiation with opponent agents
