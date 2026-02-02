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

## Quick Start

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
├── monopoly_engine/           # Core game engine package (1,801 lines)
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
│   └── game.py               # Main game orchestrator (483 lines)
│
├── tests/                     # Test suite (~3,500 lines, 345 tests)
│   ├── conftest.py           # Shared fixtures
│   ├── test_types.py         # Type validation tests (27 tests)
│   ├── test_board.py         # Board structure tests (32 tests)
│   ├── test_property.py      # Property management tests (35 tests)
│   ├── test_player.py        # Player state tests (29 tests)
│   ├── test_cards.py         # Card effect tests (42 tests)
│   ├── test_rules.py         # Rules validation tests (42 tests)
│   ├── test_actions.py       # Action validation tests (74 tests)
│   ├── test_state.py         # State management tests (30 tests)
│   └── test_game.py          # Game orchestration tests (34 tests)
│
├── scripts/                   # Utility scripts
│   └── (CLI runner, benchmarks - coming Week 5)
│
├── pyproject.toml            # Project configuration
├── uv.lock                   # Locked dependencies
└── README.md                 # This file
```

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

**Totals**: 1,801 lines of production code, 345 tests, 82% overall coverage

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

### Target Metrics

- **Throughput**: >1,000 games/second (single-threaded)
- **Latency**: <1ms per action validation
- **Memory**: <10MB per game state

### Optimization Strategy

- Immutable board/card definitions (shared across games)
- Efficient property lookup with dictionaries
- Minimal allocations in hot paths
- Type hints enable potential Cython/mypyc compilation

## Roadmap

### Phase 1: Core Engine 🚧 (Current - 67% Complete)

**Completed (Weeks 1-4)**:
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
- ✅ 345 tests, 82% coverage
- ✅ Full type safety (mypy strict mode)

**Remaining (Weeks 5-6)**:
- ⏳ Integration tests and validation against legacy implementation (Week 5)
- ⏳ CLI runner for manual testing (Week 5)
- ⏳ Performance benchmarking and optimization (Week 6)
- ⏳ Documentation and polish (Week 6)

**Status**: 10/10 core modules complete, 1,801 lines of production code

### Phase 2: Gymnasium Integration (Next)
- OpenAI Gym/Gymnasium wrapper
- Observation space definition
- Action space encoding
- Reward shaping for RL

### Phase 3: Web Backend
- FastAPI REST API
- WebSocket for real-time updates
- Game session management
- Persistent storage

### Phase 4: Frontend
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

**Status**: Phase 1 - 67% Complete (Weeks 1-4 Done, Weeks 5-6 Remaining)
**Version**: 0.4.0 (Week 4)
**Last Updated**: 2026-02-02
