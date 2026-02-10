# MonopEli

A full-stack Monopoly platform: pure Python game engine, reinforcement learning training, FastAPI backend with WebSocket, and React frontend — all containerized and production-ready.

## Quick Start

### Docker (recommended)

```bash
cp .env.example .env
make up

# Backend:  http://localhost:8000
# Frontend: http://localhost:3000
```

### Production (single-port Nginx reverse proxy)

```bash
make prod
# Everything on http://localhost:8080
```

### Local Development

```bash
# Install uv (https://github.com/astral-sh/uv)
curl -LsSf https://astral.sh/uv/install.sh | sh

uv sync                              # install dependencies
uv run python -m pytest              # run tests
uv run uvicorn api.main:app --reload # start API server
cd frontend && npm install && npm run dev  # start frontend
```

## Architecture

```
MonopEli-Prime/
├── monopoly_engine/     Core game engine — pure logic, zero I/O
├── monopoly_gym/        PettingZoo + Gymnasium RL environments
├── agents/              AI agents (random, rule-based, aggressive, conservative)
├── training/            MaskablePPO training, curriculum learning, self-play
├── api/                 FastAPI backend with WebSocket & lobby system
├── frontend/            React 19 + TypeScript + Tailwind CSS
├── deploy/              Nginx, Prometheus, Grafana configs
├── tests/               ~1,200 backend tests
├── scripts/             CLI tools, benchmarks, training scripts
└── .github/workflows/   CI/CD (multi-Python, Docker, Trivy scan)
```

## Game Engine

The core `monopoly_engine/` package implements the complete Monopoly ruleset with no external dependencies beyond the standard library.

**Features:**
- All 40 board spaces, 32 Chance/Community Chest cards, 15 action types
- Property purchase, rent, mortgage, building (even-build rule enforced)
- Jail mechanics, bankruptcy, player elimination, winner determination
- Full JSON serialization for save/load
- Deterministic replay with random seed support
- Type-safe — passes `mypy --strict` with zero errors

**Performance:** 1,234 games/second (single-threaded)

```python
from monopoly_engine import MonopolyGame, GameConfig

config = GameConfig(num_players=4, starting_money=1500)
game = MonopolyGame(config)

game.roll_dice()
game.move_current_player()
game.handle_landing()

state_json = game.to_json()
```

### Modules

| Module | Purpose |
|--------|---------|
| `types.py` | Enums, TypedDicts, Protocols |
| `board.py` | Immutable board definition (40 spaces) |
| `property.py` | Property ownership, mortgages, buildings |
| `player.py` | Player state and inventory |
| `cards.py` | Chance & Community Chest cards |
| `rules.py` | Rent calculation, building rules, game rules |
| `actions.py` | All 15 action types with validate-then-execute |
| `state.py` | Game state container with serialization |
| `game.py` | Main game orchestrator |
| `exceptions.py` | Custom exception hierarchy |

## RL Training

The `monopoly_gym/` package wraps the engine as standard RL environments.

### Environments

- **`MonopolyEnv`** — PettingZoo AEC multi-agent environment
- **`SingleAgentMonopolyEnv`** — Gymnasium wrapper for Stable-Baselines3

Both provide a 149-dimensional action space with action masking (only valid moves are selectable) and a dict observation space encoding the full game state.

### Training an Agent

```bash
uv sync --extra training

# Train PPO against random opponents
uv run python scripts/train_ppo.py --timesteps 100000

# Evaluate
uv run python scripts/evaluate_agent.py --model models/ppo_monopoly --games 100

# Self-play
uv run python scripts/train_self_play.py --timesteps 500000
```

### Curriculum Learning

4-stage progressive difficulty: random → rule-based → conservative → aggressive opponents.

```python
from training import CurriculumConfig, CurriculumTrainer

config = CurriculumConfig(
    stages=[
        ("random", 100_000, 0.8),
        ("rule_based", 200_000, 0.6),
        ("aggressive", 300_000, 0.5),
    ],
)
trainer = CurriculumTrainer(config)
model = trainer.train()
```

### Results

| Metric | Value |
|--------|-------|
| Win rate vs random | **94%** |
| Win rate vs rule-based | 22% (2.8x random baseline) |
| Training throughput | ~1,000 steps/sec |

## API Server

FastAPI backend with real-time WebSocket communication.

```bash
# Development
uv run uvicorn api.main:app --reload --port 8000

# Production
uv run uvicorn api.main:create_app --factory --workers 4 --port 8000

# API docs: http://localhost:8000/api/docs
```

### Endpoints

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
| `/ws/games/{id}` | WS | Real-time game updates |

### Key Capabilities

- **WebSocket** with heartbeat and auto-reconnect
- **Lobby system** — public/private rooms, invite codes, AI player slots, ready states
- **AI opponents** — 4 agent types with configurable think delay
- **Player sessions** — reconnection support with disconnect tracking
- **Middleware** — request logging with IDs, rate limiting (token bucket), Prometheus metrics, standardized error responses

## Frontend

React 19 single-page application with real-time game board.

```bash
cd frontend
npm install
npm run dev    # http://localhost:5173
```

### Stack

React 19 · TypeScript 5.9 · Vite 7 · Tailwind CSS 4 · Zustand 5 · React Router 7 · Axios · Vitest · Playwright

### Features

- Interactive board with property ownership visualization
- Player panel with net worth tracking
- Action panel for all game moves (buy, build, mortgage, trade)
- Animated dice display
- Lobby system with room creation, settings, and AI slots
- Game event log
- Responsive design with mobile controls
- Keyboard navigation and ARIA accessibility
- WebSocket auto-reconnect with connection status indicator

## Deployment

### Docker Compose Profiles

| Command | What it runs |
|---------|-------------|
| `make up` | Dev: backend (8000) + frontend (3000) |
| `make prod` | Production: Nginx reverse proxy (8080) |
| `make monitoring` | Production + Prometheus (9090) + Grafana (3001) |
| `make clean` | Remove containers, volumes, caches |

### Monitoring

Prometheus scrapes `/metrics` from the backend. Grafana ships with a pre-built dashboard (auto-provisioned).

```bash
make monitoring
# Grafana: http://localhost:3001 (admin/admin)
```

### CI/CD

GitHub Actions workflows:
- **ci.yml** — lint (ruff, mypy, ESLint), test (Python 3.11/3.12/3.13), coverage (Codecov), Trivy scan
- **docker.yml** — build images, E2E tests, push to GHCR

## Development

### Requirements

- Python 3.11+ (3.13 recommended)
- Node 22+ (for frontend)
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Docker & Docker Compose (for containerized workflow)

### Common Commands

```bash
# Tests
make test-backend               # Python tests with coverage
make test-frontend              # Vitest
make test-e2e                   # Playwright (requires running prod stack)

# Quality
make lint                       # ruff + mypy + ESLint

# Manual
uv run python -m pytest --cov=monopoly_engine --cov-report=html
uv run python -m mypy monopoly_engine --strict
uv run python -m ruff check monopoly_engine api
uv run python scripts/benchmark.py          # performance benchmark
uv run python scripts/cli_runner.py         # interactive CLI game
```

### Adding Dependencies

```bash
uv add package-name              # runtime
uv add --dev package-name        # dev
uv sync                          # install everything
```

## Configuration

All backend settings use the `MONOPELI_` prefix. Copy `.env.example` to `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `MONOPELI_HOST` | `0.0.0.0` | Bind address |
| `MONOPELI_PORT` | `8000` | API port |
| `MONOPELI_DEBUG` | `false` | Debug mode |
| `MONOPELI_LOG_LEVEL` | `INFO` | Log level |
| `MONOPELI_CORS_ORIGINS` | `localhost:3000,5173` | Allowed origins |
| `MONOPELI_MAX_CONCURRENT_GAMES` | `1000` | Game limit |
| `MONOPELI_RATE_LIMIT_PER_MINUTE` | `60` | Rate limit |
| `MONOPELI_AI_THINK_DELAY_MS` | `500` | AI response delay |

See `.env.example` for the full list.

## Testing

| Suite | Tests | Coverage |
|-------|-------|----------|
| Game engine | ~400 | 91% |
| RL environment | ~500 | 92% |
| API backend | 199 | — |
| Training | 51+ | — |
| Frontend (Vitest) | 348 | — |
| Frontend (Playwright) | 30+ scenarios | — |

```bash
# Full backend suite
uv run python -m pytest

# With HTML coverage report
uv run python -m pytest --cov-report=html
open htmlcov/index.html
```

## Project Stats

| Metric | Value |
|--------|-------|
| Backend production code | ~8,000 lines |
| Backend test code | ~15,000 lines |
| Frontend code | ~5,000 lines |
| Game performance | 1,234 games/sec |
| Action types | 15 |
| Board spaces | 40 |
| Action dimensions | 149 |
| Agent types | 4 |
| API endpoints | 9 REST + 1 WebSocket |
| Python versions tested | 3.11, 3.12, 3.13 |

## License

[Add license information]

---

Built as part of the MonopEli project — Monopoly for AI training and web gaming.
