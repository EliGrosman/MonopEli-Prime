# Repository Guidelines

## Project Structure & Module Organization

- `monopoly_engine/`: core Monopoly rules and state; keep this layer pure Python with no I/O.
- `monopoly_gym/`: Gymnasium/PettingZoo environments; `agents/`, `training/`, and `mcts/`: AI policies, reinforcement learning, tree search, and LLM trading.
- `api/`: FastAPI routes, lobby management, and WebSocket services.
- `frontend/src/`: React/TypeScript components, hooks, Zustand stores, types, and utilities. Static assets live in `frontend/public/`.
- `tests/`: Python tests, including `tests/api/`; frontend unit tests sit beside source files, and browser tests live in `frontend/e2e/`.
- `scripts/`: training, evaluation, and benchmark entry points; `deploy/`: infrastructure configuration.

## Build, Test, and Development Commands

Run from the repository root unless noted:

- `uv sync --dev --extra training`: install Python development and training dependencies, matching backend CI.
- `uv run uvicorn api.main:app --reload`: start the local API.
- In `frontend/`, run `npm ci`, then `npm run dev` to start Vite; `npm run build` checks TypeScript and builds production assets.
- `make up` / `make down`: start/stop the Docker development stack; `make build` rebuilds images.
- `make test-backend` / `make test-frontend`: run Python or frontend unit tests.
- `make lint`: run Ruff, strict engine mypy checks, and frontend ESLint.
- In `frontend/`, run `npm run format:check` and `npm run type-check` for additional CI checks.

## Coding Style & Naming Conventions

Use four-space Python indentation, snake_case functions/modules, PascalCase classes, and type annotations. Ruff enforces a 100-character line limit and import/naming rules; mypy is configured for strict checking.

Use two-space TypeScript indentation, PascalCase component filenames, and `use*` hook names. Follow frontend Prettier configuration and ESLint rules; run `npm run format` from `frontend/` to format source files.

## Testing Guidelines

Use pytest, Hypothesis, and pytest-asyncio for Python tests; name files/functions `test_*` and classes `Test*`. Reuse `conftest.py` fixtures and seeded game scenarios. Pytest generates coverage reports; no minimum coverage threshold is configured.

Use Vitest/React Testing Library for colocated `*.test.ts(x)` files. Run Playwright browser tests with `npm run test:e2e` in `frontend/`; name them `*.e2e.ts`. Add regression tests for changed behavior.

## Commit & Pull Request Guidelines

History uses short, descriptive subjects such as `Fix LLM response parsing`; no consistent Conventional Commits prefix is established. Write focused, action-oriented subjects.

PRs should describe the problem, changes, and validation commands/results; link relevant issues and include screenshots for visible UI changes. Ensure applicable CI checks pass.

## Configuration

Copy `.env.example` to `.env` for local configuration. Keep credentials and API keys out of commits.
