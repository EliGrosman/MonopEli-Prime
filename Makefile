.PHONY: up down build logs test-backend test-frontend lint clean prod prod-down

# Development
up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

logs:
	docker compose logs -f

# Production
prod:
	docker compose -f docker-compose.prod.yml up -d --build

prod-down:
	docker compose -f docker-compose.prod.yml down --remove-orphans

prod-logs:
	docker compose -f docker-compose.prod.yml logs -f

# Monitoring (add to production)
monitoring:
	docker compose -f docker-compose.prod.yml -f docker-compose.monitoring.yml up -d --build

monitoring-down:
	docker compose -f docker-compose.prod.yml -f docker-compose.monitoring.yml down

# Testing
test-backend:
	uv run python -m pytest

test-frontend:
	cd frontend && npm test -- --run

test-e2e:
	cd frontend && PLAYWRIGHT_BASE_URL=http://localhost:8080 npx playwright test --project=chromium

lint:
	uv run python -m ruff check monopoly_engine api
	uv run python -m mypy monopoly_engine --strict
	cd frontend && npm run lint

# Cleanup
clean:
	docker compose down -v --rmi local
	docker compose -f docker-compose.prod.yml down -v --rmi local 2>/dev/null || true
	rm -rf htmlcov .coverage .mypy_cache .pytest_cache
	rm -rf frontend/dist frontend/playwright-report frontend/test-results
