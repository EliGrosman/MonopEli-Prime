#!/bin/bash
set -e

echo "=== MonopEli Deployment ==="

# Pull latest code
echo "Pulling latest code..."
git pull origin main

# Build and start services
echo "Building and starting services..."
docker compose -f docker-compose.prod.yml -f docker-compose.monitoring.yml build
docker compose -f docker-compose.prod.yml -f docker-compose.monitoring.yml up -d

# Wait for backend to be healthy
echo "Waiting for backend health check..."
for i in $(seq 1 30); do
    if docker compose -f docker-compose.prod.yml exec -T backend python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" 2>/dev/null; then
        echo "Backend is healthy!"
        break
    fi
    if [ "$i" -eq 30 ]; then
        echo "ERROR: Backend failed to start"
        docker compose -f docker-compose.prod.yml -f docker-compose.monitoring.yml logs backend
        exit 1
    fi
    echo "  Waiting... ($i/30)"
    sleep 2
done

echo "=== Deployed successfully ==="
docker compose -f docker-compose.prod.yml -f docker-compose.monitoring.yml ps
