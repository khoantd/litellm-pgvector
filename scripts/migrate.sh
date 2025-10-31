#!/usr/bin/env bash

set -euo pipefail

# Run Prisma database migrations (db push)
# Usage: ./scripts/migrate.sh
# Requires: DATABASE_URL environment variable

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

echo "Running Prisma migrations..."
echo "Database: ${DATABASE_URL:-not set}"

if [ -z "${DATABASE_URL:-}" ]; then
  echo "Error: DATABASE_URL environment variable is not set" >&2
  exit 1
fi

# Generate Prisma client first
echo "[1/2] Generating Prisma client..."
prisma generate

# Push schema to database
echo "[2/2] Pushing schema to database..."
prisma db push --accept-data-loss

echo "Migration complete!"

