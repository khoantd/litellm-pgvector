#!/usr/bin/env bash

set -euo pipefail

# Build script for production Docker image
# - Builds the admin UI
# - Ensures static assets are placed under static/admin for FastAPI
# - Builds the multi-stage Docker image

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

IMAGE_NAME=${IMAGE_NAME:-vector-store-api}
IMAGE_TAG=${IMAGE_TAG:-latest}
PLATFORM=${PLATFORM:-linux/amd64}
PUSH=${PUSH:-false}
# Optional toggles
NO_PROVENANCE=${NO_PROVENANCE:-false}  # when true, appends --provenance=false to buildx
PUSH_MODE=${PUSH_MODE:-buildx}        # 'buildx' (default) or 'engine' for --load + docker push

echo "[1/3] Building Admin UI..."
pushd admin-ui >/dev/null
if command -v npm >/dev/null 2>&1; then
  npm ci || npm i
  npm run build
else
  echo "npm not found. Please install Node.js/npm." >&2
  exit 1
fi
popd >/dev/null

echo "[2/3] Staging static assets..."
mkdir -p static/admin
rsync -a --delete admin-ui/dist/ static/admin/

echo "[3/3] Building Docker image ${IMAGE_NAME}:${IMAGE_TAG} for platform ${PLATFORM}..."

if docker buildx version >/dev/null 2>&1; then
  # Ensure a builder exists
  BUILDER_NAME=${BUILDER_NAME:-vsapi_builder}
  if ! docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
    docker buildx create --name "$BUILDER_NAME" >/dev/null
  fi
  docker buildx use "$BUILDER_NAME" >/dev/null

  if [ "$PUSH" = "true" ]; then
    # Buildx direct push (default) or fallback to engine push
    if [ "$PUSH_MODE" = "engine" ]; then
      # Build into local daemon, then push via classic engine
      docker buildx build \
        --platform "$PLATFORM" \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        --load \
        .
      docker push "${IMAGE_NAME}:${IMAGE_TAG}"
    else
      # Optionally disable provenance/attestations to avoid EOF issues on some registries
      if [ "$NO_PROVENANCE" = "true" ]; then
        docker buildx build \
          --platform "$PLATFORM" \
          -t "${IMAGE_NAME}:${IMAGE_TAG}" \
          --push \
          --provenance=false \
          .
      else
        docker buildx build \
          --platform "$PLATFORM" \
          -t "${IMAGE_NAME}:${IMAGE_TAG}" \
          --push \
          .
      fi
    fi
  else
    # --load imports the image into the local docker daemon (handy on macOS)
    docker buildx build \
      --platform "$PLATFORM" \
      -t "${IMAGE_NAME}:${IMAGE_TAG}" \
      --load \
      .
  fi
else
  echo "docker buildx not available; building with classic docker build (host arch)." >&2
  docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .
fi

echo "Done. To run:"
echo "  docker run --rm -p 8000:8003 --env-file .env ${IMAGE_NAME}:${IMAGE_TAG}"
echo "Note: Container listens on 8003 internally (Dockerfile CMD)."


