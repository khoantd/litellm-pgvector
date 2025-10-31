#!/usr/bin/env bash

set -euo pipefail

# Build and optionally push the backend API image (no admin UI)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

IMAGE_NAME=${IMAGE_NAME:-vector-store-api}
IMAGE_TAG=${IMAGE_TAG:-latest}
PLATFORM=${PLATFORM:-linux/amd64}
PUSH=${PUSH:-false}
# Optional toggles
NO_PROVENANCE=${NO_PROVENANCE:-false}  # when true, appends --provenance=false to buildx
PUSH_MODE=${PUSH_MODE:-buildx}        # 'buildx' (default) or 'engine' for --load + docker push

echo "[Backend API] Building ${IMAGE_NAME}:${IMAGE_TAG} for platform ${PLATFORM}..."

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
        -f Dockerfile.backend \
        --load \
        .
      docker push "${IMAGE_NAME}:${IMAGE_TAG}"
    else
      # Optionally disable provenance/attestations to avoid EOF issues on some registries
      if [ "$NO_PROVENANCE" = "true" ]; then
        docker buildx build \
          --platform "$PLATFORM" \
          -t "${IMAGE_NAME}:${IMAGE_TAG}" \
          -f Dockerfile.backend \
          --push \
          --provenance=false \
          .
      else
        docker buildx build \
          --platform "$PLATFORM" \
          -t "${IMAGE_NAME}:${IMAGE_TAG}" \
          -f Dockerfile.backend \
          --push \
          .
      fi
    fi
  else
    # --load imports the image into the local docker daemon (handy on macOS)
    docker buildx build \
      --platform "$PLATFORM" \
      -t "${IMAGE_NAME}:${IMAGE_TAG}" \
      -f Dockerfile.backend \
      --load \
      .
  fi
else
  echo "docker buildx not available; building with classic docker build (host arch)." >&2
  docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f Dockerfile.backend .
fi

echo "Done. Container listens on 8003 internally (uvicorn)."


