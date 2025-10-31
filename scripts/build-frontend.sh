#!/usr/bin/env bash

set -euo pipefail

# Build and optionally push the Admin UI image (Nginx static)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR/admin-ui"

IMAGE_NAME=${IMAGE_NAME:-vector-store-admin}
IMAGE_TAG=${IMAGE_TAG:-latest}
PLATFORM=${PLATFORM:-linux/amd64}
PUSH=${PUSH:-false}
# Optional: set backend API base URL (e.g., "https://api.example.com" or "" for relative/same-domain)
VITE_API_BASE_URL=${VITE_API_BASE_URL:-}
# Optional toggles
NO_PROVENANCE=${NO_PROVENANCE:-false}  # when true, appends --provenance=false to buildx
PUSH_MODE=${PUSH_MODE:-buildx}        # 'buildx' (default) or 'engine' for --load + docker push

echo "[Admin UI] Building ${IMAGE_NAME}:${IMAGE_TAG} for platform ${PLATFORM}..."
if [ -n "$VITE_API_BASE_URL" ]; then
  echo "  Using API base: ${VITE_API_BASE_URL}"
else
  echo "  Using relative URLs (same domain routing)"
fi

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
      if [ -n "$VITE_API_BASE_URL" ]; then
        docker buildx build \
          --platform "$PLATFORM" \
          --build-arg "VITE_API_BASE_URL=${VITE_API_BASE_URL}" \
          -t "${IMAGE_NAME}:${IMAGE_TAG}" \
          -f Dockerfile \
          --load \
          .
      else
        docker buildx build \
          --platform "$PLATFORM" \
          -t "${IMAGE_NAME}:${IMAGE_TAG}" \
          -f Dockerfile \
          --load \
          .
      fi
      docker push "${IMAGE_NAME}:${IMAGE_TAG}"
    else
      # Optionally disable provenance/attestations to avoid EOF issues on some registries
      if [ "$NO_PROVENANCE" = "true" ]; then
        if [ -n "$VITE_API_BASE_URL" ]; then
          docker buildx build \
            --platform "$PLATFORM" \
            --build-arg "VITE_API_BASE_URL=${VITE_API_BASE_URL}" \
            -t "${IMAGE_NAME}:${IMAGE_TAG}" \
            -f Dockerfile \
            --push \
            --provenance=false \
            .
        else
          docker buildx build \
            --platform "$PLATFORM" \
            -t "${IMAGE_NAME}:${IMAGE_TAG}" \
            -f Dockerfile \
            --push \
            --provenance=false \
            .
        fi
      else
        if [ -n "$VITE_API_BASE_URL" ]; then
          docker buildx build \
            --platform "$PLATFORM" \
            --build-arg "VITE_API_BASE_URL=${VITE_API_BASE_URL}" \
            -t "${IMAGE_NAME}:${IMAGE_TAG}" \
            -f Dockerfile \
            --push \
            .
        else
          docker buildx build \
            --platform "$PLATFORM" \
            -t "${IMAGE_NAME}:${IMAGE_TAG}" \
            -f Dockerfile \
            --push \
            .
        fi
      fi
    fi
  else
    # --load imports the image into the local docker daemon (handy on macOS)
    if [ -n "$VITE_API_BASE_URL" ]; then
      docker buildx build \
        --platform "$PLATFORM" \
        --build-arg "VITE_API_BASE_URL=${VITE_API_BASE_URL}" \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        -f Dockerfile \
        --load \
        .
    else
      docker buildx build \
        --platform "$PLATFORM" \
        -t "${IMAGE_NAME}:${IMAGE_TAG}" \
        -f Dockerfile \
        --load \
        .
    fi
  fi
else
  echo "docker buildx not available; building with classic docker build (host arch)." >&2
  if [ -n "$VITE_API_BASE_URL" ]; then
    docker build --build-arg VITE_API_BASE_URL="${VITE_API_BASE_URL}" -t "${IMAGE_NAME}:${IMAGE_TAG}" -f Dockerfile .
  else
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -f Dockerfile .
  fi
fi

echo "Done. Serve at /admin on port 3006 in the container."


