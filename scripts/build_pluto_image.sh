#!/bin/bash
# Build pluto training image locally on Daniel PC.
# Registry tagging and push are handled in a later step.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOCKER_DIR="$(cd "${SCRIPT_DIR}/../docker" && pwd)"

cd "${DOCKER_DIR}"
docker build -f Dockerfile.pluto -t pluto:0.1.0 .
