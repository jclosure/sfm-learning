#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/images"
  exit 1
fi

IMAGE_DIR="$1"
mkdir -p outputs
sfm-learning "$IMAGE_DIR" -o outputs/sparse.ply

echo "Wrote outputs/sparse.ply"
