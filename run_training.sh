#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/src"

echo "=== ResNet18 line: train_resnet.py ==="
python3 train_resnet.py "$@"

echo "=== Scratch line: train.py ==="
python3 train.py
