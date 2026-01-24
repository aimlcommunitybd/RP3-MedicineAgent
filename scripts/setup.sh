#!/bin/bash
set -e

echo "Installing pipx globally (if not installed)..."
sudo apt-get update -y
sudo apt install pipx
pipx ensurepath

echo "Installing uv CLI via pipx (if not installed)..."
pipx install uv || echo "uv might already be installed or pipx missing"

echo "Installing Python packages..."
uv sync
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# uv pip install "transformers[torch]"

echo "Installing global dependencies..."
sudo apt-get install -y git-lfs
git lfs install

echo "Setup complete!"
