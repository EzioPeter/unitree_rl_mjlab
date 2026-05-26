# Installation Guide

## System Requirements

- Operating system: Ubuntu 22.04 or 24.04
- GPU: NVIDIA GPU
- Driver: 550 or newer recommended

## 1. Install uv

This project uses uv for the Python environment.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Pin a Python version:

```bash
uv python pin 3.10.18
```

For Ubuntu 24.04 / Blackwell GPUs, Python 3.11 is also supported:

```bash
uv python pin 3.11.14
```

## 2. Install System Dependencies

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

## 3. Sync Python Dependencies

From the project root:

```bash
uv sync
```

Run commands through uv:

```bash
uv run python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

FlashSAC on G1 uses the same uv environment:

```bash
uv run python scripts/train_flashsac.py
```
