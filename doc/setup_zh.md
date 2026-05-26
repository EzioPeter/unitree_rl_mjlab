# 安装配置文档

## 系统要求

- 操作系统：推荐 Ubuntu 22.04 或 24.04
- 显卡：NVIDIA GPU
- 驱动版本：建议 550 或更高

## 1. 安装 uv

本项目使用 uv 管理 Python 环境。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

固定 Python 版本：

```bash
uv python pin 3.10.18
```

如果是 Ubuntu 24.04 / Blackwell GPU，也可以使用 Python 3.11：

```bash
uv python pin 3.11.14
```

## 2. 安装系统依赖

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
```

## 3. 同步 Python 依赖

在项目根目录执行：

```bash
uv sync
```

之后所有训练、回放命令都通过 uv 运行：

```bash
uv run python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

G1 上的 FlashSAC 训练也使用同一个 uv 环境：

```bash
uv run python scripts/train_flashsac.py
```
