#!/usr/bin/env bash
set -euo pipefail

# 进入脚本所在目录，确保环境安装在脚本目录下
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 统一使用 uv 管理环境，避免依赖系统 pip 模块
if ! command -v uv >/dev/null 2>&1; then
  if command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  elif command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  else
    echo "ERROR: wget/curl 均不可用，无法安装 uv" >&2
    exit 1
  fi
fi

# 补充 uv 常见安装路径，兼容新开终端未加载 profile 的场景
export PATH="$HOME/.local/bin:$PATH"

# 在当前目录创建并复用 .venv，确保虚拟环境位置稳定
uv venv .venv

# 在当前目录的 .venv 中安装固定依赖集合（与现有库保持一致）
uv pip install \
  akshare \
  vectorbt \
  pandas \
  numpy \
  matplotlib \
  plotly
