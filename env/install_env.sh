#!/usr/bin/env bash
set -euo pipefail

# 进入脚本所在目录，确保环境安装在脚本目录下
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 统一使用 Poetry 管理依赖，确保与pyproject.toml一致
if ! command -v poetry >/dev/null 2>&1; then
  if command -v wget >/dev/null 2>&1; then
    wget -qO- https://install.python-poetry.org | python3 -
  elif command -v curl >/dev/null 2>&1; then
    curl -sSL https://install.python-poetry.org | python3 -
  else
    echo "ERROR: wget/curl 均不可用，无法安装 Poetry" >&2
    exit 1
  fi
fi

# 补充 Poetry 常见安装路径，兼容新开终端未加载 profile 的场景
export PATH="$HOME/.local/bin:$PATH"

# 固定虚拟环境到项目目录下的 .venv，保持路径稳定
poetry config virtualenvs.in-project true --local

# 先生成/更新锁文件，再按锁文件安装依赖，保证可复现
poetry lock --no-interaction
poetry install --no-interaction --no-ansi
