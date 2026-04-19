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

# 固定虚拟环境到项目目录下的 .venv，避免 Poetry 继续复用外部共享解释器目录。
VENV_PATH="$SCRIPT_DIR/.venv"
PYTHON_BIN="$(command -v python3 || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "ERROR: 未找到 python3，无法创建项目虚拟环境" >&2
  exit 1
fi

# 先固化本地 Poetry 配置，再显式要求创建虚拟环境。
poetry config virtualenvs.create true --local
poetry config virtualenvs.in-project true --local

# 若 Poetry 当前绑定到项目外环境，则先移除旧绑定，避免 install 继续落到共享解释器目录。
CURRENT_ENV_PATH="$(poetry env info --path 2>/dev/null || true)"
if [ -n "$CURRENT_ENV_PATH" ] && [ "$CURRENT_ENV_PATH" != "$VENV_PATH" ]; then
  poetry env remove --all --no-interaction || true
fi

# 先显式创建项目内 .venv，再把 Poetry 绑定到该解释器，避免继续复用外部共享解释器目录。
if [ ! -x "$VENV_PATH/bin/python" ]; then
  "$PYTHON_BIN" -m venv "$VENV_PATH"
fi
poetry env use "$VENV_PATH/bin/python"

# 先生成/更新锁文件，再按锁文件安装依赖，避免安装 root 包失败。
poetry lock --no-interaction
poetry install --no-root --no-interaction --no-ansi

# 安装完成后强制校验结果路径，确保本次安装真正落在项目内 .venv。
RESOLVED_ENV_PATH="$(poetry env info --path)"
if [ "$RESOLVED_ENV_PATH" != "$VENV_PATH" ]; then
  echo "ERROR: Poetry 环境路径异常，期望: $VENV_PATH，实际: $RESOLVED_ENV_PATH" >&2
  exit 1
fi
if [ ! -x "$VENV_PATH/bin/python" ]; then
  echo "ERROR: 项目虚拟环境创建失败，缺少可执行文件: $VENV_PATH/bin/python" >&2
  exit 1
fi

echo "Poetry 环境安装完成: $RESOLVED_ENV_PATH"
echo "Python 可执行文件: $VENV_PATH/bin/python"
