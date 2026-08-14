#!/usr/bin/env bash
set -euo pipefail

# 使用独立 CPU 环境，避免修改 MuJoCo 机器的系统 Python。
ENV_DIR="${1:-/root/taili_sim2sim_env}"
BASE_PYTHON="${BASE_PYTHON:-/opt/conda/bin/python}"

"${BASE_PYTHON}" -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/pip" install --no-cache-dir --upgrade pip setuptools wheel
"${ENV_DIR}/bin/pip" install --no-cache-dir typing-extensions==4.15.0
"${ENV_DIR}/bin/pip" install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.7.0
"${ENV_DIR}/bin/pip" install --no-cache-dir mujoco==3.4.0 pyyaml
"${ENV_DIR}/bin/python" -c \
  'import torch, mujoco, yaml; print(torch.__version__, mujoco.__version__)'
touch "${ENV_DIR}/.ready"
