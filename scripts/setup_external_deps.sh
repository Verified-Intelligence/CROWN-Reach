#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT_DIR}/external"
FLOWSTAR_DIR="${ROOT_DIR}/flowstar"

mkdir -p "${EXTERNAL_DIR}"

if [[ ! -d "${FLOWSTAR_DIR}/flowstar-toolbox" ]]; then
  echo "[setup] Cloning Flow* into ${FLOWSTAR_DIR}"
  git clone https://github.com/chenxin415/flowstar.git "${FLOWSTAR_DIR}"
else
  echo "[setup] Flow* already exists at ${FLOWSTAR_DIR}"
fi

if [[ ! -d "${EXTERNAL_DIR}/auto_LiRPA/.git" ]]; then
  echo "[setup] Cloning auto_LiRPA into ${EXTERNAL_DIR}/auto_LiRPA"
  git clone https://github.com/Verified-Intelligence/auto_LiRPA.git "${EXTERNAL_DIR}/auto_LiRPA"
else
  echo "[setup] auto_LiRPA already exists at ${EXTERNAL_DIR}/auto_LiRPA"
fi

echo "[setup] Building Flow*"
make -C "${FLOWSTAR_DIR}/flowstar-toolbox"

echo "[setup] Installing Python dependencies"
pip install -U pip setuptools wheel
pip install -r "${ROOT_DIR}/requirements.txt"
pip install -e "${EXTERNAL_DIR}/auto_LiRPA"

echo "[setup] Done"
