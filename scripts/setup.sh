#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROBUSTOPS_PYTHON:-python}"
cd "${ROOT_DIR}"

"${PYTHON_BIN}" scripts/setup_dependencies.py
# Keep PyPI available for dependencies absent from the CUDA wheel index.
# Install the CUDA wheels before resolving the training stack.
"${PYTHON_BIN}" -m pip install -c requirements.txt \
  --index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://pypi.org/simple torch torchvision torchaudio
"${PYTHON_BIN}" -m pip install -r requirements.txt -e './verl[vllm]' -e '.[test,plot]'
# The FlashAttention wheel contains CUDA targets sm80 and sm120.
FLASH_WHEEL='flash_attn-2.8.3+cu130.torch2.11.abi1.sm80sm120-cp312-cp312-linux_x86_64.whl'
FLASH_SHA256='12154159f06ec92f6bd6c120b915a3f839b4ddccf910319c8ba63e869b4a8d82'
FLASH_RELEASE='flash-attn-2.8.3-cu130-torch2.11-cp312'
FLASH_URL="https://github.com/haohuiliang/RobustOps/releases/download/${FLASH_RELEASE}/${FLASH_WHEEL}"
FLASH_DOWNLOAD_DIR="$(mktemp -d)"
trap 'rm -rf -- "${FLASH_DOWNLOAD_DIR}"' EXIT

if ! "${PYTHON_BIN}" -m pip download --no-deps \
  --dest "${FLASH_DOWNLOAD_DIR}" "${FLASH_URL}#sha256=${FLASH_SHA256}"; then
  if ! command -v gh >/dev/null 2>&1; then
    printf 'FlashAttention wheel download failed. Check release access and rerun setup.\n' >&2
    exit 1
  fi
  rm -f -- "${FLASH_DOWNLOAD_DIR}/${FLASH_WHEEL}"
  env -u DEBUG -u GH_DEBUG gh release download "${FLASH_RELEASE}" \
    --repo haohuiliang/RobustOps --pattern "${FLASH_WHEEL}" \
    --dir "${FLASH_DOWNLOAD_DIR}"
fi

"${PYTHON_BIN}" - "${FLASH_DOWNLOAD_DIR}/${FLASH_WHEEL}" "${FLASH_SHA256}" <<'PY'
import hashlib
import sys

with open(sys.argv[1], 'rb') as wheel:
    actual = hashlib.file_digest(wheel, 'sha256').hexdigest()
if actual != sys.argv[2]:
    raise SystemExit('FlashAttention wheel SHA256 mismatch; installation aborted.')
print('FlashAttention wheel SHA256 verified.')
PY
"${PYTHON_BIN}" -m pip install --no-deps -c requirements.txt "${FLASH_DOWNLOAD_DIR}/${FLASH_WHEEL}"
"${PYTHON_BIN}" -c 'import flash_attn; import flash_attn_2_cuda; print("FlashAttention", flash_attn.__version__, "import OK")'
