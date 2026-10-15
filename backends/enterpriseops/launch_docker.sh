#!/usr/bin/env bash
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${BACKEND_DIR}/../.." && pwd)"

if [[ $# -ne 6 ]]; then
  echo "usage: $0 DOMAIN IMAGE CONTAINER_NAME HOST_PORT CONTAINER_PORT CONFIGURED_DIGEST" >&2
  exit 2
fi

DOMAIN=$1
IMAGE=$2
CONTAINER_NAME=$3
HOST_PORT=$4
CONTAINER_PORT=$5
CONFIGURED_DIGEST=$6

declare -A COMPAT_DIGESTS=(
  [csm]="sha256:eaa456ac9aa85728426e7d3813a0bbca0949d6a8695be30e26f03894e6e6b189"
  [itsm]="sha256:a234ae3fb7cee196ba25e6b9957969dea829919b6e8271dddae128f065aaf39f"
  [hr]="sha256:1ea1c1d64d4be35e8062e56f00b8318e9e6c09289cfa56bcfd0595bfa59ac64d"
  [email]="sha256:69c2081fe4ab0962b86233f9fb52b307b8ad0019f6746ba64ce75851036201cd"
  [drive]="sha256:3475962fcf6da7675e194dbf138de01fa3e96134a302ad47316e4111a5e63f32"
  [teams]="sha256:602655e46f6501885540c36dc9b12114cb173c75063d7f25c17ed0652695fa78"
  [calendar]="sha256:994c5421a6dd065861bc7f813a177f6d408875e9df60fe8d012959bc4510da02"
)
declare -A COMPAT_SCRIPTS=(
  [csm]="${BACKEND_DIR}/bootstrap_csm.py"
  [itsm]="${BACKEND_DIR}/bootstrap_itsm.py"
  [hr]="${BACKEND_DIR}/bootstrap_hr.py"
  [email]="${BACKEND_DIR}/bootstrap_email.py"
  [drive]="${BACKEND_DIR}/bootstrap_drive.py"
  [teams]="${BACKEND_DIR}/bootstrap_teams.py"
  [calendar]="${BACKEND_DIR}/bootstrap_calendar.py"
)

if [[ ! "${CONFIGURED_DIGEST}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "error: ${DOMAIN} MCP configured digest is invalid: ${CONFIGURED_DIGEST:-missing}" >&2
  exit 2
fi
if ! docker image inspect --format '{{json .RepoDigests}}' "${IMAGE}" \
  | grep -Fq "@${CONFIGURED_DIGEST}"; then
  echo "error: local ${DOMAIN} Docker image does not prove manifest ${CONFIGURED_DIGEST}" >&2
  exit 2
fi

if [[ -n "${COMPAT_DIGESTS[${DOMAIN}]:-}" ]]; then
  COMPAT_DIGEST="${COMPAT_DIGESTS[${DOMAIN}]}"
  COMPAT_SCRIPT="${COMPAT_SCRIPTS[${DOMAIN}]}"
  if [[ "${CONFIGURED_DIGEST}" != "${COMPAT_DIGEST}" ]]; then
    echo "error: ${DOMAIN} MCP compatibility only supports ${COMPAT_DIGEST}" >&2
    exit 2
  fi
  if [[ ! -f "${COMPAT_SCRIPT}" ]]; then
    echo "error: ${DOMAIN} MCP compatibility script is missing: ${COMPAT_SCRIPT}" >&2
    exit 2
  fi
  if [[ ! -f "${BACKEND_DIR}/backend_errors.py" ]]; then
    echo "error: shared backend error contract is missing" >&2
    exit 2
  fi
  exec docker run --rm -d \
    -v "${BACKEND_DIR}:/robustops_backend_compat:ro" \
    -e PYTHONPATH=/robustops_backend_compat \
    --name "${CONTAINER_NAME}" \
    -p "127.0.0.1:${HOST_PORT}:${CONTAINER_PORT}" \
    "${IMAGE}" python -c "$(<"${COMPAT_SCRIPT}")"
fi

exec docker run --rm -d \
  --name "${CONTAINER_NAME}" \
  -p "127.0.0.1:${HOST_PORT}:${CONTAINER_PORT}" \
  "${IMAGE}"
