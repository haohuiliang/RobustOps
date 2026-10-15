#!/usr/bin/env bash
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${BACKEND_DIR}/../.." && pwd)"
RUNTIME_ROOT="${ROBUSTOPS_MCP_RUNTIME_ROOT:-${ROBUSTOPS_EMAIL_MCP_RUNTIME_ROOT:-${ROOT_DIR}/.runtime/enterpriseops_mcp}}"
UDOCKER_BIN="${RUNTIME_ROOT}/venv/bin/udocker"
export UDOCKER_DIR="${RUNTIME_ROOT}/udocker"
DOMAIN="${1:-}"
declare -A COMPAT_DIGESTS=(
  [csm]="sha256:eaa456ac9aa85728426e7d3813a0bbca0949d6a8695be30e26f03894e6e6b189"
  [itsm]="sha256:a234ae3fb7cee196ba25e6b9957969dea829919b6e8271dddae128f065aaf39f"
  [hr]="sha256:1ea1c1d64d4be35e8062e56f00b8318e9e6c09289cfa56bcfd0595bfa59ac64d"
  [email]="sha256:69c2081fe4ab0962b86233f9fb52b307b8ad0019f6746ba64ce75851036201cd"
  [drive]="sha256:3475962fcf6da7675e194dbf138de01fa3e96134a302ad47316e4111a5e63f32"
  [teams]="sha256:602655e46f6501885540c36dc9b12114cb173c75063d7f25c17ed0652695fa78"
  [calendar]="sha256:994c5421a6dd065861bc7f813a177f6d408875e9df60fe8d012959bc4510da02"
)
declare -A PINNED_DIGESTS=(
  [email]="sha256:69c2081fe4ab0962b86233f9fb52b307b8ad0019f6746ba64ce75851036201cd"
  [calendar]="sha256:994c5421a6dd065861bc7f813a177f6d408875e9df60fe8d012959bc4510da02"
  [drive]="sha256:3475962fcf6da7675e194dbf138de01fa3e96134a302ad47316e4111a5e63f32"
  [csm]="sha256:eaa456ac9aa85728426e7d3813a0bbca0949d6a8695be30e26f03894e6e6b189"
  [hr]="sha256:1ea1c1d64d4be35e8062e56f00b8318e9e6c09289cfa56bcfd0595bfa59ac64d"
  [itsm]="sha256:a234ae3fb7cee196ba25e6b9957969dea829919b6e8271dddae128f065aaf39f"
  [teams]="sha256:602655e46f6501885540c36dc9b12114cb173c75063d7f25c17ed0652695fa78"
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

case "${DOMAIN}" in
  email)
    HOST_PORT="${ROBUSTOPS_MCP_HOST_PORT:-${ROBUSTOPS_EMAIL_MCP_HOST_PORT:-8004}}"
    CONTAINER_PORT=8005
    ;;
  calendar)
    HOST_PORT="${ROBUSTOPS_MCP_HOST_PORT:-${ROBUSTOPS_CALENDAR_MCP_HOST_PORT:-8003}}"
    CONTAINER_PORT=8003
    ;;
  drive)
    HOST_PORT="${ROBUSTOPS_MCP_HOST_PORT:-${ROBUSTOPS_DRIVE_MCP_HOST_PORT:-8009}}"
    CONTAINER_PORT=8005
    ;;
  csm)
    HOST_PORT="${ROBUSTOPS_MCP_HOST_PORT:-${ROBUSTOPS_CSM_MCP_HOST_PORT:-8001}}"
    CONTAINER_PORT=8005
    ;;
  hr)
    HOST_PORT="${ROBUSTOPS_MCP_HOST_PORT:-${ROBUSTOPS_HR_MCP_HOST_PORT:-8008}}"
    CONTAINER_PORT=8005
    ;;
  itsm)
    HOST_PORT="${ROBUSTOPS_MCP_HOST_PORT:-${ROBUSTOPS_ITSM_MCP_HOST_PORT:-8006}}"
    CONTAINER_PORT=8005
    ;;
  teams)
    HOST_PORT="${ROBUSTOPS_MCP_HOST_PORT:-${ROBUSTOPS_TEAMS_MCP_HOST_PORT:-8002}}"
    CONTAINER_PORT=8005
    ;;
  *)
    echo "usage: $0 {calendar|csm|drive|email|hr|itsm|teams}" >&2
    exit 2
    ;;
esac

CONTAINER="robustops-${DOMAIN}-mcp"

if [[ ! -x "${UDOCKER_BIN}" ]]; then
  echo "error: MCP udocker runtime is not prepared; run backends/enterpriseops/prepare_udocker.sh ${DOMAIN}" >&2
  exit 2
fi
if ! "${UDOCKER_BIN}" --allow-root inspect "${CONTAINER}" >/dev/null 2>&1; then
  echo "error: ${DOMAIN} MCP udocker container is missing; run backends/enterpriseops/prepare_udocker.sh ${DOMAIN}" >&2
  exit 2
fi

CONFIGURED_DIGEST="${ROBUSTOPS_MCP_IMAGE_DIGEST:-${PINNED_DIGESTS[${DOMAIN}]}}"
RUNTIME_INFO="${RUNTIME_ROOT}/runtime_info_${DOMAIN}.txt"
if [[ ! -f "${RUNTIME_INFO}" ]]; then
  echo "error: ${DOMAIN} MCP runtime metadata is missing: ${RUNTIME_INFO}" >&2
  exit 2
fi
ACTUAL_DIGEST="$(awk -F= '$1 == "manifest_digest" {print $2}' "${RUNTIME_INFO}")"
if [[ "${ACTUAL_DIGEST}" != "${CONFIGURED_DIGEST}" ]]; then
  echo "error: ${DOMAIN} MCP runtime manifest is ${ACTUAL_DIGEST:-unknown}, expected ${CONFIGURED_DIGEST}" >&2
  exit 2
fi

# Route the two images' internal file handlers outside the container layers.
LOG_ROOT="${ROBUSTOPS_LOG_ROOT:-${ROOT_DIR}/../log-archive}"
LOG_VOLUMES=()
if [[ "${DOMAIN}" == email ]]; then
  mkdir -p "${LOG_ROOT}/mcp/email"
  touch "${LOG_ROOT}/mcp/email/email_api.log"
  LOG_VOLUMES+=("--volume=${LOG_ROOT}/mcp/email/email_api.log:/app/email_api.log")
elif [[ "${DOMAIN}" == csm ]]; then
  mkdir -p "${LOG_ROOT}/mcp/csm"
  LOG_VOLUMES+=("--volume=${LOG_ROOT}/mcp/csm:/app/logs")
fi

if [[ -n "${COMPAT_DIGESTS[${DOMAIN}]:-}" ]]; then
  if [[ "${CONFIGURED_DIGEST}" != "${COMPAT_DIGESTS[${DOMAIN}]}" ]]; then
    echo "error: ${DOMAIN} MCP compatibility only supports ${COMPAT_DIGESTS[${DOMAIN}]}" >&2
    exit 2
  fi
  COMPAT_SCRIPT="${COMPAT_SCRIPTS[${DOMAIN}]}"
  if [[ ! -f "${COMPAT_SCRIPT}" ]]; then
    echo "error: ${DOMAIN} MCP compatibility script is missing: ${COMPAT_SCRIPT}" >&2
    exit 2
  fi
  if [[ ! -f "${BACKEND_DIR}/backend_errors.py" ]]; then
    echo "error: shared backend error contract is missing" >&2
    exit 2
  fi
  COMPAT_BOOTSTRAP="$(<"${COMPAT_SCRIPT}")"
  exec "${UDOCKER_BIN}" --allow-root run --nobanner \
    --volume="${BACKEND_DIR}:/robustops_backend_compat" \
    --env=PYTHONPATH=/robustops_backend_compat \
    --volume=/etc/hosts:/etc/hosts \
    "${LOG_VOLUMES[@]}" \
    --publish="${HOST_PORT}:${CONTAINER_PORT}" \
    "${CONTAINER}" python -c "${COMPAT_BOOTSTRAP}"
fi

exec "${UDOCKER_BIN}" --allow-root run --nobanner \
  --volume=/etc/hosts:/etc/hosts \
  "${LOG_VOLUMES[@]}" \
  --publish="${HOST_PORT}:${CONTAINER_PORT}" \
  "${CONTAINER}"
