#!/usr/bin/env bash
set -euo pipefail

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${BACKEND_DIR}/../.." && pwd)"
PYTHON_BIN="${ROBUSTOPS_PYTHON:-python3}"
RUNTIME_ROOT="${ROBUSTOPS_MCP_RUNTIME_ROOT:-${ROBUSTOPS_EMAIL_MCP_RUNTIME_ROOT:-${ROOT_DIR}/.runtime/enterpriseops_mcp}}"
VENV_DIR="${RUNTIME_ROOT}/venv"
export UDOCKER_DIR="${RUNTIME_ROOT}/udocker"
DOMAIN="${1:-}"

UDOCKER_VERSION="1.3.17"
DOCKER_REGISTRY="${ROBUSTOPS_DOCKER_REGISTRY:-https://registry-1.docker.io}"
DOCKER_AUTH_URL="${ROBUSTOPS_DOCKER_AUTH_URL-https://auth.docker.io/token}"
DOCKER_REGISTRY_SERVICE="${ROBUSTOPS_DOCKER_REGISTRY_SERVICE:-registry.docker.io}"
DOCKER_REGISTRY="${DOCKER_REGISTRY%/}"

if [[ "${DOMAIN}" == "all" ]]; then
  for domain in calendar csm drive email hr itsm teams; do
    "${BASH_SOURCE[0]}" "${domain}"
  done
  exit 0
fi

case "${DOMAIN}" in
  email)
    IMAGE_REPOSITORY="shivakrishnareddyma225/enterpriseops-gym-mcp-email"
    MANIFEST_DIGEST="sha256:69c2081fe4ab0962b86233f9fb52b307b8ad0019f6746ba64ce75851036201cd"
    ;;
  calendar)
    IMAGE_REPOSITORY="shivakrishnareddyma225/enterpriseops-gym-mcp-calendar"
    MANIFEST_DIGEST="sha256:994c5421a6dd065861bc7f813a177f6d408875e9df60fe8d012959bc4510da02"
    ;;
  drive)
    IMAGE_REPOSITORY="shivakrishnareddyma225/enterpriseops-gym-mcp-drive"
    MANIFEST_DIGEST="sha256:3475962fcf6da7675e194dbf138de01fa3e96134a302ad47316e4111a5e63f32"
    ;;
  csm)
    IMAGE_REPOSITORY="shivakrishnareddyma225/enterpriseops-gym-mcp-csm"
    MANIFEST_DIGEST="sha256:eaa456ac9aa85728426e7d3813a0bbca0949d6a8695be30e26f03894e6e6b189"
    ;;
  hr)
    IMAGE_REPOSITORY="shivakrishnareddyma225/enterpriseops-gym-mcp-hr"
    MANIFEST_DIGEST="sha256:1ea1c1d64d4be35e8062e56f00b8318e9e6c09289cfa56bcfd0595bfa59ac64d"
    ;;
  itsm)
    IMAGE_REPOSITORY="shivakrishnareddyma225/enterpriseops-gym-mcp-itsm"
    MANIFEST_DIGEST="sha256:a234ae3fb7cee196ba25e6b9957969dea829919b6e8271dddae128f065aaf39f"
    ;;
  teams)
    IMAGE_REPOSITORY="shivakrishnareddyma225/enterpriseops-gym-mcp-teams"
    MANIFEST_DIGEST="sha256:602655e46f6501885540c36dc9b12114cb173c75063d7f25c17ed0652695fa78"
    ;;
  *)
    echo "usage: $0 {all|calendar|csm|drive|email|hr|itsm|teams}" >&2
    exit 2
    ;;
esac

IMAGE="${IMAGE_REPOSITORY}:latest"
CONTAINER="robustops-${DOMAIN}-mcp"

mkdir -p "${RUNTIME_ROOT}"

if [[ ! -x "${VENV_DIR}/bin/udocker" ]] \
  || ! "${VENV_DIR}/bin/udocker" --allow-root --version >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m venv --clear "${VENV_DIR}"
  "${VENV_DIR}/bin/pip" install --disable-pip-version-check "udocker==${UDOCKER_VERSION}"
fi
UDOCKER_BIN="${VENV_DIR}/bin/udocker"

if [[ ! -x "${UDOCKER_DIR}/bin/proot-x86_64" ]]; then
  "${UDOCKER_BIN}" --allow-root install
fi

if ! "${UDOCKER_BIN}" --allow-root inspect "${IMAGE}" >/dev/null 2>&1; then
  manifest_headers=(
    -H 'Accept: application/vnd.docker.distribution.manifest.v2+json'
  )
  if [[ -n "${DOCKER_AUTH_URL}" ]]; then
    registry_token="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(sys.stdin)["token"])' \
      < <(curl -fsS --get \
        --data-urlencode "service=${DOCKER_REGISTRY_SERVICE}" \
        --data-urlencode "scope=repository:${IMAGE_REPOSITORY}:pull" \
        "${DOCKER_AUTH_URL}"))"
    manifest_headers+=( -H "Authorization: Bearer ${registry_token}" )
  fi
  resolved_digest="$(curl -fsSI "${manifest_headers[@]}" \
    "${DOCKER_REGISTRY}/v2/${IMAGE_REPOSITORY}/manifests/latest" \
    | awk 'tolower($1) == "docker-content-digest:" {gsub("\r", "", $2); print $2}')"
  if [[ "${resolved_digest}" != "${MANIFEST_DIGEST}" ]]; then
    echo "error: ${DOMAIN} MCP latest now resolves to ${resolved_digest:-unknown}, expected ${MANIFEST_DIGEST}" >&2
    exit 2
  fi
  "${UDOCKER_BIN}" --allow-root pull --registry="${DOCKER_REGISTRY}" "${IMAGE}"
fi

if ! image_verify_output="$("${UDOCKER_BIN}" --allow-root verify "${IMAGE}" 2>&1)"; then
  printf '%s\n' "${image_verify_output}" >&2
  echo "error: ${DOMAIN} MCP image layer verification failed; repair or remove the damaged local image before retrying" >&2
  exit 2
fi

if ! "${UDOCKER_BIN}" --allow-root inspect "${CONTAINER}" >/dev/null 2>&1; then
  "${UDOCKER_BIN}" --allow-root create --name="${CONTAINER}" "${IMAGE}"
fi

"${UDOCKER_BIN}" --allow-root setup --execmode=P1 "${CONTAINER}" >/dev/null
if ! runtime_check_output="$("${UDOCKER_BIN}" --allow-root run --nobanner \
  "${CONTAINER}" python -c 'import uvicorn; print(f"uvicorn={uvicorn.__version__}")' 2>&1)"; then
  printf '%s\n' "${runtime_check_output}" >&2
  echo "error: ${DOMAIN} MCP container cannot import its required uvicorn runtime" >&2
  exit 2
fi
printf '%s\n' "${runtime_check_output}"

printf '%s\n' \
  "udocker_version=${UDOCKER_VERSION}" \
  "image=${IMAGE}" \
  "registry=${DOCKER_REGISTRY}" \
  "manifest_digest=${MANIFEST_DIGEST}" \
  "container=${CONTAINER}" \
  > "${RUNTIME_ROOT}/runtime_info_${DOMAIN}.txt"

printf '%s MCP udocker runtime prepared: %s\n' "${DOMAIN}" "${RUNTIME_ROOT}"
