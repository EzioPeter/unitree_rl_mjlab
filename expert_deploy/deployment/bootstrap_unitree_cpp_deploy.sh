#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TARGET="${1:-${REPO_ROOT}/.local/unitree_cpp_deploy}"
DEPLOY_SOURCE="${REPO_ROOT}/expert_deploy/deployment"
UPSTREAM_URL="https://github.com/wty-yy/unitree_cpp_deploy.git"
UPSTREAM_COMMIT="9400a4a73eaa9f79e7e07f2df50061cc7fb7520c"
ORT_VERSION="1.23.2"
ORT_ARCHIVE="onnxruntime-linux-x64-${ORT_VERSION}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_ARCHIVE}"
ORT_SHA256="1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b"

if [[ -e "${TARGET}" ]]; then
  echo "Target already exists: ${TARGET}" >&2
  exit 2
fi

mkdir -p "${TARGET}"
git -C "${TARGET}" init
git -C "${TARGET}" remote add origin "${UPSTREAM_URL}"
git -C "${TARGET}" sparse-checkout init --cone
git -C "${TARGET}" sparse-checkout set deploy
git -C "${TARGET}" fetch --depth 1 --filter=blob:none origin "${UPSTREAM_COMMIT}"
git -C "${TARGET}" checkout --detach FETCH_HEAD
cp -a "${DEPLOY_SOURCE}/overlay/." "${TARGET}/"

case "$(uname -m)" in
  x86_64|amd64) ;;
  *)
    echo "This pinned controller build currently supports x86_64 Linux only." >&2
    echo "For another architecture, replace the ONNX Runtime package and CMake paths together." >&2
    exit 3
    ;;
esac

ORT_PARENT="${TARGET}/deploy/thirdparty"
ORT_DIR="${ORT_PARENT}/onnxruntime-linux-x64-${ORT_VERSION}"
if [[ ! -f "${ORT_DIR}/include/onnxruntime_cxx_api.h" ]]; then
  TMP_DIR="$(mktemp -d)"
  trap 'rm -rf "${TMP_DIR}"' EXIT
  curl -L --fail --retry 3 -o "${TMP_DIR}/${ORT_ARCHIVE}" "${ORT_URL}"
  echo "${ORT_SHA256}  ${TMP_DIR}/${ORT_ARCHIVE}" | sha256sum --check -
  tar -xzf "${TMP_DIR}/${ORT_ARCHIVE}" -C "${ORT_PARENT}"
fi

POLICY_DST="${TARGET}/logs/go2/g0_d0_rrcalf_0p5"
mkdir -p "${POLICY_DST}"
cp -a "${DEPLOY_SOURCE}/policies/g0_d0_rrcalf_0p5/." "${POLICY_DST}/"
cp "${DEPLOY_SOURCE}/config.yaml" "${TARGET}/deploy/robots/go2/config/config.yaml"

echo "Prepared deployment tree: ${TARGET}"
echo "Build with: cmake -S ${TARGET}/deploy/robots/go2 -B ${TARGET}/deploy/robots/go2/build"
echo "Then: cmake --build ${TARGET}/deploy/robots/go2/build -j"
