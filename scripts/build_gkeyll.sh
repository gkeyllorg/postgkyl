#!/bin/sh
set -eu

usage() {
    cat <<EOF
usage: sh $0 [-h]

Build the existing Gkeyll core (libg0core.so) in gkeyll/, obtaining the
checkout only when absent. Local edits are built as-is; updating upstream is
a separate operation (scripts/update_gkeyll.sh).

options:
  -h, --help   show this help message and exit

environment:
  CC           C compiler (default: cc)
  ARCH_FLAGS   architecture flags (default: none, for portable binaries)
  BUILD_JOBS   parallel make jobs (default: half the processors)
EOF
}
case "$#:${1:-}" in
    0:) ;;
    1:-h|1:--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
esac

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
GKEYLL_DIR="${ROOT_DIR}/gkeyll"
if [ ! -e "${GKEYLL_DIR}/.git" ]; then
    sh "${SCRIPT_DIR}/update_gkeyll.sh"
fi

CC="${CC:-cc}"
echo "# Configuring gkeyll core (CC=${CC}, lapack-lite, app=core)"
(cd "${GKEYLL_DIR}" && ./configure "CC=${CC}" --use-lapack-lite=yes --app=core)

ARCH_FLAGS="${ARCH_FLAGS:-}"
export ARCH_FLAGS

echo "# Building libg0core.so (ARCH_FLAGS=${ARCH_FLAGS:-<none -- compiler default>})"
if [ -z "${BUILD_JOBS:-}" ]; then
    BUILD_JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)
    BUILD_JOBS=$((BUILD_JOBS > 1 ? BUILD_JOBS / 2 : 1))
fi
(cd "${GKEYLL_DIR}" && make core "ARCH_FLAGS=${ARCH_FLAGS}" \
    -j"${BUILD_JOBS}")

SO_PATH="${GKEYLL_DIR}/build/core/libg0core.so"
if [ ! -f "${SO_PATH}" ]; then
    echo "error: expected ${SO_PATH} after build, but it is missing" >&2
    exit 1
fi
echo "# Built ${SO_PATH}"
