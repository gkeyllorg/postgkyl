#!/bin/sh
# Fetches (if needed) and builds the vendored Gkeyll `core` app as
# libg0core.so, for the gpython/ layer to bind against. Invoked automatically
# by `pip install`/`pip install -e` via setup.py, and safe to re-run by hand.
#
# gkeyll/ is a plain, detached clone pinned by scripts/gkeyll-revision (zero
# external deps: no MPI/CUDA/SuperLU/Lua, LAPACK replaced by the bundled
# lapack-lite). Only core/ is needed to build libg0core.so, so moments/,
# vlasov/, gyrokinetic/, and pkpm/ (~200MB combined) are excluded via
# sparse-checkout and are never fetched, not merely deleted after the fact.
set -e

REPO_URL="https://github.com/ammarhakim/gkeyll.git"
SPARSE_DIRS="core gkeyll install-deps machines"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
GKEYLL_DIR="${ROOT_DIR}/gkeyll"
REVISION_FILE="${SCRIPT_DIR}/gkeyll-revision"

if [ ! -f "${REVISION_FILE}" ]; then
    echo "error: pinned Gkeyll revision file is missing: ${REVISION_FILE}" >&2
    exit 1
fi
IFS= read -r GKEYLL_REVISION < "${REVISION_FILE}"
case "${GKEYLL_REVISION}" in
    *[!0-9a-f]*|'')
        echo "error: ${REVISION_FILE} must contain one lowercase commit SHA" >&2
        exit 1
        ;;
esac
if [ "${#GKEYLL_REVISION}" -ne 40 ]; then
    echo "error: ${REVISION_FILE} must contain a full 40-character commit SHA" >&2
    exit 1
fi

if [ ! -e "${GKEYLL_DIR}/.git" ]; then
    echo "# gkeyll/ not present -- fetching pinned ${GKEYLL_REVISION} (core-only, sparse + blobless)"
    rmdir "${GKEYLL_DIR}" 2>/dev/null || true
    mkdir "${GKEYLL_DIR}"
    git -C "${GKEYLL_DIR}" init
    git -C "${GKEYLL_DIR}" remote add origin "${REPO_URL}"
    git -C "${GKEYLL_DIR}" sparse-checkout init --cone
    git -C "${GKEYLL_DIR}" sparse-checkout set ${SPARSE_DIRS}
    git -C "${GKEYLL_DIR}" fetch --depth 1 --filter=blob:none origin "${GKEYLL_REVISION}"
else
    echo "# gkeyll/ already present -- ensuring sparse-checkout excludes heavy apps"
    (cd "${GKEYLL_DIR}" && git sparse-checkout init --cone >/dev/null 2>&1 || true
     git -C "${GKEYLL_DIR}" sparse-checkout set ${SPARSE_DIRS})
    if ! git -C "${GKEYLL_DIR}" cat-file -e "${GKEYLL_REVISION}^{commit}" 2>/dev/null; then
        git -C "${GKEYLL_DIR}" fetch --depth 1 --filter=blob:none origin "${GKEYLL_REVISION}"
    fi
fi

# A dirty producer tree makes the native artifact's source unknowable even
# when HEAD is pinned.  Refuse it instead of recording misleading build info.
if ! git -C "${GKEYLL_DIR}" diff --quiet || \
   ! git -C "${GKEYLL_DIR}" diff --cached --quiet; then
    echo "error: ${GKEYLL_DIR} has tracked modifications; cannot build the pinned Gkeyll source" >&2
    exit 1
fi
git -C "${GKEYLL_DIR}" checkout --detach "${GKEYLL_REVISION}"
ACTUAL_REVISION=$(git -C "${GKEYLL_DIR}" rev-parse HEAD)
if [ "${ACTUAL_REVISION}" != "${GKEYLL_REVISION}" ]; then
    echo "error: expected Gkeyll ${GKEYLL_REVISION}, checked out ${ACTUAL_REVISION}" >&2
    exit 1
fi
echo "# Using pinned Gkeyll revision ${GKEYLL_REVISION}"

CC="${CC:-cc}"
echo "# Configuring gkeyll core (CC=${CC}, lapack-lite, app=core)"
(cd "${GKEYLL_DIR}" && ./configure "CC=${CC}" --use-lapack-lite=yes --app=core)

ARCH_FLAGS="${ARCH_FLAGS:-}"
export ARCH_FLAGS

echo "# Building libg0core.so (ARCH_FLAGS=${ARCH_FLAGS:-<none -- compiler default>})"
(cd "${GKEYLL_DIR}" && make core "ARCH_FLAGS=${ARCH_FLAGS}" \
    -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)")

SO_PATH="${GKEYLL_DIR}/build/core/libg0core.so"
if [ ! -f "${SO_PATH}" ]; then
    echo "error: expected ${SO_PATH} after build, but it is missing" >&2
    exit 1
fi
echo "# Built ${SO_PATH}"

# Build the _gpython extension against gkyl_gpython.h + libg0core.so. The
# gpython shim itself (core/zero/gpython.c) was just compiled INTO
# libg0core.so above -- that step is the compile-time contract check
# (GKEYLL_C_SHIM.md).
sh "${SCRIPT_DIR}/build_gpython.sh"
