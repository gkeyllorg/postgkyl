#!/bin/sh
# Fetches (if needed) and builds the vendored Gkeyll `core` app as
# libg0core.so, for the gpython/ layer to bind against. Invoked automatically
# by `pip install`/`pip install -e` via setup.py, and safe to re-run by hand.
#
# gkeyll/ fetches the branch named by scripts/gkeyll-branch and builds the
# immutable commit in scripts/gkeyll-revision (or GKEYLL_REVISION). Zero
# external deps: no MPI/CUDA/SuperLU/Lua, LAPACK replaced by the bundled
# lapack-lite. Keep every root file and directory except vlasov/, pkpm/,
# moments/, and gyrokinetic/ (~200MB combined). A blobless fetch plus
# sparse-checkout avoids downloading their file contents. Development builds
# use the existing checkout in place, as described in README.md.
set -e

REPO_URL="https://github.com/ammarhakim/gkeyll.git"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
GKEYLL_DIR="${ROOT_DIR}/gkeyll"
BRANCH_FILE="${SCRIPT_DIR}/gkeyll-branch"
REVISION_FILE="${SCRIPT_DIR}/gkeyll-revision"

if [ ! -f "${BRANCH_FILE}" ]; then
    echo "error: Gkeyll branch file is missing: ${BRANCH_FILE}" >&2
    exit 1
fi
IFS= read -r GKEYLL_BRANCH < "${BRANCH_FILE}"
if ! git check-ref-format --branch "${GKEYLL_BRANCH}" >/dev/null 2>&1; then
    echo "error: ${BRANCH_FILE} must contain a valid Git branch name" >&2
    exit 1
fi
if [ "${GKEYLL_REVISION+x}" != x ]; then
    if [ ! -f "${REVISION_FILE}" ]; then
        echo "error: Gkeyll revision file is missing: ${REVISION_FILE}" >&2
        exit 1
    fi
    IFS= read -r GKEYLL_REVISION < "${REVISION_FILE}"
fi
case "${GKEYLL_REVISION}" in
    *[!0-9a-f]*|'')
        echo "error: GKEYLL_REVISION must be a full lowercase 40-character commit hash" >&2
        exit 1
        ;;
esac
if [ "${#GKEYLL_REVISION}" -ne 40 ]; then
    echo "error: GKEYLL_REVISION must be a full lowercase 40-character commit hash" >&2
    exit 1
fi

if [ ! -e "${GKEYLL_DIR}/.git" ]; then
    echo "# gkeyll/ not present -- fetching ${GKEYLL_BRANCH} (sparse + blobless)"
    git clone --depth 1 --filter=blob:none --no-checkout \
        --branch "${GKEYLL_BRANCH}" "${REPO_URL}" "${GKEYLL_DIR}"
else
    # Refuse local source edits before fetching or changing the checkout.
    if ! git -C "${GKEYLL_DIR}" diff --quiet || \
       ! git -C "${GKEYLL_DIR}" diff --cached --quiet; then
        echo "error: ${GKEYLL_DIR} has tracked modifications; cannot update Gkeyll" >&2
        exit 1
    fi
    echo "# Fetching the latest Gkeyll ${GKEYLL_BRANCH}"
    # Keep intervening commits so an existing shallow clone can fast-forward.
    git -C "${GKEYLL_DIR}" fetch --filter=blob:none origin \
        "+refs/heads/${GKEYLL_BRANCH}:refs/remotes/origin/${GKEYLL_BRANCH}"
fi

git -C "${GKEYLL_DIR}" sparse-checkout set --no-cone --stdin <<'EOF'
/*
!/vlasov/
!/pkpm/
!/moments/
!/gyrokinetic/
EOF
if git -C "${GKEYLL_DIR}" show-ref --verify --quiet "refs/heads/${GKEYLL_BRANCH}"; then
    git -C "${GKEYLL_DIR}" checkout "${GKEYLL_BRANCH}"
else
    git -C "${GKEYLL_DIR}" checkout --track -b "${GKEYLL_BRANCH}" "origin/${GKEYLL_BRANCH}"
fi
git -C "${GKEYLL_DIR}" merge --ff-only "refs/remotes/origin/${GKEYLL_BRANCH}"
REMOTE_REVISION=$(git -C "${GKEYLL_DIR}" rev-parse "refs/remotes/origin/${GKEYLL_BRANCH}")
ACTUAL_REVISION=$(git -C "${GKEYLL_DIR}" rev-parse HEAD)
if [ "${ACTUAL_REVISION}" != "${REMOTE_REVISION}" ]; then
    echo "error: Gkeyll ${GKEYLL_BRANCH} has local commits; cannot build the remote branch tip" >&2
    exit 1
fi
# A branch is only a discovery/update source. The immutable revision decides
# which producer code enters the wheel, independently of today's branch tip.
if ! git -C "${GKEYLL_DIR}" cat-file -e "${GKEYLL_REVISION}^{commit}" 2>/dev/null; then
    git -C "${GKEYLL_DIR}" fetch --depth 1 --filter=blob:none origin "${GKEYLL_REVISION}"
fi
git -C "${GKEYLL_DIR}" checkout --detach "${GKEYLL_REVISION}"
ACTUAL_REVISION=$(git -C "${GKEYLL_DIR}" rev-parse HEAD)
if [ "${ACTUAL_REVISION}" != "${GKEYLL_REVISION}" ]; then
    echo "error: Gkeyll checkout differs from requested revision ${GKEYLL_REVISION}" >&2
    exit 1
fi
echo "# Using Gkeyll ${ACTUAL_REVISION} (source branch ${GKEYLL_BRANCH})"

CC="${CC:-cc}"
echo "# Configuring gkeyll core (CC=${CC}, lapack-lite, app=core)"
(cd "${GKEYLL_DIR}" && ./configure "CC=${CC}" --use-lapack-lite=yes --app=core)

ARCH_FLAGS="${ARCH_FLAGS:-}"
export ARCH_FLAGS

echo "# Building libg0core.so (ARCH_FLAGS=${ARCH_FLAGS:-<none -- compiler default>})"
BUILD_JOBS=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 2)
BUILD_JOBS=$((BUILD_JOBS > 1 ? BUILD_JOBS / 2 : 1))
(cd "${GKEYLL_DIR}" && make core "ARCH_FLAGS=${ARCH_FLAGS}" \
    -j"${BUILD_JOBS}")

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
