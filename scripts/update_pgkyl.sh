#!/usr/bin/env bash
# Update an installed source checkout using the active Python environment.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
GKEYLL_DIR="${ROOT_DIR}/gkeyll"
GKEYLL_BRANCH="${GKEYLL_BRANCH:-lapack_lite_shim_bugs}"
PYTHON="${PYTHON:-python}"

INSTALL_OPTIONS=(--no-build-isolation)
if [[ "$#" -eq 1 && "$1" == "--editable" ]]; then
    INSTALL_OPTIONS+=(--editable)
elif [[ "$#" -ne 0 ]]; then
    echo "usage: $0 [--editable]" >&2
    exit 2
fi

if [[ ! -e "${GKEYLL_DIR}/.git" ]]; then
    echo "error: gkeyll/ is missing; install Postgkyl from source first (see README.md)" >&2
    exit 1
fi
if ! git -C "${GKEYLL_DIR}" diff --quiet || \
   ! git -C "${GKEYLL_DIR}" diff --cached --quiet; then
    echo "error: gkeyll/ has tracked modifications; commit or stash them before updating" >&2
    exit 1
fi

echo "# Updating Postgkyl from the current branch's upstream"
git -C "${ROOT_DIR}" pull --ff-only

# The vendored checkout is detached, so fetch the selected remote branch
# and record its commit for build_gkeyll.sh to check out and build.
echo "# Fetching the latest Gkeyll revision from ${GKEYLL_BRANCH}"
git -C "${GKEYLL_DIR}" fetch --depth 1 --filter=blob:none origin \
    "refs/heads/${GKEYLL_BRANCH}"
GKEYLL_REVISION=$(git -C "${GKEYLL_DIR}" rev-parse FETCH_HEAD)
printf '%s\n' "${GKEYLL_REVISION}" > "${SCRIPT_DIR}/gkeyll-revision"

# setup.py builds Gkeyll core and then gpython using this same interpreter.
echo "# Rebuilding Gkeyll and gpython, and reinstalling Postgkyl"
POSTGKYL_SKIP_GKEYLL_BUILD=0 "${PYTHON}" -m pip install \
    "${INSTALL_OPTIONS[@]}" "${ROOT_DIR}"
"${PYTHON}" -c 'from postgkyl import gpython; gpython.require()'
echo "# Postgkyl update complete"
