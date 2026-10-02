#!/bin/sh
# Obtain Gkeyll or explicitly fast-forward its configured upstream branch.
set -eu

REPO_URL="https://github.com/gkeyllorg/gkeyll.git"

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
GKEYLL_DIR="${ROOT_DIR}/gkeyll"
BRANCH_FILE="${SCRIPT_DIR}/gkeyll-branch"

if [ ! -f "${BRANCH_FILE}" ]; then
    echo "error: Gkeyll branch file is missing: ${BRANCH_FILE}" >&2
    exit 1
fi
IFS= read -r GKEYLL_BRANCH < "${BRANCH_FILE}"
if ! git check-ref-format --branch "${GKEYLL_BRANCH}" >/dev/null 2>&1; then
    echo "error: ${BRANCH_FILE} must contain a valid Git branch name" >&2
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
    # Replace a previous single-branch clone's fetch configuration so Git
    # recognizes the new remote branch when setting up upstream tracking.
    git -C "${GKEYLL_DIR}" remote set-branches origin "${GKEYLL_BRANCH}"
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
echo "# Using Gkeyll ${ACTUAL_REVISION} (source branch ${GKEYLL_BRANCH})"
