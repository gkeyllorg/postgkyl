#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
usage: bash $0 [-h] [--no-editable]

Update an installed source checkout using the active Python environment:
fast-forward Postgkyl and Gkeyll, rebuild Gkeyll and gpython, and reinstall
Postgkyl (editable by default).

options:
  -h, --help     show this help message and exit
  --no-editable  install a copy of the checkout instead of an editable install

environment:
  PYTHON         interpreter to install into (default: python)
EOF
}

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
GKEYLL_DIR="${ROOT_DIR}/gkeyll"
PYTHON="${PYTHON:-python}"

INSTALL_TARGET=(--editable "${ROOT_DIR}")
case "$#:${1:-}" in
    0:) ;;
    1:-h|1:--help) usage; exit 0 ;;
    1:--no-editable) INSTALL_TARGET=("${ROOT_DIR}") ;;
    *) usage >&2; exit 2 ;;
esac

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

# Fetch explicitly; ordinary installs only build the existing producer.
sh "${SCRIPT_DIR}/update_gkeyll.sh"
echo "# Rebuilding Gkeyll and gpython, and reinstalling Postgkyl"
POSTGKYL_SKIP_GKEYLL_BUILD=0 "${PYTHON}" -m pip install "${INSTALL_TARGET[@]}"
"${PYTHON}" -c 'from postgkyl import gpython; gpython.require()'
echo "# Postgkyl update complete"
