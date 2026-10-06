#!/bin/sh
set -e

usage() {
    cat <<EOF
usage: sh $0 [-h] path/to/postgkyl.whl

Install one built wheel into a fresh environment and prove that its native
bridge loads from outside the source checkout. Dependencies are installed
normally so the smoke test exercises the same NumPy ABI users receive.

options:
  -h, --help   show this help message and exit

environment:
  PYTHON                    interpreter for the fresh venv (default: python3)
  POSTGKYL_SMOKE_NO_DEPS=1  reuse the interpreter's installed dependencies
                            (offline developer check; leave unset in CI)
EOF
}

case "$#:${1:-}" in
    1:-h|1:--help) usage; exit 0 ;;
    1:*) ;;
    *) usage >&2; exit 2 ;;
esac

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
case "$1" in
    /*) WHEEL=$1 ;;
    *) WHEEL=$(pwd)/$1 ;;
esac
if [ ! -f "${WHEEL}" ]; then
    echo "error: wheel not found: ${WHEEL}" >&2
    exit 1
fi

SMOKE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/postgkyl-wheel-smoke.XXXXXX")
trap 'rm -rf "${SMOKE_DIR}"' EXIT HUP INT TERM
if [ "${POSTGKYL_SMOKE_NO_DEPS:-0}" = "1" ]; then
    # Useful for an offline developer check when the invoking interpreter
    # already has postgkyl's dependencies.  CI/release checks should leave
    # this unset and exercise normal dependency resolution.
    "${PYTHON}" -m venv --system-site-packages "${SMOKE_DIR}/venv"
    "${SMOKE_DIR}/venv/bin/python" -m pip install --no-deps "${WHEEL}"
else
    "${PYTHON}" -m venv "${SMOKE_DIR}/venv"
    "${SMOKE_DIR}/venv/bin/python" -m pip install "${WHEEL}[test]"
fi

cd "${SMOKE_DIR}"
POSTGKYL_REQUIRE_GKEYLL=1 "${SMOKE_DIR}/venv/bin/python" "${SCRIPT_DIR}/check_wheel.py"
"${SMOKE_DIR}/venv/bin/pgkyl" --version
"${SMOKE_DIR}/venv/bin/python" -m pip check
