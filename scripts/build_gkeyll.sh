#!/bin/sh
set -eu

usage() {
    cat <<EOF
usage: sh $0 [-h]

Build the selected Gkeyll module and its prerequisites in gkeyll/, obtaining the
checkout only when absent. Local edits are built as-is; updating upstream is
a separate operation (scripts/update_gkeyll.sh).

options:
  -h, --help   show this help message and exit

configuration:
  postgkyl-build.toml selects the module and build ownership.

environment:
  POSTGKYL_BUILD_CONFIG  alternate configuration file
  PYTHON       interpreter for reading configuration (default: python)
  POSTGKYL_GKEYLL_MODULE  highest module: core (default), moments, vlasov, gyrokinetic, pkpm
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
exec "${PYTHON:-python}" "${SCRIPT_DIR}/build_gkeyll.py"
