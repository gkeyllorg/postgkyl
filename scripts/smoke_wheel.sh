#!/bin/sh
# Install one built wheel into a fresh environment and prove that its native
# bridge loads from outside the source checkout.  Dependencies are installed
# normally so the smoke test exercises the same NumPy ABI users receive.
set -e

if [ "$#" -ne 1 ]; then
    echo "usage: $0 path/to/postgkyl.whl" >&2
    exit 2
fi

PYTHON="${PYTHON:-python3}"
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
SMOKE_FIELD="${ROOT_DIR}/tests/test_data/rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl"
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
    "${SMOKE_DIR}/venv/bin/python" -m pip install "${WHEEL}"
fi

cd "${SMOKE_DIR}"
POSTGKYL_SMOKE_FIELD="${SMOKE_FIELD}" \
    "${SMOKE_DIR}/venv/bin/python" - <<'PY'
import os
from pathlib import Path

import postgkyl
from postgkyl import gpython

assert gpython.available(), "the wheel's compiled Gkeyll bridge did not load"
extension = gpython.lib_path()
assert extension is not None
core = extension.with_name("libg0core.so")
assert core.is_file(), f"wheel does not contain {core}"
assert "site-packages" in str(Path(postgkyl.__file__).resolve())
assert gpython.require().api_version() == gpython.require().GPYTHON_API_VERSION
field = Path(os.environ["POSTGKYL_SMOKE_FIELD"])
modal = postgkyl.load(field)
point_values = modal.interpolate()
assert modal.backend == "gkyl"
assert point_values.backend == "numpy"
print(f"loaded {extension}")
print(f"loaded bundled {core}")
PY
"${SMOKE_DIR}/venv/bin/pgkyl" --version
"${SMOKE_DIR}/venv/bin/python" -m pip check
