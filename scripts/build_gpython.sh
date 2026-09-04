#!/bin/sh
# Builds the _gpython CPython extension into src/postgkyl/gpython/_gpython.so
# (GKEYLL_C_SHIM.md). The gpython shim itself lives in the gkeyll repo
# (core/zero/gkyl_gpython.h + core/zero/gpython.c) and is compiled INTO
# libg0core.so by gkeyll's own build -- that compile step is the contract
# check: any core API drift fails there, at the producer. This script only
# compiles the extension against gkyl_gpython.h (opaque handles + scalars) and
# links the shim symbols from libg0core.so; a stale header/library pairing
# is caught at import by the GPYTHON_API_VERSION handshake.
#
# Requires a built gkeyll/build/core/libg0core.so (scripts/build_gkeyll.sh,
# which invokes this script as its final step). Safe to re-run by hand.
set -e

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
ROOT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
GKEYLL_DIR="${ROOT_DIR}/gkeyll"
LIB_DIR="${GKEYLL_DIR}/build/core"
CSRC_DIR="${ROOT_DIR}/src/postgkyl/gpython/csrc"
OUT="${ROOT_DIR}/src/postgkyl/gpython/_gpython.so"
BUNDLED_LIB="${ROOT_DIR}/src/postgkyl/gpython/libg0core.so"

if [ ! -f "${LIB_DIR}/libg0core.so" ]; then
    echo "error: ${LIB_DIR}/libg0core.so not found; run scripts/build_gkeyll.sh first" >&2
    exit 1
fi
if [ ! -f "${GKEYLL_DIR}/core/zero/gkyl_gpython.h" ]; then
    echo "error: gkeyll/core/zero/gkyl_gpython.h not found; this gkeyll tree lacks the gpython shim" >&2
    exit 1
fi

PYTHON="${PYTHON:-python3}"
PY_INCLUDES=$("${PYTHON}" -c "import sysconfig; print(sysconfig.get_path('include'))")
NUMPY_INCLUDE=$("${PYTHON}" -c "import numpy; print(numpy.get_include())")

# _gpythonmodule.c targets NPY_2_2_API_VERSION (pyproject.toml's numpy>=2.2.6
# floor) as a best-effort backstop, but a build-time/run-time NumPy version
# skew has been reproduced to crash the extension outright (segfault / heap
# corruption inside Gkeyll's C code) rather than fail cleanly -- the pin
# alone does not make a mismatched build safe. Under pip's default build
# isolation, this ${PYTHON} is a throwaway environment that resolves
# build-system.requires' numpy independently of whatever NumPy ends up
# installed for running postgkyl, so a stale/mismatched NumPy here (e.g. an
# ambient `python3` found ahead of the intended venv on PATH) would otherwise
# silently bake a broken extension instead of failing at build time. Always
# build with `--no-build-isolation` against the NumPy you're actually going
# to run with (see README.md).
NUMPY_OK=$("${PYTHON}" -c "
import sys
import numpy
major, minor = (int(p) for p in numpy.__version__.split('.')[:2])
sys.stdout.write('yes' if (major, minor) >= (2, 2) else 'no')
")
if [ "${NUMPY_OK}" != "yes" ]; then
    NUMPY_VERSION=$("${PYTHON}" -c "import numpy; print(numpy.__version__)")
    echo "error: building _gpython against NumPy ${NUMPY_VERSION} (via ${PYTHON}), but postgkyl requires numpy>=2.2.6 (pyproject.toml)." >&2
    echo "       This usually means '${PYTHON}' resolved to a different environment than the one postgkyl is being installed into." >&2
    echo "       Reinstall with: pip install -e . --no-build-isolation" >&2
    exit 1
fi

CC="${CC:-cc}"

# Keep the extension and its sole non-system shared library together.  A
# relative loader path then works from a wheel, virtualenv, or relocated source
# checkout without referring back to this build tree.
cp "${LIB_DIR}/libg0core.so" "${BUNDLED_LIB}"

# CPython extension modules must leave the Py* symbols unresolved at link
# time; the interpreter provides them at import. Linux's -shared does this
# by default, macOS needs -undefined dynamic_lookup (same flag setuptools
# passes on Darwin).
EXT_LDFLAGS=""
if [ "$(uname -s)" = "Darwin" ]; then
    EXT_LDFLAGS="-Wl,-undefined,dynamic_lookup"
    EXT_RPATH="@loader_path"
    # Gkeyll names the Mach-O library .so for consistency across platforms.
    # Give the bundled copy a relocatable install name before linking to it.
    install_name_tool -id "@rpath/libg0core.so" "${BUNDLED_LIB}"
else
    EXT_RPATH='$ORIGIN'
fi

echo "# Building _gpython extension (CC=${CC}) -> ${OUT}"
"${CC}" -O2 -g -fPIC -shared \
    "${CSRC_DIR}/_gpythonmodule.c" \
    -I "${GKEYLL_DIR}/core/zero" \
    -I "${PY_INCLUDES}" \
    -I "${NUMPY_INCLUDE}" \
    -L "$(dirname -- "${BUNDLED_LIB}")" -lg0core -Wl,-rpath,"${EXT_RPATH}" \
    ${EXT_LDFLAGS} \
    -o "${OUT}"
echo "# Built ${OUT}"

# Record what this build was made from -- gkeyll/ is a build-time-only clone
# (.gitignore'd), so its commit is otherwise unrecoverable once installed.
# Read by postgkyl._version for `pgkyl --version`.
BUILD_INFO="${ROOT_DIR}/src/postgkyl/gpython/_build_info.py"
_git_log_field() {  # _git_log_field <repo-dir> <log-format>
    git -C "$1" log -1 --format="$2" 2>/dev/null || echo unknown
}
GKEYLL_COMMIT=$(_git_log_field "${GKEYLL_DIR}" "%H")
GKEYLL_COMMIT_DATE=$(_git_log_field "${GKEYLL_DIR}" "%cI")
GKEYLL_BRANCH=pinned
POSTGKYL_BUILD_COMMIT=$(_git_log_field "${ROOT_DIR}" "%H")
BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ)
BUILD_ARCH_FLAGS="${ARCH_FLAGS:-}"

cat > "${BUILD_INFO}" <<PYEOF
"""Generated by scripts/build_gpython.sh -- do not edit by hand.

Captures the vendored Gkeyll checkout's identity and this build's timestamp;
neither is recoverable at runtime once installed, since gkeyll/ is a
build-time-only clone (see .gitignore). Read by postgkyl._version so
\`pgkyl --version\` can report what Gkeyll this build is linked against.
"""

GKEYLL_COMMIT = "${GKEYLL_COMMIT}"
GKEYLL_COMMIT_DATE = "${GKEYLL_COMMIT_DATE}"
GKEYLL_BRANCH = "${GKEYLL_BRANCH}"
POSTGKYL_BUILD_COMMIT = "${POSTGKYL_BUILD_COMMIT}"
BUILD_DATE = "${BUILD_DATE}"
BUILD_CC = "${CC}"
BUILD_ARCH_FLAGS = "${BUILD_ARCH_FLAGS}"
PYEOF
echo "# Wrote ${BUILD_INFO}"
