"""``gpython/`` -- the foreign floor: the compiled bridge to Gkeyll.

A bottom leaf (imports nothing internal). This package is the **only** place
in postgkyl that touches the foreign world, and it does so through a compiled
contract (GKEYLL_C_SHIM.md) rather than runtime declarations:

- ``csrc/``    ``_gpythonmodule.c`` -- the CPython extension over
               ``gkyl_gpython.h``; the gpython shim itself lives in the
               gkeyll repo (``core/zero/{gkyl_gpython.h, gpython.c}``,
               compiled into ``libg0core.so`` by Gkeyll's own build)
- ``_gpython``  the built extension module -- opaque handles in, ndarrays out
- ``_lib``     loads ``_gpython`` + the ``GPYTHON_API_VERSION`` handshake;
               ``available()`` is the single capability switch;
               ``build_info()`` reads the generated ``_build_info`` (the
               vendored Gkeyll commit + build date, written by
               ``scripts/build_gpython.sh``) for ``pgkyl --version``
- ``array``    :class:`GkylArray` -- Python owner of a native ``gkyl_array``
- ``basis``    cached Gkeyll basis objects + interpolation/nodal/quad matrices
               built by evaluating Gkeyll's own basis through the shim
- ``rio``      file loading through ``gkyl_array_rio``
- ``kernels``  weak multiply/divide/inverse, coefficient lin-combs, reduce,
               integrate

Representation changes (modal · nodal · quad) are orchestration over this
floor's public functions, not floor primitives themselves -- they live in
``dg/rep.py`` (see CLAUDE.md's "Engine layers" section).

No struct layout, signature, or calling convention exists in Python: the C
compiler checks all of it against the real ``gkyl_*.h`` headers when the shim
builds, so Gkeyll API drift fails the build instead of corrupting data.

If the extension is missing, importing still succeeds; ``available()``
returns False and every entry point raises with build guidance.
"""

from ._lib import available, build_info, lib_path, require
from .array import GkylArray
from . import basis, kernels, rio

__all__ = [
    "available", "build_info", "lib_path", "require", "GkylArray", "basis",
    "kernels", "rio"
]
