---
name: native
description: Change or troubleshoot the Gkeyll native bridge, basis matrices, readers, memory ownership, or native builds in Postgkyl.
---

# Keep the foreign boundary compiled

`gpython/` is the only doorway to Gkeyll. No ctypes declarations, native struct
layouts, or foreign signatures in Python or other layers. `gpython.available()`
is the single capability switch.

The shim lives in `gkeyll/core/zero/{gkyl_gpython.h,gpython.c}` and builds into
Gkeyll's `libg0core.so`. Struct access, by-value basis conventions, and function
pointer dispatch belong there, checked against that tree's headers.
`gpython/csrc/_gpythonmodule.c` wraps only the shim's opaque handles, scalars,
and buffers. `_lib.py` imports the extension and checks `GPYTHON_API_VERSION`.

## Local Gkeyll development

Make native kernel and shim changes directly in the linked `gkeyll/` checkout
under the Postgkyl workspace. This producer tree owns the C implementation even
though Postgkyl ignores it. Do not substitute Postgkyl-managed patch files,
build-time patch application, or Python workarounds for changes that belong
there. Do not commit changes in either repository; leave the diffs for the user.
Do not change the dependency branch merely to capture local development work.

Inspect `git status` in both repositories before editing, preserve existing
changes, and report the native diff separately from the Postgkyl diff. Follow
Gkeyll's own instructions and formatting configuration for its source files.

For local native changes, repeat the editable installation from the workspace:

```bash
python -m pip install -e '.[test]'
```

This is the same build path used for clean installations. Setuptools builds the
extension and bundles the core library and JSON provenance. The core build
script obtains Gkeyll only when absent; an existing checkout is built as-is,
including local edits. For just the core, use `sh scripts/build_gkeyll.sh`.

`sh scripts/update_gkeyll.sh` explicitly fetches and fast-forwards the branch
named in `scripts/gkeyll-branch`; it refuses tracked edits and local commits on
that branch. `bash scripts/update_pgkyl.sh` updates both repositories
and reinstalls. Do not use either updater to build ongoing native edits or clear
those edits to satisfy an updater.

Bundle libg0core beside the extension and retain relative `$ORIGIN`/`@loader_path`
linking. Preserve generated build provenance; Gkeyll is needed at build time,
not at runtime.

## Data ownership and verification

GkylArray capsules own and release arrays. Zero-copy construction pins its NumPy
buffer; view base chains pin the capsule so views outlive their dataset. Never
return unowned C memory. Build interpolation and representation matrices by
evaluating Gkeyll's own basis through the shim, not duplicated basis formulas.

`dg/` orchestrates kernels; `io/` dispatches readers. Prefer GkylCReader for native
field reads; retain the Python reader fallback for unavailable native libraries,
partial loads, and dynvectors. Build isolation supplies NumPy headers; the
extension targets the supported NumPy API floor. Wheel checks exercise both the minimum and current runtime
NumPy. Use the editable install above for local native edits too.

Verify handshake, memory lifetime, basis/interpolation, and modal algebra using
the relevant existing tests. Report native test skips explicitly when no compiled
library is available; skips do not establish native correctness.
