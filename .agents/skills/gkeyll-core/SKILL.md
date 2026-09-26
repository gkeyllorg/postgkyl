---
name: gkeyll-core
description: Locate and modify the Gkeyll C implementation and its Postgkyl shim in the linked local producer checkout.
---

# Gkeyll Core

Postgkyl is built on top of the Gkeyll C library, which provides the core functionality for handling and processing data in a high-performance manner.
The core layer of Gkeyll owns DG interpolation kernels, array manipulation, DG arithmetic, and other fundamental operations.
The linked `gkeyll/` folder in the Postgkyl workspace is the source tree to edit
for native kernels, dispatch, and the gpython shim. Shared low-level behavior
belongs in this C library. Postgkyl wraps it through the gpython shim.

Follow the [native skill's local development workflow](../native/SKILL.md#local-gkeyll-development)
for direct edits, rebuilding, and the requirement to leave both repositories
uncommitted. Being ignored by the parent repository does not move ownership
of native fixes into Postgkyl patch files.
