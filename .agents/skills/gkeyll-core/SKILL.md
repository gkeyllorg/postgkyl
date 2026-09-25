---
name: gkeyll-core
description: Description of the C layer of the Gkeyll library which postgkyl is built on.
---

# Gkeyll Core

Postgkyl is built on top of the Gkeyll C library, which provides the core functionality for handling and processing data in a high-performance manner.
The core layer of Gkeyll owns DG interpolation kernels, array manipulation, DG arithmetic, and other fundamental operations.
Gkeyll can be found at https://github.com/gkeyllorg/gkeyll or in the /gkeyll folder of postgkyl after it is cloned.
When a feature has a low-level common feature, it should be implemented in the C library rather than Python.
Postgkyl wraps Gkeyll in the gpython shim.
