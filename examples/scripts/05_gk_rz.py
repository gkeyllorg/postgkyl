"""Gyrokinetic R-Z mapping through fluent, functional, and reusable APIs.

Run directly:
    MPLBACKEND=Agg PYTHONPATH=src python examples/scripts/05_gk_rz.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import numpy as np

import postgkyl as pg
from postgkyl.operations import gyrokinetics as gk_ops

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(ROOT, "tests", "test_data")
OUTPUT_DIR = os.environ.get("PGKYL_EXAMPLE_OUTPUT", os.path.join(HERE, "output"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

FIELD = os.path.join(DATA, "rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl")
data = pg.load(FIELD)

# The common path: geometry is inferred from FIELD's simulation prefix.
mapped = data.gk_rz(z_axis=0.0, phi_tor=0.0, nz_interp=2)
fig = mapped.plot(show=False)
fig.savefig(os.path.join(OUTPUT_DIR, "05_gk_rz.png"))

# The functional spelling is the identical operation.
functional = pg.gk_rz(data, z_axis=0.0, phi_tor=0.0, nz_interp=2)
np.testing.assert_allclose(functional.values, mapped.values)

# Reuse geometry and projection when several fields/frames or toroidal angles
# share one computational grid.
geometry = gk_ops.resolve_geometry(data.file_name)
projection = gk_ops.resolve_rz_projection(data, geometry, z_axis=0.0, nz_interp=2)
at_zero = gk_ops.map_to_rz(data, projection, phi_tor=0.0)
at_quarter_turn = gk_ops.map_to_rz(data, projection, phi_tor=np.pi / 2)
np.testing.assert_allclose(at_zero.values, mapped.values)
assert not np.allclose(at_zero.values, at_quarter_turn.values, rtol=1e-12, atol=0.0)

print("05_gk_rz: OK")
