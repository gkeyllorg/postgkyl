"""Characterization tests for the moved gyrokinetic flux-surface operation."""

from __future__ import annotations

import os

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.operations import gyrokinetics as gk_ops

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIELD = os.path.join(ROOT, "tests", "test_data",
    "rt_gk_tcv_nt_iwl_3x2v_p1-elc_M0_5.gkyl")

needs_gkeyll = pytest.mark.skipif(not gpython.available(),
    reason="no compiled Gkeyll (libg0core.so) found")


@needs_gkeyll
def test_flux_surface_move_preserves_output_and_projection_reuse():
  data = pg.load(FIELD)
  geometry = gk_ops.resolve_geometry(data.file_name)
  grid = gk_ops.resolve_flux_surface_grid(data, geometry, x_idx=0,
      nphi=4, nz_interp=2)
  first = gk_ops.extract_flux_surface(data, grid)
  second = gk_ops.extract_flux_surface(data.clone(), grid)
  assert first.values.shape == (4, 64, 1)
  assert first.ctx["interpolated"] is True
  np.testing.assert_allclose(first.values, second.values)
# end


@needs_gkeyll
@pytest.mark.parametrize(("kwargs", "message"), [
    ({"nphi": 0}, "nphi must be a positive integer"),
    ({"nz_interp": 0}, "nz_interp must be a positive integer"),
    ({"x_idx": -1}, "out of bounds"),
])
def test_flux_surface_public_validation(kwargs, message):
  data = pg.load(FIELD)
  geometry = gk_ops.resolve_geometry(data.file_name)
  with pytest.raises(ValueError, match=message):
    gk_ops.resolve_flux_surface_grid(data, geometry, **kwargs)
  # end
# end
