"""Independent affine-flow references for fluid enstrophy and strain."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.diagnostics.mom import enstrophy as ens

GEN = Path(__file__).parent / "test_data" / "generated"
needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


@needs_gkeyll
@pytest.mark.parametrize("ndim, curl, gradient", [(2, 9.0, 14.0),
                                                  (3, 6.0, 16.0)])
def test_real_affine_moments_integrate_entire_domain(ndim, curl, gradient):
  # rho=2, u=(y+z,-x,2*x-y). Its curl is (-1,-1,-2) in 3D;
  # with z=0 the 2D curl is (-1,-2,-2). The gradient has squared norm
  # 8 and 7 respectively. Unit volume makes their integrals immediate.
  out = ens.enstrophy(str(GEN / f"moments_{ndim}d_affine_"), 0, 0)
  np.testing.assert_allclose(out.enstrophy, [curl], rtol=3e-14, atol=3e-14)
  np.testing.assert_allclose(out.incompressible_enstrophy, [gradient],
                             rtol=3e-14,
                             atol=3e-14)


@needs_gkeyll
@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("representation", ["nodal", "quad"])
def test_packed_point_representations_resolve_physical_moments(
    ndim, representation):
  data = pg.load(str(
      GEN / f"moments_{ndim}d_affine_0.gkyl")).represent(to=representation)
  expected = (9, 14) if ndim == 2 else (6, 16)
  np.testing.assert_allclose(ens._enstrophy_terms(data),
                             expected,
                             rtol=3e-14,
                             atol=3e-14)


@pytest.mark.parametrize("ndim", [2, 3])
def test_point_velocity_gradient_includes_off_diagonal_and_boundary_cells(ndim):
  # The same affine field sampled at genuine cell centers. The numerical
  # derivative is exact for affine functions, including every boundary.
  grid = [np.linspace(0, 1, n + 1) for n in [3, 4, 5][:ndim]]
  coords = np.meshgrid(*[(edges[:-1] + edges[1:]) / 2 for edges in grid],
                       indexing="ij")
  x, y = coords[:2]
  z = coords[2] if ndim == 3 else 0
  values = np.stack([
      np.full_like(x, 2), 2 * (y + z), -2 * x, 2 * (2 * x - y),
      np.full_like(x, 100)
  ],
                    axis=-1)
  data = pg.GData()
  data.push(grid, values)
  expected = (9, 14) if ndim == 2 else (6, 16)
  np.testing.assert_allclose(ens._enstrophy_terms(data),
                             expected,
                             rtol=3e-14,
                             atol=3e-14)


@needs_gkeyll
@pytest.mark.parametrize("ndim", [2, 3])
def test_constant_velocity_frames_have_zero_enstrophy(ndim):
  out = ens.enstrophy(str(GEN / f"moments_{ndim}d_uniform_"), 1, 2)
  np.testing.assert_allclose(out.enstrophy, np.zeros(2), rtol=0, atol=1e-27)
  np.testing.assert_allclose(out.incompressible_enstrophy,
                             np.zeros(2),
                             rtol=0,
                             atol=1e-27)


def test_traces_are_frozen():
  traces = ens.EnstrophyTraces(np.array([1.0]), np.array([2.0]))
  with pytest.raises(FrozenInstanceError):
    traces.enstrophy = np.array([3.0])
