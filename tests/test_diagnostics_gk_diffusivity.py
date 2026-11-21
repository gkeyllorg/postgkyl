"""Tests for ``postgkyl.diagnostics.gk.diffusivity``.

A bilinear field's DG expansion is exact, so its values at the cells'
corners are the field's at the nodes and cells sharing a node agree. Per
cell constants jump between cells: the node value is the mean of the cells
sharing it and the gap the largest jump. The synthetic run of
``gk_tokamak_fixture`` writes ``anom_diff = D g^xx J`` with ``D = (1 + block)
+ x / 2`` and ``g^xx J = 2 + y / 4``: their product is bilinear, so the weak
product is exact and weak division recovers ``D`` exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

import gk_tokamak_fixture as fx
from postgkyl import gpython
from postgkyl.diagnostics.gk import diffusivity as dif

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")


@pytest.fixture(scope="module")
def run(tmp_path_factory):
  return str(fx.write_run(tmp_path_factory.mktemp("run")))


def _node_x(block):
  return fx.cell_edges(block)[0]


@needs_gkeyll
def test_node_values_of_a_bilinear_field():
  poly = (1.0, 2.0, -3.0, 0.5)  # 1 + 2 x - 3 y + x y / 2, block 2's grid
  coeffs = fx.bilinear_coefficients(2, poly)
  values, gap = dif.node_values(coeffs, "serendipity", 1)
  xe, ye = fx.cell_edges(2)
  X, Y = np.meshgrid(xe, ye, indexing="ij")
  np.testing.assert_allclose(values,
                             1 + 2 * X - 3 * Y + 0.5 * X * Y,
                             rtol=1e-14,
                             atol=1e-14)
  assert gap < 1e-13


@needs_gkeyll
def test_node_values_of_cell_constants():
  # Constant c per cell (coefficient 2 c on the 1/2 basis function).
  c = np.array([[1.0, 2.0], [3.0, 7.0]])
  coeffs = np.zeros((2, 2, 4))
  coeffs[..., 0] = 2 * c
  values, gap = dif.node_values(coeffs, "serendipity", 1)
  expected = np.array([[1.0, 1.5, 2.0], [2.0, 13 / 4, 4.5], [3.0, 5.0, 7.0]])
  np.testing.assert_allclose(values, expected, rtol=1e-15)
  assert gap == pytest.approx(6.0, rel=1e-15)  # 1 and 7 share the centre


@needs_gkeyll
def test_anomalous_diffusivity_divides_back(run):
  for species in ("elc", "ion"):
    got = dif.anomalous_diffusivity(fx.NAME, species, path=run)
    assert not got.uniform and got.species == species
    assert got.gap < 1e-13
    for b in range(len(fx.BLOCKS)):
      a, slope, _, _ = fx.diffusivity(b)
      xe, ye = fx.cell_edges(b)
      xc = 0.5 * (xe[1:] + xe[:-1])
      np.testing.assert_allclose(got.nodes[b],
                                 np.repeat((a + slope * xe)[:, None], ye.size,
                                           1),
                                 rtol=1e-13)
      np.testing.assert_allclose(got.cells[b],
                                 np.repeat((a + slope * xc)[:, None],
                                           ye.size - 1, 1),
                                 rtol=1e-13)


@needs_gkeyll
def test_uniform_diffusivity_reads_no_file(run):
  got = dif.anomalous_diffusivity(fx.NAME, "nobody", path=run, diffusivity=0.7)
  assert got.uniform and got.gap == 0.0
  for b, (_, _, (nr, nt), _, _) in enumerate(fx.BLOCKS):
    np.testing.assert_array_equal(got.nodes[b], np.full((nr + 1, nt + 1), 0.7))
    np.testing.assert_array_equal(got.cells[b], np.full((nr, nt), 0.7))
  assert "uniform (given)" in str(got)


@needs_gkeyll
@pytest.mark.parametrize("side, sign", [("outboard", 1.0), ("inboard", -1.0)])
def test_midplane_diffusivity(run, side, sign):
  got = dif.midplane_diffusivity(fx.NAME, "elc", side, path=run)
  # Node rows r = 0.5, 0.75, 1 (core), then 1.25, 1.5 (SOL: its r = 1 row
  # is the core's last one, kept with the inner block).
  r = np.array([0.5, 0.75, 1.0, 1.25, 1.5])
  np.testing.assert_allclose(got.R, fx.R0 + sign * r, rtol=1e-15)
  np.testing.assert_array_equal(got.blocks, [0, 0, 0, 1, 1])
  D0 = fx.diffusivity(0)[0] + 0.5 * _node_x(0)
  D1 = fx.diffusivity(1)[0] + 0.5 * _node_x(1)[1:]
  np.testing.assert_allclose(got.values, np.r_[D0, D1], rtol=1e-13)
  assert got.side == side
