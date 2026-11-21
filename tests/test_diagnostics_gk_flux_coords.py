"""Analytic tests for ``postgkyl.diagnostics.gk.flux_coords``.

Edge curves are checked against circle arcs and straight segments with known
Gauss nodes; psi at the corners against bilinear fields whose orthonormal
serendipity p = 1 coefficients are written in closed form; L_theta against
straight and circular cell-centre lines.

Run: PYTHONPATH=src pytest tests/test_diagnostics_gk_flux_coords.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from postgkyl import gpython
from postgkyl.diagnostics.gk import flux_coords as fc

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

# The two-point Gauss-Legendre nodes +-1/sqrt(3) of [-1, 1], on [0, 1].
GAUSS = 0.5 + 0.5 / np.sqrt(3) * np.array([-1.0, 1.0])


def polar(r, a):
  """(R, Z) of radius ``r`` and angle ``a``."""
  return np.stack([r * np.cos(a), r * np.sin(a)], -1)


def annulus(radii, angles):
  """An annular block: node rows on circles of ``radii``, node columns on
  rays at ``angles``, with the exact Gauss nodes of every edge (on the arc
  for constant-psi edges, on the ray for constant-theta ones)."""
  r, a = np.asarray(radii, float), np.asarray(angles, float)
  corners = polar(r[:, None], a[None, :])
  a_gauss = (a[:-1, None] + GAUSS * np.diff(a)[:, None]).ravel()  # (2 nt,)
  r_gauss = (r[:-1, None] + GAUSS * np.diff(r)[:, None]).ravel()  # (2 nr,)
  surf0 = polar(r[:, None], a_gauss[None, :])
  surf1 = polar(r_gauss[:, None], a[None, :])
  return corners, surf0, surf1


def cubic_error_bound(radius, span):
  """A bound on how far the cubic through an arc's corners and Gauss nodes
  strays from the arc: each coordinate's interpolation error is
  f''''(xi) / 4! omega(t), omega(t) = t (t - g1) (t - g2) (t - 1), and the
  fourth t-derivative of r (cos, sin)(a0 + t span) is at most r span^4."""
  t = np.linspace(0, 1, 2001)
  omega = np.abs(t * (t - GAUSS[0]) * (t - GAUSS[1]) * (t - 1)).max()
  return np.sqrt(2) * radius * span**4 * omega / 24


class TestEdgeCurves:

  def test_lagrange_weights_reproduce_cubics(self):
    knots = np.r_[0.0, GAUSS, 1.0]
    p = np.polynomial.Polynomial([1.0, 2.0, -3.0, 0.5])
    w = fc.lagrange_weights(fc.EDGE_T, knots)
    np.testing.assert_allclose(w @ p(knots), p(fc.EDGE_T), rtol=0, atol=1e-14)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=0, atol=1e-15)

  def test_gauss_t_are_on_edge_t(self):
    np.testing.assert_allclose(fc.GAUSS_T, GAUSS, rtol=0, atol=1e-16)
    assert all(np.isclose(fc.EDGE_T, g).any() for g in GAUSS)

  def test_polylines_pass_through_corners_and_gauss_nodes(self):
    corners, surf0, surf1 = annulus([1.0, 1.3, 1.5], np.linspace(0, 2, 4))
    rows, cols = fc.edge_polylines(corners, surf0, surf1)
    k = len(fc.EDGE_T) + 1
    nr, nt = corners.shape[0] - 1, corners.shape[1] - 1
    assert rows.shape == (nr + 1, k * nt + 1, 2)
    assert cols.shape == (nt + 1, k * nr + 1, 2)
    np.testing.assert_allclose(rows[:, ::k], corners, rtol=0, atol=1e-15)
    np.testing.assert_allclose(cols[:, ::k],
                               corners.transpose(1, 0, 2),
                               rtol=0,
                               atol=1e-15)
    g = [1 + int(np.argmin(np.abs(fc.EDGE_T - x))) for x in GAUSS]
    for j in range(nt):
      for q in range(2):
        np.testing.assert_allclose(rows[:, k * j + g[q]],
                                   surf0[:, 2 * j + q],
                                   rtol=0,
                                   atol=1e-14)

  def test_arc_rows_stay_within_the_interpolation_bound(self):
    radii, span = np.array([1.0, 1.4]), np.pi / 8
    corners, surf0, surf1 = annulus(radii, np.arange(9) * span)
    rows, _ = fc.edge_polylines(corners, surf0, surf1)
    for i, r in enumerate(radii):
      off = np.abs(np.hypot(*rows[i].T) - r)
      assert off.max() <= cubic_error_bound(r, span)
      assert off.max() > 0  # a cubic is not a circle

  def test_arc_length_within_derived_bound(self):
    # The polyline's length differs from the arc's r * theta by at most the
    # chord deficit of its samples (r delta^3 / 24 each) plus twice the
    # sample error per segment.
    r, span, nt = 2.0, np.pi / 8, 8
    corners, surf0, surf1 = annulus([r, r + 0.1], np.arange(nt + 1) * span)
    rows, _ = fc.edge_polylines(corners, surf0, surf1)
    length = fc.arc_lengths(rows[:1])[0, -1]
    deltas = span * np.diff(np.r_[0.0, fc.EDGE_T, 1.0])
    bound = nt * (np.sum(r * deltas**3 / 24) +
                  2 * len(deltas) * cubic_error_bound(r, span))
    assert abs(length - r * nt * span) <= bound
    # The chords alone fall short by several times that bound.
    assert r * nt * span - nt * 2 * r * np.sin(span / 2) > 4 * bound

  def test_straight_edges_are_reproduced_exactly(self):
    corners, surf0, surf1 = annulus([1.0, 1.25, 2.0], [0.3, 0.9])
    _, cols = fc.edge_polylines(corners, surf0, surf1)
    # Each node column is a ray: radius linear in the edge parameter.
    for j, a in enumerate([0.3, 0.9]):
      t = np.r_[0.0, fc.EDGE_T]
      expect = np.r_[[1.0 + 0.25 * t, 1.25 + 0.75 * t]].ravel()
      np.testing.assert_allclose(np.hypot(*cols[j, :-1].T),
                                 expect,
                                 rtol=0,
                                 atol=1e-14)
      np.testing.assert_allclose(np.arctan2(cols[j, :, 1], cols[j, :, 0]),
                                 a,
                                 rtol=0,
                                 atol=1e-14)

  def test_nodes_by_edges(self):
    corners, surf0, surf1 = annulus([1.0, 1.5], np.linspace(0, 1, 5))
    assert fc.nodes_by_edges(corners, surf0, surf1)
    assert not fc.nodes_by_edges(corners, np.zeros_like(surf0), surf1)
    assert not fc.nodes_by_edges(corners, surf0, np.zeros_like(surf1))

  def test_cell_polygons_walk_each_cells_corners(self):
    rng = np.random.default_rng(3)
    nr, nt = 2, 3
    nodes = rng.normal(size=(nr + 1, nt + 1, 2))
    poly = fc.cell_polygons(nodes, nodes.transpose(1, 0, 2))
    assert poly.shape == (nr, nt, 4, 2)
    for i in range(nr):
      for j in range(nt):
        np.testing.assert_array_equal(poly[i, j], [
            nodes[i, j], nodes[i, j + 1], nodes[i + 1, j + 1], nodes[i + 1, j]
        ])

  def test_curved_cell_polygon_corners(self):
    corners, surf0, surf1 = annulus([1.0, 1.2, 1.3], np.linspace(0, 1, 4))
    rows, cols = fc.edge_polylines(corners, surf0, surf1)
    poly = fc.cell_polygons(rows, cols)
    k = len(fc.EDGE_T) + 1
    assert poly.shape == (2, 3, 4 * k, 2)
    for i in range(2):
      for j in range(3):
        np.testing.assert_allclose(poly[i, j, ::k], [
            corners[i, j], corners[i, j + 1], corners[i + 1, j + 1],
            corners[i + 1, j]
        ],
                                   rtol=0,
                                   atol=1e-15)


def bilinear_coefficients(a, b, c, d, xe, ye):
  """Orthonormal serendipity p = 1 coefficients ``(nr, nt, 4)`` of
  psi = a + b X + c Y + d X Y on the grid with cell edges ``xe``, ``ye``:
  with X = Xc + hx x, the basis 1/2, sqrt(3) x / 2, sqrt(3) y / 2, 3 x y / 2
  takes 2 psi(Xc, Yc), 2 / sqrt(3) (b + d Yc) hx, 2 / sqrt(3) (c + d Xc) hy,
  2 / 3 d hx hy."""
  xc, yc = np.meshgrid(0.5 * (xe[1:] + xe[:-1]),
                       0.5 * (ye[1:] + ye[:-1]),
                       indexing="ij")
  hx, hy = np.meshgrid(0.5 * np.diff(xe), 0.5 * np.diff(ye), indexing="ij")
  return np.stack([
      2 * (a + b * xc + c * yc + d * xc * yc),
      2 / np.sqrt(3) * (b + d * yc) * hx,
      2 / np.sqrt(3) * (c + d * xc) * hy,
      2 / 3 * d * hx * hy,
  ], -1)


@needs_gkeyll
class TestPsiNodes:

  def test_bilinear_field_at_the_corners(self):
    xe = np.array([0.0, 0.1, 0.35, 0.4])  # unequal psi cells
    ye = np.array([-1.0, 0.0, 0.5, 2.0, 2.2])
    a, b, c, d = 0.3, -2.0, 0.7, 1.5
    psi, gap = fc.corner_values(bilinear_coefficients(a, b, c, d, xe, ye),
                                "serendipity", 1)
    X, Y = np.meshgrid(xe, ye, indexing="ij")
    np.testing.assert_allclose(psi,
                               a + b * X + c * Y + d * X * Y,
                               rtol=0,
                               atol=1e-14)
    assert gap <= 1e-14

  def test_gap_of_a_cellwise_constant_field(self):
    # psi = 10 i + j per cell (i along psi): neighbours disagree by 10 across
    # psi and 1 across theta; each node keeps its last cell's value, the one
    # below-left of it where there is one.
    nr, nt = 3, 4
    level = 10.0 * np.arange(nr)[:, None] + np.arange(nt)[None, :]
    coeffs = np.zeros((nr, nt, 4))
    coeffs[..., 0] = 2 * level
    psi, gap = fc.corner_values(coeffs, "serendipity", 1)
    assert gap == pytest.approx(11.0, abs=1e-12)  # the diagonal neighbour
    expect = np.empty((nr + 1, nt + 1))
    for i in range(nr + 1):
      for j in range(nt + 1):
        expect[i, j] = level[max(i - 1, 0), max(j - 1, 0)]
    np.testing.assert_allclose(psi, expect, rtol=0, atol=1e-13)


class TestFluxRows:

  def test_linear_psi(self):
    simag, sibry = -0.4, 0.9
    s = np.array([0.2, 0.5, 0.95, 1.05])  # psi_N of each node row
    psi = simag + (sibry - simag) * s[:, None] * np.ones((1, 5))
    nodes, cells = fc.flux_rows(psi, simag, sibry)
    np.testing.assert_allclose(nodes, s, rtol=0, atol=1e-15)
    np.testing.assert_allclose(cells,
                               0.5 * (s[1:] + s[:-1]),
                               rtol=0,
                               atol=1e-15)

  def test_row_mean_over_theta(self):
    # A psi varying along theta is averaged over the row's nodes.
    psi = np.array([[0.0, 1.0, 2.0], [3.0, 3.0, 6.0]])
    nodes, cells = fc.flux_rows(psi, 0.0, 2.0)
    np.testing.assert_allclose(nodes, [0.5, 2.0], rtol=0, atol=0)
    np.testing.assert_allclose(cells, [1.25], rtol=0, atol=0)


class TestPoloidalLength:

  def test_straight_rows_across_two_blocks(self):
    # Node rows along Z = y_i with nonuniform X nodes; the second block
    # continues the first: L_theta at a cell centre is its X minus X_0.
    y = np.array([0.0, 0.3, 1.0])
    x1, x2 = np.array([0.0, 0.5, 0.7, 1.6]), np.array([1.6, 1.7, 2.5])
    blocks = [
        np.stack(np.broadcast_arrays(x[None, :], y[:, None]), -1)
        for x in (x1, x2)
    ]
    line, starts = fc.centre_line(blocks)
    assert starts == (0, 3)
    np.testing.assert_allclose(
        line[..., 1], 0.5 * (y[1:] + y[:-1])[:, None] * np.ones((1, 6)))
    lengths = fc.arc_lengths(line)
    np.testing.assert_allclose(lengths[:, -1], 2.5, rtol=0, atol=1e-15)
    for x, i0 in ((x1, 0), (x2, 3)):
      lc = fc.at_cell_centres(lengths, i0, len(x) - 1, 1)
      np.testing.assert_allclose(lc,
                                 np.broadcast_to(0.5 * (x[1:] + x[:-1]),
                                                 lc.shape),
                                 rtol=0,
                                 atol=1e-15)

  def test_circular_chord_rows(self):
    # Chords between nodes on circles of radii r_i at equal angle steps: the
    # centre line's nodes lie at the mean radius rho on the same rays, so a
    # cell centre lies (j + 1/2) chords of 2 rho sin(delta / 2) along it.
    radii, delta, nt = np.array([1.0, 1.2, 1.6]), 0.2, 7
    corners, _, _ = annulus(radii, np.arange(nt + 1) * delta)
    line, _ = fc.centre_line([corners])
    lengths = fc.arc_lengths(line)
    rho = 0.5 * (radii[1:] + radii[:-1])
    chord = 2 * rho * np.sin(delta / 2)
    np.testing.assert_allclose(fc.at_cell_centres(lengths, 0, nt, 1),
                               chord[:, None] * (np.arange(nt) + 0.5),
                               rtol=1e-14,
                               atol=0)
    np.testing.assert_allclose(lengths[:, -1], nt * chord, rtol=1e-14)

  def test_circular_curved_rows_within_bound(self):
    # Curved edges: the centre line follows the circle of the mean radius,
    # each sample within the larger row's cubic error of it.
    radii, span, nt = np.array([2.0, 2.1]), np.pi / 8, 8
    corners, surf0, surf1 = annulus(radii, np.arange(nt + 1) * span)
    rows, _ = fc.edge_polylines(corners, surf0, surf1)
    k = len(fc.EDGE_T) + 1
    line, _ = fc.centre_line([rows])
    lengths = fc.arc_lengths(line)
    rho = radii.mean()
    eps = cubic_error_bound(radii.max(), span)
    deltas = span * np.diff(np.r_[0.0, fc.EDGE_T, 1.0])
    bound = nt * (np.sum(rho * deltas**3 / 24) + 2 * len(deltas) * eps)
    assert abs(lengths[0, -1] - rho * nt * span) <= bound
    # Each cell centre (t = 1/2 of its edge) lies half a cell along.
    centres = fc.at_cell_centres(lengths, 0, nt, k)[0]
    assert np.all(np.abs(centres - rho * span * (np.arange(nt) + 0.5)) <= bound)


def test_midplane_row_radii_on_curves_and_chords():
  # Node rows on circles r = 0.5, 0.7 about (R0, 0) at theta = -0.3, 0.1,
  # 0.5: the chord from -0.3 to 0.1 lies r cos(0.2) from the centre and
  # crosses the ray theta = 0 at R0 + r cos(0.2) / cos(0.1). The curves,
  # sampled at 8 points to an edge through theta = 0, cross at R0 + r.
  r0, k = 1.5, 8
  radii = np.array([0.5, 0.7])

  def ring(theta):
    return np.stack([
        r0 + radii[:, None] * np.cos(theta)[None, :],
        radii[:, None] * np.sin(theta)[None, :]
    ], -1)

  nodes = ring(np.linspace(-0.3, 0.5, 3))
  cols = nodes.transpose(1, 0, 2)  # (nt + 1, nr + 1, 2): chords
  layout = SimpleNamespace(nodes=(nodes, ))
  curved = fc.CellEdges("t", ".", (ring(np.linspace(-0.3, 0.5, 2 * k + 1)), ),
                        (cols, ), True)
  chords = fc.CellEdges("t", ".", (nodes, ), (cols, ), False, "no surfaces")
  assert (curved.points_per_edge, chords.points_per_edge) == (k, 1)
  np.testing.assert_allclose(fc.midplane_row_radii(layout, curved, 0, None),
                             r0 + radii,
                             rtol=0,
                             atol=1e-14)
  np.testing.assert_allclose(fc.midplane_row_radii(layout, chords, 0, None),
                             r0 + radii * np.cos(0.2) / np.cos(0.1),
                             rtol=1e-14)
