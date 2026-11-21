"""Tests for ``postgkyl.diagnostics.gk.dt_limits``.

A synthetic single-block, two-species 2x2v run is written whose every rate
has a closed form: uniform ``b.grad z = b``, a potential linear in x and z,
a field strength linear in z, constant metric factors, and uniform velocity
grids. With the drifts off, the CFL rate of a phase-space cell is

    3 b max|v_node| / dz  +  5 b max_mu |q E0 + mu G| / (m dv)

(streaming through its z face, the parallel electric field and mirror force
through its lower v_par face; none for the first v_par cell). The
polarization weight is recovered from a field energy written as ``W0`` times
``1/2 int g^xx J (d_x phi)^2 = 1/2 alpha^2 Lx Lz``, and ``omega_H =
|q| sqrt(n0 / m) gf`` with ``gf = (2 pi / dz) (Lx / pi) / sqrt(W0)``.

Modal coefficients are written in closed form on the orthonormal Legendre
bases: 2D ``1/2, sqrt(3) x / 2, sqrt(3) y / 2, 3 x y / 2``; 1D ``1/sqrt(2),
sqrt(3/2) x``.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from generate_test_data import write_gkyl_dynvector, write_gkyl_field
from postgkyl import gpython

dtl = importlib.import_module("postgkyl.diagnostics.gk.dt_limits")

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

E = 1.602176634e-19
SPECIES = {"elc": (9.10938e-31, -E), "ion": (3.34358e-27, E)}
NX, NZ = 2, 3
LO, UP = [0.0, -1.0], [1.0, 1.0]
DX, DZ = (UP[0] - LO[0]) / NX, (UP[1] - LO[1]) / NZ
LX, LZ = UP[0] - LO[0], UP[1] - LO[1]
B, B0, G = 0.8, 2.0, 0.5  # b.grad z; |B| = B0 + G z (T)
E0, ALPHA = -300.0, 40.0  # phi = E0 z + ALPHA x (V)
N0, W0 = 3e19, 2.5e-7  # density; polarization weight
NV, NMU, V, MU = 4, 3, 4e6, 1e-16  # v_par in [-V, V], mu in [0, MU]
TIMES = (0.0, 1e-6, 2e-6)


def _centres(lo, up, n):
  edges = np.linspace(lo, up, n + 1)
  return 0.5 * (edges[1:] + edges[:-1]), (up - lo) / (2 * n)


def _conf(c0=0.0, cx=0.0, cz=0.0, ncomp=4):
  """Coefficients of ``c0 + cx x + cz z`` on every conf cell; zero padding
  up to ``ncomp`` (vector fields' other components)."""
  xc, _ = _centres(LO[0], UP[0], NX)
  zc, _ = _centres(LO[1], UP[1], NZ)
  out = np.zeros((NX, NZ, ncomp))
  for i, x in enumerate(xc):
    for j, z in enumerate(zc):
      out[i, j, :4] = (2 * (c0 + cx * x + cz * z), cx * DX / np.sqrt(3),
                       cz * DZ / np.sqrt(3), 0.0)
  return out


def _write_conf(path, values, **kw):
  write_gkyl_field(path, [NX, NZ], LO, UP, values, 1, "serendipity", **kw)


def _velocity_map():
  vc, hv = _centres(-V, V, NV)
  mc, hm = _centres(0.0, MU, NMU)
  out = np.zeros((NV, NMU, 4))
  out[..., 0] = vc[:, None] * np.sqrt(2)
  out[..., 1] = hv * np.sqrt(2 / 3)
  out[..., 2] = mc[None, :] * np.sqrt(2)
  out[..., 3] = hm * np.sqrt(2 / 3)
  return out


@pytest.fixture
def run(tmp_path):
  p = lambda what: tmp_path / f"tiny-{what}.gkyl"  # noqa: E731
  for name, value in (("gxxj", 1.0), ("cmag", 1.0), ("jacobtot_inv", 1.0),
                      ("bmag", B0), ("B3overB", B)):
    _write_conf(p(f"geo_int_{name}"), _conf(value))
  for name in ("dualcurlbhatoverB", "bioverJB"):
    _write_conf(p(f"geo_int_{name}"), _conf(ncomp=12))
  _write_conf(p("geo_corn_bmag"), _conf(B0, cz=G))
  for s, (m, q) in SPECIES.items():
    write_gkyl_field(p(f"{s}_mapc2p_vel"), [NV, NMU], [-1.0, 0.0], [1.0, 1.0],
                     _velocity_map(), 1, "serendipity")
    for k, t in enumerate(TIMES):
      _write_conf(p(f"{s}_M0_{k}"),
                  _conf(N0),
                  time=t,
                  frame=k,
                  metadata={
                      "mass": m,
                      "charge": q
                  })
  for k, t in enumerate(TIMES):
    _write_conf(p(f"field_{k}"), _conf(cx=ALPHA, cz=E0), time=t, frame=k)
  energy = W0 * 0.5 * ALPHA**2 * LX * LZ
  write_gkyl_dynvector(p("field_energy"), np.array(TIMES),
                       np.full((3, 1), energy))
  return tmp_path


def _expected():
  """The rebuilt limits, from the closed forms in the module docstring."""
  vc, hv = _centres(-V, V, NV)
  mc, hm = _centres(0.0, MU, NMU)
  v_node = vc[-1] + hv * np.sqrt(0.6)  # outermost Gauss node of the last cell
  mu_nodes = np.concatenate([mc - hm / np.sqrt(3), mc + hm / np.sqrt(3)])
  mu_top = mc[-1] + hm / np.sqrt(3)
  streaming = 3 * B * v_node / DZ
  out = {}
  for s, (m, q) in SPECIES.items():
    vpar = 5 * B * np.abs(q * E0 + mu_nodes * G).max() / (m * 2 * hv)
    out[s] = dict(rate=streaming + vpar,
                  streaming=streaming,
                  vpar=vpar,
                  epar_own=5 * B * abs(q * E0) / (m * 2 * hv),
                  mirror_own=5 * B * mu_top * G / (m * 2 * hv))
  gf = (2 * np.pi / DZ) * (LX / np.pi) / np.sqrt(W0)
  omega = max(abs(q) * np.sqrt(N0 / m) * gf for m, q in SPECIES.values())
  return out, mu_top, 1.7 / omega


@needs_gkeyll
def test_rebuild_matches_closed_form(run):
  exp, mu_top, dt_h = _expected()
  lim = dtl.dt_limits("tiny", ["elc", "ion"], path=str(run), drifts="off")
  e = exp["elc"]
  assert e["rate"] > exp["ion"]["rate"]  # the electrons bind
  np.testing.assert_allclose(lim.dt_cfl, 1.0 / e["rate"], rtol=1e-13)
  np.testing.assert_allclose(lim.dt_omegaH, dt_h, rtol=1e-12)
  # The binding cell: the last v_par and mu cells, first conf cell.
  np.testing.assert_array_equal(lim.cfl_cell,
                                [[0, 0, 0, 0, NV - 1, NMU - 1]] * 3)
  # q E0 and mu G have one sign for electrons: the v_par rate splits by size.
  qe = abs(SPECIES["elc"][1] * E0)
  np.testing.assert_allclose(lim.share["streaming"],
                             e["streaming"] / e["rate"],
                             rtol=1e-13)
  np.testing.assert_allclose(lim.share["epar"],
                             e["vpar"] * qe / (qe + mu_top * G) / e["rate"],
                             rtol=1e-13)
  np.testing.assert_allclose(lim.share["mirror"],
                             e["vpar"] * mu_top * G / (qe + mu_top * G) /
                             e["rate"],
                             rtol=1e-13)
  for term in ("exb", "magnetic", "collisions", "diffusion"):
    np.testing.assert_array_equal(lim.share[term], 0.0)
    assert np.all(np.isinf(lim.alone[term]))
  np.testing.assert_allclose(lim.alone["streaming"],
                             1.0 / e["streaming"],
                             rtol=1e-13)
  np.testing.assert_allclose(lim.alone["epar"], 1.0 / e["epar_own"], rtol=1e-13)
  np.testing.assert_allclose(lim.alone["mirror"],
                             1.0 / e["mirror_own"],
                             rtol=1e-13)
  assert dt_h < 1.0 / e["rate"]  # here omega_H sets the step
  assert lim.binding == ("omega_H", ) * 3
  assert lim.species == ("elc", "ion")
  assert lim.vel_cells == {"0|elc": (NV, NMU), "0|ion": (NV, NMU)}
  np.testing.assert_allclose(lim.time, TIMES)
  assert any("W = 2.5e-07" in n for n in lim.notes)


@needs_gkeyll
def test_steps_taken_and_agreement(run):
  exp, _, dt_h = _expected()
  rebuilt = min(1.0 / exp["elc"]["rate"], dt_h)
  # The steps taken after each frame: as rebuilt, then 3 % short.
  write_gkyl_dynvector(run / "tiny-dt.gkyl",
                       np.array(TIMES) + 1e-9,
                       np.array([[rebuilt], [rebuilt], [0.97 * rebuilt]]))
  lim = dtl.dt_limits("tiny", ["elc", "ion"], path=str(run), drifts="off")
  np.testing.assert_allclose(lim.taken, rebuilt * np.array([1, 1, 0.97]))
  np.testing.assert_allclose(lim.agreement, (0.985, 0.97, 1.0), rtol=1e-12)
  assert lim.reproduced  # within 2 x 1 % of 1
  one = dtl.dt_limits("tiny", ["elc", "ion"],
                      path=str(run),
                      frame="2",
                      drifts="off")
  np.testing.assert_array_equal(one.frames, [2])
  assert np.isnan(one.agreement[0])  # no frame after a first one


@needs_gkeyll
def test_settings_and_closures(run):
  lim = dtl.dt_limits("tiny", ["elc", "ion"],
                      path=str(run),
                      frame="1",
                      cfl_frac=0.5,
                      cfl_frac_omegaH=3.4,
                      drifts="off")
  exp, _, dt_h = _expected()
  np.testing.assert_allclose(lim.dt_cfl, 0.5 / exp["elc"]["rate"], rtol=1e-13)
  np.testing.assert_allclose(lim.dt_omegaH, dt_h * 3.4 / 1.7, rtol=1e-12)
  # One charged species: the library applies no omega_H limit.
  alone = dtl.dt_limits("tiny", ["elc"], path=str(run), drifts="off")
  assert np.all(np.isinf(alone.dt_omegaH))
  # A uniform diffusivity adds 2 c0(D g^xx J) (2 / dx)^2 = 4 D (2 / dx)^2.
  diff = dtl.dt_limits("tiny", ["elc", "ion"],
                       path=str(run),
                       frame="1",
                       drifts="off",
                       diffusivity=7.0)
  np.testing.assert_allclose(1 / diff.dt_cfl - 1 / lim.dt_cfl * 0.5,
                             4 * 7.0 * (2 / DX)**2,
                             rtol=1e-12)


@needs_gkeyll
def test_binding_cfl_term(run):
  # A 400 times larger polarization weight makes omega_H 20 times slower, so
  # the CFL limit binds; E_par carries the largest share of the cell's rate.
  exp, mu_top, dt_h = _expected()
  write_gkyl_dynvector(run / "tiny-field_energy.gkyl", np.array(TIMES),
                       np.full((3, 1), 400 * W0 * 0.5 * ALPHA**2 * LX * LZ))
  lim = dtl.dt_limits("tiny", ["elc", "ion"], path=str(run), drifts="off")
  np.testing.assert_allclose(lim.dt_omegaH, 20 * dt_h, rtol=1e-12)
  assert 20 * dt_h > 1.0 / exp["elc"]["rate"]
  assert abs(SPECIES["elc"][1] * E0) > mu_top * G  # E_par's part is larger
  assert lim.binding == ("epar", ) * 3


def test_direction_splits_by_size_at_the_fastest_node():
  a = np.array([[1.0, -3.0]])
  b = np.array([[2.0, 1.0]])
  parts, own = dtl._direction({"a": a, "b": b}, 10.0)
  # |a + b| = (3, 2): the first node is the fastest, where |a|:|b| = 1:2.
  np.testing.assert_allclose(parts["a"], [10.0])
  np.testing.assert_allclose(parts["b"], [20.0])
  np.testing.assert_allclose(own["a"], [30.0])
  np.testing.assert_allclose(own["b"], [20.0])


@needs_gkeyll
def test_refuses_other_bases(run):
  write_gkyl_field(run / "tiny-geo_int_gxxj.gkyl", [NX, NZ], LO, UP,
                   np.zeros((NX, NZ, 8)), 2, "serendipity")
  with pytest.raises(ValueError, match="2D p = 1 serendipity"):
    dtl.dt_limits("tiny", ["elc", "ion"], path=str(run), drifts="off")
