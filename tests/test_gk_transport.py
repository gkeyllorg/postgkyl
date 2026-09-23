"""Postgkyl module for testing the radial transport diagnostics (gk-transport).

Two levels are checked:
  - transport_coefficients, the pure algebra on nodal profiles, against a
    manufactured case with linear n and T profiles;
  - compute_transport and the gk-transport command end to end, on a synthetic
    3x simulation whose fluxes and profiles are known in closed form.

The synthetic simulation has profiles linear in x and uniform in (y,z),
  n = N0 - GN*x,  T = T0 - GT*x,  M2 = M20 + GM2*x,
a potential phi = a*y + e*z and uniform geometry (J, 1/(J*B), b_i, g^xx), so
that v_E^x = (b_y*e - b_z*a)/(J*B) is uniform. Every weak DG product is then
exact, and so are the expected values:
  Gamma = n*vE,  Q = (m/2)*M2*vE,  q = Q - conv*T*Gamma,
  D = Gamma/(gxx*GN),  chi = q/(n*gxx*GT).
"""
import os

import click
import numpy as np
import pytest

import postgkyl.commands as cmd
import postgkyl.tools.gk_transport as gkt
import postgkyl.utils.gk_quantities.gkquantity as gkquantity
from postgkyl.data import GData
from postgkyl.pgkyl import cli

try:
  from postgkyl.tools.gkeyll_dg_ops import GkeyllDGops
  GkeyllDGops()
  _DGOPS_AVAILABLE = True
except Exception:  # noqa: BLE001 - any failure means the lib is unusable here
  _DGOPS_AVAILABLE = False

_needs_dgops = pytest.mark.skipif(not _DGOPS_AVAILABLE, reason="requires the gkylsoft DG library")


class TestTransportCoefficients:
  """The algebra of q, D and chi on nodal profiles."""

  x = np.linspace(0.0, 1.0, 9)
  n0, gn, t0, gt, gxx = 2.0e19, 1.5e19, 3.0e-17, 2.0e-17, 1.7

  def _profiles(self):
    n = self.n0 - self.gn*self.x
    T = self.t0 - self.gt*self.x
    gamma = 4.0e19*(1.0 + self.x)
    Q = 9.0*gamma*T
    return gamma, Q, n, T

  def test_manufactured_profiles(self):
    gamma, Q, n, T = self._profiles()
    res = gkt.transport_coefficients(gamma, Q, n, T, self.gxx, -self.gn*np.ones_like(n),
                                     -self.gt*np.ones_like(n), conv=1.5)
    q = Q - 1.5*T*gamma
    assert np.allclose(res["q"], q)
    assert np.allclose(res["D"], gamma/(self.gxx*self.gn))
    assert np.allclose(res["chi"], q/(n*self.gxx*self.gt))
    assert np.all(res["D"] > 0.0) and np.all(res["chi"] > 0.0)

  def test_conv_zero_keeps_the_energy_flux(self):
    gamma, Q, n, T = self._profiles()
    res = gkt.transport_coefficients(gamma, Q, n, T, self.gxx, -np.ones_like(n),
                                     -np.ones_like(n), conv=0.0)
    assert np.allclose(res["q"], Q)

  def test_flat_gradients_are_masked(self):
    gamma, Q, n, T = self._profiles()
    dn_dx = -self.gn*np.cos(np.pi*self.x)  # Vanishes at x = 0.5.
    res = gkt.transport_coefficients(gamma, Q, n, T, self.gxx, dn_dx, np.zeros_like(n))
    assert np.isnan(res["D"][4])
    assert np.all(np.isfinite(np.delete(res["D"], 4)))
    assert np.all(np.isnan(res["chi"])), "a flat T profile has no chi"


# --- Synthetic 3x simulation ---------------------------------------------------

_NAME = "gksynth"
_SPECIES = "ion"
_FRAMES = (3, 4, 5)
_CELLS = (6, 4, 3)
_LENGTHS = (0.8, 1.3, 2.1)
_NUM_BASIS = 8
_PSI0 = 2.0**-1.5

_MASS = 3.343e-27
_N0, _GN = 3.0e19, 1.2e19
_T0, _GT = 4.0e-17, 1.5e-17
_M20, _GM2 = 5.0e10, 2.0e10
_PHI_Y, _PHI_Z = 250.0, -40.0
_B_Y, _B_Z = 0.3, 0.95
_JACOBGEO = 1.6
_JACOBTOT_INV = 0.45
_GXX = 2.2
_VE_X = (_B_Y*_PHI_Z - _B_Z*_PHI_Y)*_JACOBTOT_INV


def _grid():
  return [np.linspace(0.0, L, n + 1) for L, n in zip(_LENGTHS, _CELLS)]

def _linear(comps, frame_scale=1.0) -> GData:
  """Field whose components are offset + sum_d slope_d*x_d (exact in p1)."""
  centers = np.meshgrid(*[0.5*(g[:-1] + g[1:]) for g in _grid()], indexing="ij")
  values = np.zeros((*_CELLS, _NUM_BASIS*len(comps)))
  for comp, (offset, slopes) in enumerate(comps):
    base = comp*_NUM_BASIS
    values[..., base] = frame_scale*(offset + sum(s*c for s, c in zip(slopes, centers)))/_PSI0
    for d, slope in enumerate(slopes):
      dx = _LENGTHS[d]/_CELLS[d]
      values[..., base + 1 + d] = frame_scale*slope*dx/(2.0*np.sqrt(3.0)*_PSI0)
  gdata = GData(ctx={"poly_order": 1, "basis_type": "serendipity", "mass": _MASS,
                     "charge": 1.0})
  gdata.push(_grid(), values)
  return gdata

def _const(*avgs) -> GData:
  return _linear([(a, (0.0, 0.0, 0.0)) for a in avgs])

def _frame_scale(frame: int) -> float:
  """The fluxes scale with the frame so the time average is not trivially one frame."""
  return 1.0 + 0.1*(frame - _FRAMES[0])

def _synthetic_files():
  """Map file-name suffix -> GData factory for the synthetic simulation."""
  return {
    "geo_int_jacobgeo": lambda f: _const(_JACOBGEO),
    "geo_int_jacobtot_inv": lambda f: _const(_JACOBTOT_INV),
    "geo_int_b_i": lambda f: _const(0.0, _B_Y, _B_Z),
    "geo_int_gij": lambda f: _const(_GXX, 0.0, 0.0, 1.0, 0.0, 1.0),
    "field": lambda f: _linear([(0.0, (0.0, _PHI_Y, _PHI_Z))], _frame_scale(f)),
    "M0": lambda f: _linear([(_N0, (-_GN, 0.0, 0.0))]),
    "M2": lambda f: _linear([(_M20, (_GM2, 0.0, 0.0))]),
    "MaxwellianMoments": lambda f: _linear([(_N0, (-_GN, 0.0, 0.0)), (0.0, (0.0, 0.0, 0.0)),
                                            (_T0/_MASS, (-_GT/_MASS, 0.0, 0.0))]),
  }

@pytest.fixture
def synthetic_sim(tmp_path, monkeypatch):
  """Write marker files for a synthetic 3x simulation and serve its data."""
  path = str(tmp_path)
  factories = _synthetic_files()
  for stem in factories:
    if stem.startswith("geo_"):
      open(os.path.join(path, f"{_NAME}-{stem}.gkyl"), "w").close()
    elif stem == "field":
      for f in _FRAMES:
        open(os.path.join(path, f"{_NAME}-field_{f}.gkyl"), "w").close()
    else:
      for f in _FRAMES:
        open(os.path.join(path, f"{_NAME}-{_SPECIES}_{stem}_{f}.gkyl"), "w").close()

  def _load(file_name=None, *args, **kwargs):
    base = os.path.basename(file_name)[len(_NAME) + 1:-len(".gkyl")]
    if base.startswith("geo_"):
      return factories[base](None)
    stem, _, frame = base.rpartition("_")
    stem = stem[len(_SPECIES) + 1:] if stem.startswith(f"{_SPECIES}_") else stem
    return factories[stem](int(frame))

  monkeypatch.setattr(gkquantity, "GData", _load)
  monkeypatch.setattr(gkt, "GData", lambda file_name=None, **kw: (
    _load(file_name) if file_name else GData(**kw)))
  return path

def _expected(x, conv=1.5):
  scale = np.mean([_frame_scale(f) for f in _FRAMES])
  n = _N0 - _GN*x
  T = _T0 - _GT*x
  gamma = n*_VE_X*scale
  Q = 0.5*_MASS*(_M20 + _GM2*x)*_VE_X*scale
  q = Q - conv*T*gamma
  return {"n": n, "T": T, "gamma": gamma, "Q": Q, "q": q, "gxx": _GXX*np.ones_like(x),
          "D": gamma/(_GXX*_GN), "chi": q/(n*_GXX*_GT)}

def _nodes(grid_edges):
  return 0.5*(grid_edges[:-1] + grid_edges[1:])


def test_electrostatic_run_falls_back_to_the_ExB_flux(synthetic_sim):
  """Without apar output, the total fluxes use their electrostatic source combo
  even though some of the electromagnetic sources (M0, M1 via MaxwellianMoments)
  exist, i.e. a missing source after the first one invalidates its combo."""
  from postgkyl.utils.gk_quantities.registry import gk_quant_registry
  for qname in ("part_flux", "energy_flux"):
    combo_idx, frames = gk_quant_registry.get(qname).get_avail_source(
      synthetic_sim + "/", _NAME, _SPECIES, ":")
    assert combo_idx == 1, qname
    assert frames == list(_FRAMES)


@_needs_dgops
class TestComputeTransport:

  def test_time_averaged_profiles(self, synthetic_sim):
    (res,) = gkt.compute_transport(synthetic_sim, _NAME, _SPECIES, frame=":",
                                   extra={"mass": _MASS})
    assert res["frames"] == list(_FRAMES)
    x = _nodes(res["grid"][0])
    for key, expected in _expected(x).items():
      assert np.allclose(res[key], expected, rtol=1e-10), key
    assert np.all(np.sign(res["D"]) == np.sign(_VE_X)), "an inward flux down the gradient gives D < 0"

  def test_frame_selection_and_per_frame(self, synthetic_sim):
    results = gkt.compute_transport(synthetic_sim, _NAME, _SPECIES, frame="4:6",
                                    extra={"mass": _MASS}, per_frame=True)
    assert [r["frames"] for r in results] == [[4], [5]]
    x = _nodes(results[1]["grid"][0])
    n = _N0 - _GN*x
    assert np.allclose(results[1]["gamma"], n*_VE_X*_frame_scale(5), rtol=1e-10)

  def test_uniform_fluxes_have_no_turbulent_part(self, synthetic_sim):
    (res,) = gkt.compute_transport(synthetic_sim, _NAME, _SPECIES, fluct="yz",
                                   extra={"mass": _MASS})
    assert np.allclose(res["gamma"], 0.0, atol=1e-8*_N0*abs(_VE_X))

  def test_command_pushes_the_requested_outputs(self, synthetic_sim):
    ctx = click.core.Context(cli)
    ctx.obj = {"data": cmd.DataSpace(), "verbose": False, "compgrid": None}
    ctx.invoke(cmd.gk_transport, name=_NAME, species=_SPECIES, path=synthetic_sim,
               outputs="gamma,D,chi", extra=f"mass={_MASS}")
    out = {d.get_tag(): d for d in ctx.obj["data"].iterator()}
    assert set(out) == {"transport_gamma", "transport_D", "transport_chi"}
    grid = out["transport_D"].get_grid()[0]
    assert np.allclose(out["transport_D"].get_values()[..., 0], _expected(_nodes(grid))["D"],
                       rtol=1e-10)
    assert "D" in out["transport_D"].get_label()

  def test_command_rejects_unknown_outputs(self, synthetic_sim):
    ctx = click.core.Context(cli)
    ctx.obj = {"data": cmd.DataSpace(), "verbose": False, "compgrid": None}
    with pytest.raises(click.exceptions.UsageError):
      ctx.invoke(cmd.gk_transport, name=_NAME, species=_SPECIES, path=synthetic_sim,
                 outputs="gamma,nope")
