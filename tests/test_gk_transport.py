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


class TestGyroBohm:
  """The gyro-Bohm normalization of D and chi, on nodal profiles."""

  x = np.linspace(0.0, 1.0, 5)
  m_i, q_i, m_e = 3.343e-27, 1.602e-19, 9.109e-31
  gxx = 1.7

  def _window(self, n, T, dn_dx, dT_dx, mass, charge, bmag=2.0, frames=(1, 2)):
    res = {"frames": list(frames), "n": n, "T": T, "dn_dx": dn_dx, "dT_dx": dT_dx,
           "gxx": self.gxx*np.ones_like(n), "bmag": bmag*np.ones_like(n),
           "mass": mass, "charge": charge}
    res.update(gkt.transport_coefficients(3.0e19*np.ones_like(n), 2.0e3*np.ones_like(n),
                                          n, T, res["gxx"], dn_dx, dT_dx))
    return res

  def _ion(self, **kw):
    n, T = 2.0e19*(1.0 - 0.3*self.x), 3.0e-17*(1.0 - 0.5*self.x)
    return self._window(n, T, -0.6e19*np.ones_like(n), -1.5e-17*np.ones_like(n),
                        self.m_i, self.q_i, **kw)

  def _elc(self, **kw):
    n, T = 2.0e19*(1.0 - 0.3*self.x), 5.0e-17*(1.0 - 0.2*self.x)
    return self._window(n, T, -0.6e19*np.ones_like(n), -1.0e-17*np.ones_like(n),
                        self.m_e, -self.q_i, **kw)

  def _expected(self, res, Te, bmag=2.0):
    c_s = np.sqrt(Te/self.m_i)
    D0 = (c_s*self.m_i/(self.q_i*bmag))**2*c_s
    L_n = -res["n"]/(np.sqrt(self.gxx)*res["dn_dx"])
    L_T = -res["T"]/(np.sqrt(self.gxx)*res["dT_dx"])
    return res["D"]*L_n/D0, res["chi"]*L_T/D0

  def test_adiabatic_electrons_use_Ti_over_Te(self):
    results = {"ion": [self._ion()]}
    gkt.gyro_bohm_normalize(results, Ti_over_Te=2.0)
    res = results["ion"][0]
    D_gB, chi_gB = self._expected(res, res["T"]/2.0)
    assert np.allclose(res["D_gB"], D_gB, rtol=1e-12)
    assert np.allclose(res["chi_gB"], chi_gB, rtol=1e-12)

  def test_kinetic_electrons_use_their_temperature_and_the_ion_mass(self):
    """Every species is normalized by rho_s^2 c_s built on T_e and m_i,
    over its own gradient lengths."""
    results = {"elc": [self._elc()], "ion": [self._ion()]}
    gkt.gyro_bohm_normalize(results, Ti_over_Te=7.0)  # Ignored: electrons are listed.
    Te = results["elc"][0]["T"]
    for species in ("elc", "ion"):
      res = results[species][0]
      D_gB, chi_gB = self._expected(res, Te)
      assert np.allclose(res["D_gB"], D_gB, rtol=1e-12), species
      assert np.allclose(res["chi_gB"], chi_gB, rtol=1e-12), species

  def test_reference_values_replace_the_profiles(self):
    results = {"ion": [self._ion(bmag=5.0)]}
    gkt.gyro_bohm_normalize(results, Te_ref=4.0e-17, bmag_ref=2.0)
    res = results["ion"][0]
    D_gB, chi_gB = self._expected(res, 4.0e-17, bmag=2.0)
    assert np.allclose(res["D_gB"], D_gB, rtol=1e-12)
    assert np.allclose(res["chi_gB"], chi_gB, rtol=1e-12)

  def test_needs_an_ion_species(self):
    with pytest.raises(ValueError, match="ion species"):
      gkt.gyro_bohm_normalize({"elc": [self._elc()]})

  def test_needs_the_magnetic_field(self):
    res = self._ion()
    res["bmag"] = None
    with pytest.raises(ValueError, match="geo_int_bmag"):
      gkt.gyro_bohm_normalize({"ion": [res]})

  def test_needs_the_mass(self):
    res = self._ion()
    res["mass"] = None
    with pytest.raises(ValueError, match="--extra mass="):
      gkt.gyro_bohm_normalize({"ion": [res]})


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
_BMAG = 1.9
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
    "geo_int_bmag": lambda f: _const(_BMAG),
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

  def test_command_pushes_the_gyro_bohm_diffusivities(self, synthetic_sim):
    """Adiabatic electrons (ion only): T_e = T_i/Ti_over_Te, B from geo_int_bmag."""
    ctx = click.core.Context(cli)
    ctx.obj = {"data": cmd.DataSpace(), "verbose": False, "compgrid": None}
    ctx.invoke(cmd.gk_transport, name=_NAME, species=_SPECIES, path=synthetic_sim,
               outputs="D_gB,chi_gB", extra=f"mass={_MASS},Ti_over_Te=2")
    out = {d.get_tag(): d for d in ctx.obj["data"].iterator()}
    assert set(out) == {"transport_D_gB", "transport_chi_gB"}

    x = _nodes(out["transport_D_gB"].get_grid()[0])
    exp = _expected(x)
    c_s = np.sqrt(0.5*exp["T"]/_MASS)
    D0 = (c_s*_MASS/(1.0*_BMAG))**2*c_s  # The synthetic charge is 1.
    L_n = exp["n"]/(np.sqrt(_GXX)*_GN)
    L_T = exp["T"]/(np.sqrt(_GXX)*_GT)
    assert np.allclose(out["transport_D_gB"].get_values()[..., 0], exp["D"]*L_n/D0, rtol=1e-8)
    assert np.allclose(out["transport_chi_gB"].get_values()[..., 0], exp["chi"]*L_T/D0,
                       rtol=1e-8)

  def test_command_rejects_unknown_outputs(self, synthetic_sim):
    ctx = click.core.Context(cli)
    ctx.obj = {"data": cmd.DataSpace(), "verbose": False, "compgrid": None}
    with pytest.raises(click.exceptions.UsageError):
      ctx.invoke(cmd.gk_transport, name=_NAME, species=_SPECIES, path=synthetic_sim,
                 outputs="gamma,nope")


@_needs_dgops
class TestLocalTransportQuantities:
  """The local D, chi, D_gB and chi_gB of gk-load-quantity on the synthetic simulation.

  At frame 3 (scale 1), with uniform (y,z) and linear n, T profiles:
    D = n vE/(gxx GN),  chi = q/(n gxx GT),  q = Q - 1.5*T*Gamma,
    D_gB = D L_n/(rho_s^2 c_s),  chi_gB = chi L_T/(rho_s^2 c_s),
  with L_n = n/(sqrt(gxx) GN), L_T = T/(sqrt(gxx) GT) and, for adiabatic
  electrons, rho_s^2 c_s = (T/Ti_over_Te)^(3/2) sqrt(m)/(q^2 B^2) (q = 1 here).
  D is a product of linear and constant fields, hence exact in p1; the others
  involve weak inverses and powers of linear fields, so they are compared to
  the exact values at the cell centers with a tolerance.
  """
  frame = _FRAMES[0]

  def _load(self, path, quantity, extra):
    ctx = click.core.Context(cli)
    ctx.obj = {"data": cmd.DataSpace(), "verbose": False, "compgrid": None}
    ctx.invoke(cmd.gk_load_quantity, quantity=quantity, name=_NAME, species=_SPECIES,
               frame=str(self.frame), path=path, extra=extra)
    (out,) = list(ctx.obj["data"].iterator())
    assert out.get_num_dims() == 3
    return out.get_values()[..., 0]*_PSI0  # Cell averages.

  def _exact(self, ti_over_te=1.0):
    x = _nodes(_grid()[0])[:, None, None]*np.ones(_CELLS)
    exp = _expected(x)
    scale = _frame_scale(self.frame)/np.mean([_frame_scale(f) for f in _FRAMES])
    D, chi = scale*exp["D"], scale*exp["chi"]
    D0 = (exp["T"]/ti_over_te)**1.5*np.sqrt(_MASS)/_BMAG**2
    L_n = exp["n"]/(np.sqrt(_GXX)*_GN)
    L_T = exp["T"]/(np.sqrt(_GXX)*_GT)
    return {"D": D, "chi": chi, "D_gB": D*L_n/D0, "chi_gB": chi*L_T/D0}

  def test_D_is_exact(self, synthetic_sim):
    assert np.allclose(self._load(synthetic_sim, "D", f"mass={_MASS}"), self._exact()["D"],
                       rtol=1e-10)

  @pytest.mark.parametrize("quantity", ["chi", "D_gB", "chi_gB"])
  def test_matches_the_closed_form(self, synthetic_sim, quantity):
    vals = self._load(synthetic_sim, quantity, f"mass={_MASS},Ti_over_Te=2")
    assert np.allclose(vals, self._exact(ti_over_te=2.0)[quantity], rtol=2e-2)

  def test_gyro_bohm_references_replace_the_profiles(self, synthetic_sim):
    """With Te_ref and bmag_ref, rho_s^2 c_s is a constant."""
    Te_ref, bmag_ref = 3.0e-17, 2.5
    vals = self._load(synthetic_sim, "D_gB",
                      f"mass={_MASS},Te_ref={Te_ref},bmag_ref={bmag_ref}")
    exact = self._exact()
    x = _nodes(_grid()[0])[:, None, None]*np.ones(_CELLS)
    L_n = _expected(x)["n"]/(np.sqrt(_GXX)*_GN)
    D0 = Te_ref**1.5*np.sqrt(_MASS)/bmag_ref**2
    assert np.allclose(vals, exact["D"]*L_n/D0, rtol=2e-2)
