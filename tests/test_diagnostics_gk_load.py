"""Tests for the gyrokinetic loader stack:
``postgkyl.diagnostics.gk.{distf,quantity,quantities,registry,
load_quantity}``.

Ported/extended from ``tests_bak/test_gk_load_quantity.py`` (the registry
smoke test, using the same "synthetic constant DG field + monkeypatched
``GData``" technique) and the ``TestResolveFrames``/``TestLoadGkDistf``
classes of ``tests_bak/test_loader.py`` (``pg.load.gk_distf``'s dispatch
tests do not port: this architecture has no ``pg.load`` namespace object --
``load_distf``/``resolve_frames`` are plain free functions, tested
directly). Real end-to-end coverage uses the ``rt_gk_tcv_iwl*`` fixtures
staged in ``tests/test_data`` for this layer.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from postgkyl import gpython
from postgkyl.gdata import GData, GDataGroup
from postgkyl.gdatastate.gdatastate import GDataState
from postgkyl.diagnostics.gk import distf, quantities as ff, quantity as qmod, utils
from postgkyl.diagnostics.gk.load_quantity import (available_quantities,
                                                   load_quantity)
from postgkyl.diagnostics.gk.registry import gk_quant_registry

needs_gkeyll = pytest.mark.skipif(
    not gpython.available(), reason="no compiled Gkeyll (libg0core.so) found")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "tests", "test_data")
GK_NAME = "rt_gk_tcv_iwl_1x2v_p1"
HMOM_NAME = "rt_gk_tcv_iwl_adapt_source_1x2v_p1"


def _field(values, grid=None, **ctx):
  """A pre-interpolated (field-domain) dataset for unit-testing the
  ``fetch_*`` combinators without needing the compiled shim."""
  d = GDataState(ctx=dict(ctx, interpolated=True))
  values = np.asarray(values, dtype=np.float64)
  if grid is None:
    grid = [
        np.arange(values.shape[ax] + 1, dtype=np.float64)
        for ax in range(values.ndim - 1)
    ]
  d.push(grid, values)
  return d


class TestResolveFrames:
  """Ported from tests_bak/test_loader.py's TestResolveFrames."""

  def test_single_int(self):
    assert distf.resolve_frames(5, name="n", species="ion") == [5]

  def test_list(self):
    assert distf.resolve_frames([1, 2, 3], name="n", species="ion") == [1, 2, 3]

  def test_csv_string(self):
    assert distf.resolve_frames("0,2,4", name="n", species="ion") == [0, 2, 4]

  def test_single_element_list(self):
    assert distf.resolve_frames([7], name="n", species="ion") == [7]

  def test_range_discovers_files(self, tmp_path, monkeypatch):
    for f in (0, 1, 2, 3):
      (tmp_path / f"sim-ion_{f}.gkyl").touch()
    monkeypatch.chdir(tmp_path)
    assert distf.resolve_frames("1:3", name="sim", species="ion") == [1, 2]
    assert distf.resolve_frames(":", name="sim", species="ion") == [0, 1, 2, 3]
    assert distf.resolve_frames("0:4:2", name="sim", species="ion") == [0, 2]

  def test_numeric_string(self):
    assert distf.resolve_frames("7", name="n", species="ion") == [7]

  def test_range_without_matching_files_has_a_clear_error(
      self, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="No distribution frames found"):
      distf.resolve_frames(":", name="sim", species="ion")

  def test_range_requires_a_positive_step(self, tmp_path, monkeypatch):
    (tmp_path / "sim-ion_0.gkyl").touch()
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
      distf.resolve_frames("::0", name="sim", species="ion")


class TestLoadGkDistfFrames:
  """The public loader resolves frame syntax around the per-frame core."""

  def _stub(self, monkeypatch):
    calls = []

    def fake_load_distf_frame(*, frame, tag, **kwargs):
      calls.append(frame)
      data = GData(tag=tag, ctx={"frame": frame})
      data.push([np.array([0.0, 1.0])], np.array([[float(frame)]]))
      return data

    monkeypatch.setattr(distf, "_load_distf_frame", fake_load_distf_frame)
    return calls

  def test_integer_frame_returns_one_dataset(self, monkeypatch):
    calls = self._stub(monkeypatch)
    out = distf.load_distf("sim", "ion", 3)
    assert isinstance(out, GData)
    assert calls == [3]

  def test_csv_frames_return_a_labelled_fluent_group(self, monkeypatch):
    calls = self._stub(monkeypatch)
    out = distf.load_distf("sim", "ion", "0,2,4")
    assert isinstance(out, GDataGroup)
    assert calls == [0, 2, 4]
    assert [data.label for data in out] == ["0", "2", "4"]

  def test_single_element_list_still_returns_a_group(self, monkeypatch):
    self._stub(monkeypatch)
    out = distf.load_distf("sim", "ion", [7])
    assert isinstance(out, GDataGroup)
    assert len(out) == 1

  def test_range_loads_only_discovered_frames(self, tmp_path, monkeypatch):
    calls = self._stub(monkeypatch)
    for frame in (0, 2, 5):
      (tmp_path / f"sim-ion_fdot_{frame}.gkyl").touch()
    monkeypatch.chdir(tmp_path)
    out = distf.load_distf("sim", "ion", "0:5", suffix="fdot")
    assert isinstance(out, GDataGroup)
    assert calls == [0, 2]


class TestLoadGkDistfKeywordOnly:
  """``load_distf``'s options must be keyword-only (PYTHON_PRINCIPLES #7 /
  doctrine IV) so a caller can never silently swap two boolean flags by
  passing them positionally."""

  def test_tag_cannot_be_passed_positionally(self):
    with pytest.raises(TypeError):
      distf.load_distf("sim", "ion", 0, "f")


@needs_gkeyll
class TestLoadGkDistfReal:
  """End-to-end against the staged rt_gk_tcv_iwl_1x2v_p1 fixtures.

  ``mapc2p_vel``/``jacobvel`` in the fixture set carry no DG (basis_type/
  poly_order) metadata, so the coordinate-mapping options (``use_c2p_vel``
  etc., which need ``operations.map`` to read that metadata off the mapping file)
  cannot be exercised against these particular files; only the default
  (no-mapping) path is covered here.
  """

  def test_shape_and_grid(self):
    out = distf.load_distf(name=os.path.join(DATA, GK_NAME),
                           species="elc",
                           frame=250,
                           jacobtot_inv_file=os.path.join(
                               DATA, f"{GK_NAME}-geo_int_jacobtot_inv.gkyl"))
    assert out.num_dims == 3
    assert out.num_comps == 1
    assert out.values.shape[:3] == tuple(int(c) for c in out.num_cells)
    assert np.all(np.isfinite(out.values))

  def test_missing_jacobtot_inv_file_raises(self):
    with pytest.raises(Exception):
      distf.load_distf(name=os.path.join(DATA, GK_NAME),
                       species="elc",
                       frame=250,
                       jacobtot_inv_file=os.path.join(DATA,
                                                      "does_not_exist.gkyl"))


class _FakeDistfData(GData):
  """A ``GData`` whose ``.interpolate()`` is a stubbed no-op (real
  interpolation needs the compiled Gkeyll shim), keeping the real computing
  operators (``*``/``/``) so ``load_distf``'s weak-multiply-then-divide
  step still runs (as a plain NumPy op, since these fakes are never
  gkyl-native) -- letting ``load_distf``'s coordinate-map branches
  (``use_c2p_vel``/``use_mc2nu``/``use_mapc2p``) be exercised without real
  mapc2p_vel/mc2nu/mapc2p DG fixtures (the staged rt_gk_tcv_iwl* files carry
  no such metadata -- see TestLoadGkDistfReal)."""

  def interpolate(self,
                  *,
                  basis=None,
                  p=None,
                  num_interp=None,
                  inplace=False,
                  tag=None,
                  label=None):
    return self


class TestLoadGkDistfCoordinateMaps:
  """Unit tests of ``load_distf``'s ``use_c2p_vel``/``use_mc2nu``/
  ``use_mapc2p`` branches, stubbed through ``distf.load``/``operations.map``
  since the compiled-Gkeyll fixtures have no mapping-file metadata to
  exercise them against."""

  def _stub(self, monkeypatch):
    grid = [np.linspace(0.0, 1.0, 5)]
    values = np.ones((4, 1))
    registry = {
        "sim-ion_0.gkyl": (grid, values),
        "sim-ion_jacobvel.gkyl": (grid, values),
        "sim-geo_int_jacobtot_inv.gkyl": (grid, values),
        "sim-ion_mapc2p_vel.gkyl": (grid, values),
    }

    def fake_load(file_name="",
                  *,
                  tag="default",
                  label="",
                  ctx=None,
                  value_form=None,
                  **read_kwargs):
      d = _FakeDistfData(tag=tag, label=label, ctx=ctx)
      if file_name:
        d.push(*registry[file_name])
        d._file_name = file_name
      return d

    monkeypatch.setattr(distf, "load", fake_load)
    calls = []

    def fake_map(data, mapping, *, space, basis_type=None, poly_order=None):
      # mapc2p_vel is pre-loaded (to attach basis_type/poly_order overrides)
      # before it reaches operations.map, so `mapping` arrives as a
      # dataset there, not a filename -- unlike the conf-space maps
      # (mc2nu/mapc2p), which are still passed through as bare paths.
      recorded = mapping._file_name if hasattr(mapping,
                                               "_file_name") else mapping
      calls.append((recorded, space))
      return data

    monkeypatch.setattr(distf.operations, "map", fake_map)
    return calls

  def test_use_c2p_vel(self, monkeypatch):
    calls = self._stub(monkeypatch)
    out = distf.load_distf("sim", "ion", 0, use_c2p_vel=True)
    assert calls == [("sim-ion_mapc2p_vel.gkyl", "vel")]
    assert out.ctx["grid_type"] == "c2p_vel"

  def test_use_mc2nu(self, monkeypatch):
    calls = self._stub(monkeypatch)
    out = distf.load_distf("sim", "ion", 0, use_mc2nu=True)
    assert calls == [("sim-geo_corn_mc2nu_pos_deflated.gkyl", "conf")]
    assert out.ctx["grid_type"] == "mc2nu"

  def test_use_mapc2p(self, monkeypatch):
    calls = self._stub(monkeypatch)
    out = distf.load_distf("sim", "ion", 0, use_mapc2p=True)
    assert calls == [("sim-geo_corn_mapc2p_deflated.gkyl", "conf")]
    assert out.ctx["grid_type"] == "mapc2p"

  def test_use_mc2nu_takes_precedence_over_mapc2p(self, monkeypatch):
    calls = self._stub(monkeypatch)
    out = distf.load_distf("sim", "ion", 0, use_mc2nu=True, use_mapc2p=True)
    assert calls == [("sim-geo_corn_mc2nu_pos_deflated.gkyl", "conf")]
    assert out.ctx["grid_type"] == "mc2nu"

  def test_use_c2p_vel_and_mapc2p_both_applied(self, monkeypatch):
    calls = self._stub(monkeypatch)
    out = distf.load_distf("sim", "ion", 0, use_c2p_vel=True, use_mapc2p=True)
    assert calls == [("sim-ion_mapc2p_vel.gkyl", "vel"),
                     ("sim-geo_corn_mapc2p_deflated.gkyl", "conf")]
    assert out.ctx["grid_type"] == "c2p_vel + mapc2p"

  def test_no_grid_type_key_when_no_maps_requested(self, monkeypatch):
    self._stub(monkeypatch)
    out = distf.load_distf("sim", "ion", 0)
    assert "grid_type" not in out.ctx


class TestFetchCombinators:
  """Unit tests of the generic component-extraction/combinator factories --
  pure field-domain math, no compiled shim needed."""

  def test_component_extraction(self):
    d = _field(np.array([[1.0, 2.0, 3.0]] * 3))
    out = ff._component(d, 1)
    np.testing.assert_allclose(out.values[..., 0], 2.0)

  def test_component_all(self):
    d = _field(np.array([[1.0, 2.0, 3.0]] * 3))
    out = ff._component(d, None)
    assert out.values.shape[-1] == 3

  def test_binop_add(self):
    a = _field(np.array([[1.0, 10.0]] * 2))
    fetch = ff._make_fetch_binop(0, 0, 0, 1, lambda x, y: x + y)
    out = fetch([a])
    np.testing.assert_allclose(out.values[..., 0], 11.0)

  def test_fetch_s1c0_div_s0c0(self):
    m0 = _field(np.full((3, 1), 2.0))
    m1 = _field(np.full((3, 1), 6.0))
    out = ff.fetch_s1c0_div_s0c0([m0, m1])
    np.testing.assert_allclose(out.values, 3.0)


class TestFetchPhysics:
  """Analytic checks of the derived-quantity formulas, using hand-built
  field-domain fixtures (mass/charge as ctx or via the ``**extra`` fallback,
  matching ``_get_ctx_val``'s contract)."""

  def test_M1_from_H(self):
    hmom = _field(np.full((3, 2), 1.0), mass=2.0)
    hmom.values[..., 0] = 4.0
    hmom.values[..., 1] = 3.0
    out = ff.fetch_M1_from_H([hmom])
    np.testing.assert_allclose(out.values[..., 0], 4.0 * 3.0 / 2.0)

  def test_Tpar_from_BiMax(self):
    bimax = _field(np.zeros((2, 4)), mass=3.0)
    bimax.values[..., 2] = 5.0
    out = ff.fetch_Tpar_from_BiMax([bimax])
    np.testing.assert_allclose(out.values[..., 0], 15.0)

  def test_Tpar_from_M0_M1_M2par(self):
    m0 = _field(np.full((2, 1), 2.0), mass=4.0)
    m1 = _field(np.full((2, 1), 6.0))
    m2par = _field(np.full((2, 1), 10.0))
    out = ff.fetch_Tpar_from_M0_M1_M2par([m0, m1, m2par])
    # Tpar = mass*(M2par - M1**2/M0)/M0 = 4*(10 - 36/2)/2 = 4*(-8)/2 = -16
    np.testing.assert_allclose(out.values[..., 0], -16.0)

  def test_temp_from_Tpar_Tperp(self):
    Tpar = _field(np.full((2, 1), 3.0))
    Tperp = _field(np.full((2, 1), 6.0))
    out = ff.fetch_temp_from_Tpar_Tperp([Tpar, Tperp])
    np.testing.assert_allclose(out.values[..., 0], (3.0 + 2 * 6.0) / 3.0)

  def test_press_p(self):
    m0 = _field(np.full((2, 1), 2.0))
    Tp = _field(np.full((2, 1), 5.0))
    out = ff.fetch_press_p([m0, Tp])
    np.testing.assert_allclose(out.values[..., 0], 10.0)

  def test_beta_from_bmag_press(self):
    from scipy import constants
    bmag = _field(np.full((2, 1), 2.0))
    press = _field(np.full((2, 1), 5.0))
    out = ff.fetch_beta_from_bmag_press([bmag, press])
    np.testing.assert_allclose(out.values[..., 0],
                               2.0 * constants.mu_0 * 5.0 / 4.0)

  def test_missing_ctx_key_raises(self):
    m0 = _field(np.full((2, 1), 2.0))
    with pytest.raises(KeyError):
      ff.fetch_M1_from_H([m0])

  def test_missing_ctx_key_uses_extra(self):
    hmom = _field(np.full((2, 2), 1.0))
    hmom.values[..., 0] = 4.0
    hmom.values[..., 1] = 3.0
    out = ff.fetch_M1_from_H([hmom], mass=2.0)
    np.testing.assert_allclose(out.values[..., 0], 6.0)

  def test_Tperp_from_M0_M2perp(self):
    m0 = _field(np.full((2, 1), 2.0), mass=3.0)
    m2perp = _field(np.full((2, 1), 8.0))
    out = ff.fetch_Tperp_from_M0_M2perp([m0, m2perp])
    # Tperp = 0.5*mass*(M2perp/M0) = 0.5*3*(8/2) = 6
    np.testing.assert_allclose(out.values[..., 0], 6.0)

  def test_temp_from_Max(self):
    maxmom = _field(np.zeros((2, 3)), mass=2.0)
    maxmom.values[..., 2] = 5.0
    out = ff.fetch_temp_from_Max([maxmom])
    np.testing.assert_allclose(out.values[..., 0], 10.0)

  def test_press_from_Max(self):
    maxmom = _field(np.zeros((2, 3)), mass=2.0)
    maxmom.values[..., 0] = 3.0
    maxmom.values[..., 2] = 5.0
    out = ff.fetch_press_from_Max([maxmom])
    np.testing.assert_allclose(out.values[..., 0], 2.0 * 3.0 * 5.0)

  def test_press_from_BiMax(self):
    bimax = _field(np.zeros((2, 4)), mass=2.0)
    bimax.values[..., 0] = 3.0  # M0
    bimax.values[..., 2] = 4.0  # Tpar (pre-mass)
    bimax.values[..., 3] = 5.0  # Tperp (pre-mass)
    out = ff.fetch_press_from_BiMax([bimax])
    # press = M0 * mass*(Tpar + 2*Tperp)/3 = 3 * 2*(4 + 10)/3 = 3*28/3 = 28
    np.testing.assert_allclose(out.values[..., 0], 28.0)


class TestGetCtxVal:
  """Resolution of species attributes: an explicit '--extra' override wins
  over the file's own context, which wins over raising; an '--extra' value
  may be a single scalar (every species) or one entry per species."""

  def test_extra_overrides_the_context(self):
    d = _field(np.full((2, 1), 1.0), mass=5.0)
    assert ff._get_ctx_val(d, "mass", mass=999.0) == 999.0

  def test_context_is_used_when_extra_does_not_carry_the_key(self):
    d = _field(np.full((2, 1), 1.0), mass=5.0)
    assert ff._get_ctx_val(d, "mass") == 5.0
    assert ff._get_ctx_val(d, "mass", charge=1.0) == 5.0

  def test_scalar_extra_applies_to_every_species(self):
    d = _field(np.full((2, 1), 1.0))
    for species_idx in range(3):
      assert ff._get_ctx_val(d, "mass", mass=7.0,
                             species_idx=species_idx) == 7.0

  def test_per_species_array_is_picked_by_species_index(self):
    d = _field(np.full((2, 1), 1.0))
    for species_idx, expected in enumerate([1.0, 2.0, 3.0]):
      got = ff._get_ctx_val(d,
                            "mass",
                            mass=[1.0, 2.0, 3.0],
                            species_idx=species_idx)
      assert got == expected

  def test_array_without_a_species_index_is_an_error(self):
    d = _field(np.full((2, 1), 1.0))
    with pytest.raises(KeyError, match="not resolved per species"):
      ff._get_ctx_val(d, "mass", mass=[1.0, 2.0])

  def test_too_short_an_array_is_an_error(self):
    d = _field(np.full((2, 1), 1.0))
    with pytest.raises(ValueError, match="only 2 values"):
      ff._get_ctx_val(d, "mass", mass=[1.0, 2.0], species_idx=2, species="ion2")

  def test_missing_everywhere_is_an_error(self):
    d = _field(np.full((2, 1), 1.0))
    with pytest.raises(KeyError, match="mass"):
      ff._get_ctx_val(d, "mass")


class TestHeatFluxes:
  """Lab-frame energy fluxes and fluid-frame heat fluxes."""

  def test_qpar_lab_frame(self):
    m3par = _field(np.full((2, 1), 4.0), mass=2.0)
    out = ff.fetch_qpar([m3par])
    np.testing.assert_allclose(out.values[..., 0], 0.5 * 2.0 * 4.0)

  def test_qperp_lab_frame(self):
    m3perp = _field(np.full((2, 1), 6.0), mass=3.0)
    out = ff.fetch_qperp([m3perp])
    np.testing.assert_allclose(out.values[..., 0], 0.5 * 3.0 * 6.0)

  def test_qpar_fluid_matches_the_hand_derived_formula(self):
    m0 = _field(np.full((2, 1), 2.0), mass=5.0)
    m1 = _field(np.full((2, 1), 6.0))
    m2par = _field(np.full((2, 1), 10.0))
    m3par = _field(np.full((2, 1), 40.0))
    out = ff.fetch_qpar_fluid([m0, m1, m2par, m3par])
    upar = 6.0 / 2.0
    expected = 0.5 * 5.0 * (40.0 - 3.0 * upar * 10.0 + 2.0 * upar**2 * 6.0)
    np.testing.assert_allclose(out.values[..., 0], expected)

  def test_qperp_fluid_matches_the_hand_derived_formula(self):
    m0 = _field(np.full((2, 1), 2.0), mass=5.0)
    m1 = _field(np.full((2, 1), 6.0))
    m2perp = _field(np.full((2, 1), 8.0))
    m3perp = _field(np.full((2, 1), 20.0))
    out = ff.fetch_qperp_fluid([m0, m1, m2perp, m3perp])
    upar = 6.0 / 2.0
    expected = 0.5 * 5.0 * (20.0 - upar * 8.0)
    np.testing.assert_allclose(out.values[..., 0], expected)

  def test_qpar_fluid_vanishes_for_a_maxwellian(self):
    """A Maxwellian carries no parallel heat flux in the fluid frame: the
    three terms of ``(m/2)*[M3par - 3*u*M2par + 2*u^2*M1]`` cancel exactly
    for ``M1=n*u``, ``M2par=n*(u^2+T/m)``, ``M3par=n*(u^3+3*u*T/m)``, so the
    residual is compared against the size of an individual term rather
    than an absolute zero."""
    n, u, T, m = 2.7e19, 1.3e4, 9.5e-18, 3.343e-27
    vt_sq = T / m
    m0 = _field(np.full((2, 1), n), mass=m)
    m1 = _field(np.full((2, 1), n * u))
    m2par = _field(np.full((2, 1), n * (u**2 + vt_sq)))
    m3par = _field(np.full((2, 1), n * (u**3 + 3.0 * u * vt_sq)))
    out = ff.fetch_qpar_fluid([m0, m1, m2par, m3par])
    term_scale = 0.5 * m * n * abs(u)**3
    assert np.all(np.abs(out.values[..., 0]) / term_scale < 1e-9)


class TestThermalSpeedAndLengths:
  """``vt``/Larmor-radius/Debye-length -- plain ``np.sqrt`` on interpolated
  data (this layer never touches ``dg``/``gpython``; see the module
  docstring in ``quantities.py``)."""

  def test_vt(self):
    temp = _field(np.full((2, 1), 8.0), mass=2.0)
    out = ff.fetch_vt([temp])
    np.testing.assert_allclose(out.values[..., 0], np.sqrt(4.0))

  def test_larmor_radius(self):
    temp = _field(np.full((2, 1), 4.0), mass=9.0, charge=-2.0)
    bmag = _field(np.full((2, 1), 3.0))
    out = ff.fetch_larmor_radius([temp, bmag])
    expected = np.sqrt(9.0 * 4.0) / (2.0 * 3.0)
    np.testing.assert_allclose(out.values[..., 0], expected)

  def test_debye_length(self):
    from scipy import constants
    temp = _field(np.full((2, 1), 5.0), charge=2.0)
    m0 = _field(np.full((2, 1), 7.0))
    out = ff.fetch_debye_length([temp, m0])
    expected = np.sqrt(constants.epsilon_0 * 5.0 / (7.0 * 4.0))
    np.testing.assert_allclose(out.values[..., 0], expected)


class TestSoundSpeed:
  """The multi-species sound speeds, dispatched by '--extra kind='.

  ``fetch_c_s`` is an ``is_multi_species`` fetch function: it receives one
  ``[M0, temp]`` source list per species (as
  ``GkQuantity.fetch_multi``/``load_quantity`` would hand it), not a
  flat list.
  """

  @staticmethod
  def _species(dens, temp, mass, charge):
    return [
        _field(np.full((2, 1), dens), mass=mass, charge=charge),
        _field(np.full((2, 1), temp), mass=mass, charge=charge)
    ]

  def test_ion_acoustic_single_ion_species(self):
    """With one Z=1 ion species the formula collapses to sqrt(Te/mi)."""
    from scipy import constants
    e = constants.elementary_charge
    m_e, T_e = constants.electron_mass, 9.5e-18
    n_i, T_i, m_i, z_i = 2.7e19, 6.1e-18, 3.343e-27, 1.0
    n_e = n_i * z_i

    out = ff.fetch_c_s([
        self._species(n_e, T_e, m_e, -e),
        self._species(n_i, T_i, m_i, z_i * e)
    ],
                       species=["elc", "ion"],
                       kind="ion_acoustic")
    np.testing.assert_allclose(out.values[..., 0],
                               np.sqrt(T_e / m_i),
                               rtol=1e-10)

  def test_thermo_defaults_are_gamma_e_1_gamma_i_3(self):
    from scipy import constants
    e = constants.elementary_charge
    m_e, T_e = constants.electron_mass, 9.5e-18
    n_i, T_i, m_i, z_i = 2.7e19, 6.1e-18, 3.343e-27, 1.0
    n_e = n_i * z_i

    gdatas = [
        self._species(n_e, T_e, m_e, -e),
        self._species(n_i, T_i, m_i, z_i * e)
    ]
    default = ff.fetch_c_s(gdatas, species=["elc", "ion"], kind="thermo")
    explicit = ff.fetch_c_s(gdatas,
                            species=["elc", "ion"],
                            kind="thermo",
                            gamma_e=1.0,
                            gamma_i=3.0)
    np.testing.assert_allclose(default.values, explicit.values, rtol=1e-12)

    expected = np.sqrt((1.0 * n_e * T_e + 3.0 * n_i * T_i) / (n_i * m_i))
    np.testing.assert_allclose(default.values[..., 0], expected, rtol=1e-10)

  def test_species_order_does_not_matter(self):
    """Species are identified by charge sign, so the order is irrelevant."""
    elc = self._species(1.0e19, 5.0e-18, 9.1e-31, -1.6e-19)
    ion = self._species(1.0e19, 3.0e-18, 3.3e-27, 1.6e-19)
    forward = ff.fetch_c_s([elc, ion],
                           species=["elc", "ion"],
                           kind="ion_acoustic")
    shuffled = ff.fetch_c_s([ion, elc],
                            species=["ion", "elc"],
                            kind="ion_acoustic")
    np.testing.assert_allclose(forward.values, shuffled.values, rtol=1e-12)

  def test_no_electron_species_is_an_error(self):
    ion = self._species(1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="exactly one negatively charged"):
      ff.fetch_c_s([ion], species=["ion"])

  def test_no_ion_species_is_an_error(self):
    elc = self._species(1.0, 1.0, 1.0, -1.0)
    with pytest.raises(ValueError, match="no positively charged"):
      ff.fetch_c_s([elc], species=["elc"])

  def test_unknown_kind_is_an_error(self):
    elc = self._species(1.0, 1.0, 1.0, -1.0)
    ion = self._species(1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="unknown kind"):
      ff.fetch_c_s([elc, ion], species=["elc", "ion"], kind="bogus")

  def test_per_species_extra_arrays_reach_nested_sources(self):
    """'--extra mass=1,2,charge=-1,1' must give each species its own entry,
    threaded through even though these sources carry no mass/charge in
    their own ctx -- the whole point of ``species_idx`` reaching
    ``get_src_gdata``/``_split_elc_ions``."""
    bare = lambda dens, temp: [
        _field(np.full((2, 1), dens)),
        _field(np.full((2, 1), temp))
    ]
    out = ff.fetch_c_s([bare(2.0, 8.0), bare(2.0, 8.0)],
                       species=["elc", "ion"],
                       kind="thermo",
                       mass=[1.0, 2.0],
                       charge=[-1.0, 1.0])
    # n=2, T=8 for both species; gamma_e=1, gamma_i=3 (defaults).
    expected = np.sqrt((1.0 * 2.0 * 8.0 + 3.0 * 2.0 * 8.0) / (2.0 * 2.0))
    np.testing.assert_allclose(out.values[..., 0], expected, rtol=1e-10)


class TestNormalizedQuantities:

  def test_rho_over_lambda(self):
    rho = _field(np.full((2, 1), 6.0))
    lambda_d = _field(np.full((2, 1), 3.0))
    out = ff.fetch_rho_over_lambda([rho, lambda_d])
    np.testing.assert_allclose(out.values[..., 0], 2.0)

  def test_phi_norm(self):
    from scipy import constants
    phi = _field(np.full((2, 1), 5.0))
    temp = _field(np.full((2, 1), 2.0))
    out = ff.fetch_phi_norm([phi, temp])
    np.testing.assert_allclose(out.values[..., 0],
                               constants.elementary_charge * 5.0 / 2.0)

  def test_qpar_norm(self):
    q = _field(np.full((2, 1), 12.0))
    m0 = _field(np.full((2, 1), 2.0))
    temp = _field(np.full((2, 1), 3.0))
    c_s = _field(np.full((2, 1), 2.0))
    out = ff.fetch_qpar_norm([q, m0, temp, c_s])
    np.testing.assert_allclose(out.values[..., 0], 12.0 / (2.0 * 3.0 * 2.0))


class TestDriftVelocities:
  """``fetch_gradB_vel``/``fetch_diamag_vel`` and the remaining
  ``_b_cross_grad_div_b_component`` branches (comp 1/2, cdim 1/2/3)."""

  def _synthetic(self, cdim, comp):
    grid = [np.linspace(0.0, float(n), n + 1) for n in [4, 4, 4][:cdim]]
    centers = [0.5 * (g[:-1] + g[1:]) for g in grid]
    mesh = np.meshgrid(*centers, indexing="ij")
    scalar = _field(sum(mesh)[..., np.newaxis], grid=grid)
    jacobtot_inv = _field(np.full(scalar.values.shape, 2.0), grid=grid)
    b_i = _field(np.stack([np.full(mesh[0].shape, float(k)) for k in range(3)],
                          axis=-1),
                 grid=grid)
    return scalar, jacobtot_inv, b_i

  @pytest.mark.parametrize("cdim,comp", [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1),
                                         (2, 2), (3, 0), (3, 1), (3, 2)])
  def test_all_cdim_comp_combinations_run(self, cdim, comp):
    scalar, jacobtot_inv, b_i = self._synthetic(cdim, comp)
    out = ff._b_cross_grad_div_b_component(scalar, jacobtot_inv, b_i, comp)
    assert out.values.shape == scalar.values.shape
    assert np.all(np.isfinite(out.values))

  def test_gradB_vel(self):
    scalar, jacobtot_inv, b_i = self._synthetic(1, 0)
    Tperp = _field(np.full(scalar.values.shape, 3.0),
                   grid=scalar.grid,
                   charge=2.0)
    out = ff.fetch_gradB_vel([jacobtot_inv, scalar, b_i, Tperp], dir=0)
    assert np.all(np.isfinite(out.values))

  def test_diamag_vel(self):
    scalar, jacobtot_inv, b_i = self._synthetic(1, 0)
    m0 = _field(np.full(scalar.values.shape, 5.0), grid=scalar.grid)
    pressperp = _field(np.full(scalar.values.shape, 3.0),
                       grid=scalar.grid,
                       charge=2.0)
    out = ff.fetch_diamag_vel([jacobtot_inv, scalar, b_i, m0, pressperp], dir=0)
    assert np.all(np.isfinite(out.values))

  def test_gradB_vel_requires_dir(self):
    with pytest.raises(KeyError):
      ff.fetch_gradB_vel([None, None, None, None])

  def test_diamag_vel_requires_dir(self):
    with pytest.raises(KeyError):
      ff.fetch_diamag_vel([None, None, None, None, None])


class TestLoadDistf:
  """``fetch_funcs.load_distf`` -- the registry 'distf' quantity's fetch
  function -- stubbed against ``load_distf`` so this checks the option
  translation (``dict_get_bool``, path/name joining) without needing a real
  distribution-function file set (covered end to end by
  ``TestLoadGkDistfReal`` instead)."""

  def test_forwards_options(self, monkeypatch):
    calls = {}

    def fake_load_distf(**kwargs):
      calls.update(kwargs)
      return "sentinel"

    from postgkyl.diagnostics.gk import distf as distf_mod
    monkeypatch.setattr(distf_mod, "load_distf", fake_load_distf)

    out = ff.load_distf([],
                        path="/some/path/",
                        name="sim",
                        species="ion",
                        frame="3",
                        suffix="src",
                        c2p_vel="0",
                        mc2nu="1",
                        block=2)
    assert out == "sentinel"
    assert calls["name"] == "/some/path/sim"
    assert calls["species"] == "ion"
    assert calls["frame"] == 3
    assert calls["suffix"] == "src"
    assert calls["use_c2p_vel"] is False
    assert calls["use_mc2nu"] is True
    assert calls["use_mapc2p"] is False
    assert calls["block_idx"] == 2
    assert calls["num_interp"] == 0

  def test_defaults(self, monkeypatch):
    calls = {}

    def fake_load_distf(**kwargs):
      calls.update(kwargs)
      return "sentinel"

    from postgkyl.diagnostics.gk import distf as distf_mod
    monkeypatch.setattr(distf_mod, "load_distf", fake_load_distf)

    ff.load_distf([], path="p", name="n", species="ion", frame=0)
    # c2p_vel defaults True when not given as an extra.
    assert calls["use_c2p_vel"] is True


class TestCrossGradDivB:
  """``_b_cross_grad_div_b_component`` on a 1-D synthetic field (cdim=1):
  only the 'positive' term is defined, so the formula reduces to
  ``d(f)/dx * b_i[bi_c_pos] * jacobtot_inv``."""

  def test_linear_scalar_1d(self):
    x = np.linspace(0.0, 4.0, 5)  # 4 cells, dx=1
    centers = 0.5 * (x[:-1] + x[1:])  # phi(x) = x at cell centers
    phi = _field(centers[:, np.newaxis], grid=[x])
    jacobtot_inv = _field(np.full((4, 1), 2.0), grid=[x])
    b_i = _field(np.tile([0.0, 1.0, 0.0], (4, 1)), grid=[x])
    out = ff._b_cross_grad_div_b_component(phi, jacobtot_inv, b_i, 0)
    np.testing.assert_allclose(out.values[..., 0], 2.0, rtol=1e-6)

  def test_invalid_component_raises(self):
    x = np.linspace(0.0, 1.0, 3)
    phi = _field(np.zeros((2, 1)), grid=[x])
    jacobtot_inv = _field(np.ones((2, 1)), grid=[x])
    b_i = _field(np.zeros((2, 3)), grid=[x])
    with pytest.raises(KeyError):
      ff._b_cross_grad_div_b_component(phi, jacobtot_inv, b_i, 3)

  def test_ExB_vel_requires_dir(self):
    with pytest.raises(KeyError):
      ff.fetch_ExB_vel([None, None, None, None])


class TestLoadQuantity:

  def test_available_quantities_sorted(self):
    names = available_quantities()
    assert names == sorted(names)
    assert "M0" in names
    assert "distf" in names

  def test_unknown_quantity_raises(self):
    with pytest.raises(ValueError, match="Unknown quantity"):
      load_quantity("not_a_quantity", None, "sim", path=DATA)

  @needs_gkeyll
  def test_M0_from_hamiltonian_moments_real(self):
    out = load_quantity("M0", "ion", HMOM_NAME, "250", path=DATA)
    assert len(out) == 1
    assert out[0].get_label() == r"$M_{0i}$ (m$^{-3}$)"
    assert out[0].values.shape[-1] == 1

  @needs_gkeyll
  def test_M1_from_hamiltonian_moments_real(self):
    out = load_quantity("M1", "ion", HMOM_NAME, "250", path=DATA, mass=2.0)
    assert len(out) == 1
    assert np.all(np.isfinite(out[0].values))

  @needs_gkeyll
  def test_geo_quantity_real(self):
    out = load_quantity("geo_int_jacobtot_inv", None, GK_NAME, path=DATA)
    assert len(out) == 1
    assert out[0].get_label() == r"$(J B)^{-1}$"

  @needs_gkeyll
  def test_geo_quantity_missing_file_raises(self):
    with pytest.raises(FileNotFoundError):
      load_quantity("geo_int_bmag", None, GK_NAME, path=DATA)

  def test_label_and_tag_override(self, tmp_path, monkeypatch):
    # A species-independent geo quantity needs only its own marker file.
    (tmp_path / f"sim-geo_int_bmag.gkyl").touch()
    monkeypatch.setattr(qmod, "GData",
                        lambda *a, **k: _field(np.full((2, 1), 3.0)))
    out = load_quantity("geo_int_bmag",
                        None,
                        "sim",
                        path=str(tmp_path),
                        tag="mytag",
                        label="custom")
    assert out[0].get_tag() == "mytag"
    assert out[0].get_label() == "custom"


class _SyntheticSource:
  """Serves a small, self-consistent constant-valued synthetic DG dataset
  for every source file a quantity asks for -- ported from
  tests_bak/test_gk_load_quantity.py's ``_make_synthetic_gdata``, adapted to
  push through the new ``GDataState``/``.interpolate()`` (no ``ctypes``).

  Every source is served the same synthetic values; only the charge is read
  back out of the file name (negative for an ``elc`` species, per
  ``_ELC_SPECIES``), so multi-species quantities -- which tell electrons
  from ions by the sign of the charge -- see a genuine electron species.
  """

  POLY_ORDER = 1
  BASIS_TYPE = "serendipity"
  NUM_BASIS = 2
  NUM_PHYS_COMPS = 4
  NUM_CELLS = 4

  def __call__(self, *args, **kwargs):
    values = np.zeros((self.NUM_CELLS, self.NUM_BASIS * self.NUM_PHYS_COMPS))
    for comp in range(self.NUM_PHYS_COMPS):
      values[:, comp * self.NUM_BASIS] = (comp + 2) * np.sqrt(2.0)
    file_name = str(args[0]) if args else ""
    charge = -1.0 if f"-{_ELC_SPECIES}_" in file_name else 1.0
    grid = [np.linspace(0.0, 1.0, self.NUM_CELLS + 1)]
    d = GDataState(
        ctx={
            "poly_order": self.POLY_ORDER,
            "basis_type": self.BASIS_TYPE,
            "mass": 1.0,
            "charge": charge
        })
    d.push(grid, values)
    return d


# Species names used to drive a genuine electron/ion split in the synthetic
# smoke tests: multi-species quantities (e.g. the sound speed) tell them
# apart by the sign of the charge, which _SyntheticSource keys off this name.
_ELC_SPECIES = "elc"
_ION_SPECIES = "ion"


def _collect_source_files(quant, path, name, species, frame) -> set:
  files: set[str] = set()
  for combo in quant.source:
    for src in combo:
      if isinstance(src, str):
        files.add(quant._src_file_name(path, name, species, src, frame))
      else:
        files |= _collect_source_files(src, path, name, species, frame)
  return files


def _extra_for(quant) -> dict:
  extra = {}
  if quant.is_vector:
    extra["direction"] = 0
  return extra


@needs_gkeyll
@pytest.mark.parametrize("quantity", gk_quant_registry.list())
def test_every_registered_quantity_produces_a_dataset(quantity, tmp_path,
                                                      monkeypatch):
  """Smoke test across the whole registry (weak assertion, matching
  tests_bak/test_gk_load_quantity.py): the synthetic data is not physically
  consistent across different marker files (every file gets the SAME
  constant recipe, regardless of what real quantity it names), so this
  checks "no exception, one dataset comes back", not specific numbers --
  those are covered analytically in ``TestFetchPhysics`` above."""
  if quantity == "distf":
    pytest.skip("distf delegates to load_distf, covered by "
                "TestLoadGkDistfReal against the real staged fixtures")

  quant = gk_quant_registry.get(quantity)
  name, frame = "gktest", 0
  path = str(tmp_path)
  # A multi-species quantity (e.g. the sound speed) needs an electron and
  # an ion species to combine; every other quantity is fine with just one.
  species = (f"{_ELC_SPECIES},{_ION_SPECIES}"
             if quant.is_multi_species else _ION_SPECIES)

  for species_name in species.split(","):
    for file_name in _collect_source_files(quant, path, name, species_name,
                                           frame):
      open(file_name, "w").close()

  monkeypatch.setattr(qmod, "GData", _SyntheticSource())

  out = load_quantity(quantity,
                      species,
                      name,
                      str(frame),
                      path=path,
                      **_extra_for(quant))
  assert len(out) >= 1
  assert isinstance(out[0], GDataState)


class TestGkQuantityGetAvailSource:
  """``GkQuantity.get_avail_source``/``_avail_combo_frames`` frame-list
  parsing branches, exercised directly (rather than through the full
  registry) for precise control over which frames each source combo has."""

  def _touch_frames(self, tmp_path, stem, frames):
    for f in frames:
      (tmp_path / f"{stem}{f}.gkyl").touch()

  def test_comma_separated_frame_list(self, tmp_path):
    quant = qmod.GkQuantity(name="q",
                            source=[["a"]],
                            fetch_func=[None],
                            label="q",
                            is_species_dep=True)
    self._touch_frames(tmp_path, "sim-ion_a_", [0, 2, 4])
    combo_idx, frames = quant.get_avail_source(str(tmp_path), "sim", "ion",
                                               "0,2")
    assert combo_idx == 0
    assert frames == [0, 2]

  def test_none_frame_selects_every_available(self, tmp_path):
    quant = qmod.GkQuantity(name="q",
                            source=[["a"]],
                            fetch_func=[None],
                            label="q",
                            is_species_dep=True)
    self._touch_frames(tmp_path, "sim-ion_a_", [0, 1, 3])
    combo_idx, frames = quant.get_avail_source(str(tmp_path), "sim", "ion",
                                               None)
    assert frames == [0, 1, 3]

  def test_partial_range_frame(self, tmp_path):
    quant = qmod.GkQuantity(name="q",
                            source=[["a"]],
                            fetch_func=[None],
                            label="q",
                            is_species_dep=True)
    self._touch_frames(tmp_path, "sim-ion_a_", [0, 1, 2, 3])
    combo_idx, frames = quant.get_avail_source(str(tmp_path), "sim", "ion",
                                               "1:")
    assert frames == [1, 2, 3]

  def test_mismatched_frame_sets_falls_back_to_next_combo(self, tmp_path):
    # combo 0 ("a","b") has mismatched frame sets -> rejected; combo 1 ("c")
    # is used instead.
    quant = qmod.GkQuantity(name="q",
                            source=[["a", "b"], ["c"]],
                            fetch_func=[None, None],
                            label="q",
                            is_species_dep=True)
    self._touch_frames(tmp_path, "sim-ion_a_", [0, 1])
    self._touch_frames(tmp_path, "sim-ion_b_", [0])
    self._touch_frames(tmp_path, "sim-ion_c_", [5])
    combo_idx, frames = quant.get_avail_source(str(tmp_path), "sim", "ion",
                                               None)
    assert combo_idx == 1
    assert frames == [5]

  def test_no_files_found_raises(self, tmp_path):
    quant = qmod.GkQuantity(name="q",
                            source=[["a"]],
                            fetch_func=[None],
                            label="q",
                            is_species_dep=True)
    with pytest.raises(FileNotFoundError):
      quant.get_avail_source(str(tmp_path), "sim", "ion", None)


@needs_gkeyll
class TestLoadQuantityMultiSpeciesMultiFrame:
  """Exercises ``load_quantity``'s multi-species/multi-frame label/tag
  suffix branches (only reached when more than one species or frame is
  requested)."""

  def test_multiple_species(self, tmp_path, monkeypatch):
    quant = gk_quant_registry.get("M0")
    name = "gktest"
    path = str(tmp_path)
    for species in ("ion", "elc"):
      for file_name in _collect_source_files(quant, path, name, species, 0):
        open(file_name, "w").close()
    monkeypatch.setattr(qmod, "GData", _SyntheticSource())

    out = load_quantity("M0",
                        "ion,elc",
                        name,
                        "0",
                        path=path,
                        tag="t",
                        label="custom")
    assert len(out) == 2
    assert {d.get_tag() for d in out} == {"t_ion", "t_elc"}
    assert {d.get_label() for d in out} == {"custom ion", "custom elc"}

  def test_multiple_frames_suffixes_label(self, tmp_path, monkeypatch):
    quant = gk_quant_registry.get("M0")
    name = "gktest"
    path = str(tmp_path)
    for frame in (0, 1, 2):
      for file_name in _collect_source_files(quant, path, name, "ion", frame):
        open(file_name, "w").close()
    monkeypatch.setattr(qmod, "GData", _SyntheticSource())

    out = load_quantity("M0", "ion", name, None, path=path)
    assert len(out) == 3
    assert all(" f" in d.get_label() for d in out)

  def test_multi_species_quantity_yields_a_single_dataset(
      self, tmp_path, monkeypatch):
    """A multi-species quantity (the sound speed) combines its species into
    one dataset, unlike a per-species quantity (M0), which still produces
    one dataset per species."""
    name = "gktest"
    path = str(tmp_path)
    for species in (_ELC_SPECIES, _ION_SPECIES):
      for file_name in _collect_source_files(gk_quant_registry.get("c_s"), path,
                                             name, species, 0):
        open(file_name, "w").close()
      for file_name in _collect_source_files(gk_quant_registry.get("M0"), path,
                                             name, species, 0):
        open(file_name, "w").close()
    monkeypatch.setattr(qmod, "GData", _SyntheticSource())

    species = f"{_ELC_SPECIES},{_ION_SPECIES}"
    out = load_quantity("c_s", species, name, "0", path=path)
    assert len(out) == 1

    out = load_quantity("M0", species, name, "0", path=path)
    assert len(out) == 2

  def test_multi_species_quantity_needs_a_species_list(self, tmp_path):
    with pytest.raises(ValueError, match="needs a species list"):
      load_quantity("c_s", None, "gktest", "0", path=str(tmp_path))


class TestUtils:
  """postgkyl.diagnostics.gk.utils -- file/geometry helpers ported
  from src_bak's gk_utils.py (matplotlib bits dropped, read_g*file adapted
  to postgkyl.gdata.load + .interpolate())."""

  def test_dict_get_bool_default(self):
    assert utils.dict_get_bool({}, "k", True) is True
    assert utils.dict_get_bool({}, "k", False) is False

  def test_dict_get_bool_string_true_variants(self):
    assert utils.dict_get_bool({"k": "1"}, "k", False) is True
    assert utils.dict_get_bool({"k": "True"}, "k", False) is True
    assert utils.dict_get_bool({"k": " true "}, "k", False) is True

  def test_dict_get_bool_string_false(self):
    assert utils.dict_get_bool({"k": "0"}, "k", True) is False
    assert utils.dict_get_bool({"k": "no"}, "k", True) is False

  def test_dict_get_bool_non_string(self):
    assert utils.dict_get_bool({"k": 1}, "k", False) is True
    assert utils.dict_get_bool({"k": 0}, "k", True) is False

  def test_parse_slice_string(self):
    assert utils.parse_slice_string("1:5") == slice(1, 5)
    assert utils.parse_slice_string(":5") == slice(None, 5)
    assert utils.parse_slice_string("1:") == slice(1, None)
    assert utils.parse_slice_string("1:5:2") == slice(1, 5, 2)

  def test_parse_slice_string_invalid_raises(self):
    with pytest.raises(ValueError):
      utils.parse_slice_string("a:5")

  def test_get_block_indices_single(self):
    assert utils.get_block_indices("-10", "unused") == [0]

  def test_get_block_indices_all(self, tmp_path):
    for i in range(3):
      (tmp_path / f"sim_b{i}-ion_field_0.gkyl").touch()
    pattern = str(tmp_path / "sim_b*-ion_field_0.gkyl")
    assert utils.get_block_indices("-1", pattern) == [0, 1, 2]

  def test_get_block_indices_comma_list(self):
    assert utils.get_block_indices("0,2,4", "unused") == [0, 2, 4]

  def test_get_block_indices_slice(self):
    assert utils.get_block_indices("1:4", "unused") == [1, 2, 3]

  def test_get_block_indices_single_int(self):
    assert utils.get_block_indices("2", "unused") == [2]

  def test_get_block_indices_invalid_raises(self):
    with pytest.raises(NameError):
      utils.get_block_indices("not-a-spec", "unused")

  @needs_gkeyll
  def test_read_gfile(self):
    grid, values, gdata = utils.read_gfile(
        os.path.join(DATA, f"{GK_NAME}-geo_int_jacobtot_inv.gkyl"))
    assert values.shape[0] == gdata.num_cells[0]

  def test_read_gfile_if_present_missing(self, tmp_path):
    found, grid, values, gdata = utils.read_gfile_if_present(
        str(tmp_path / "does_not_exist.gkyl"))
    assert found is False
    assert grid is None and values is None and gdata is None

  @needs_gkeyll
  def test_read_gfile_if_present_found(self):
    found, grid, values, gdata = utils.read_gfile_if_present(
        os.path.join(DATA, f"{GK_NAME}-geo_int_jacobtot_inv.gkyl"))
    assert found is True
    assert values is not None

  @needs_gkeyll
  def test_read_interpolated_gfile(self):
    grid, values, gdata = utils.read_interpolated_gfile(
        os.path.join(DATA, f"{GK_NAME}-geo_int_jacobtot_inv.gkyl"),
        poly_order=1,
        basis_type="serendipity")
    assert gdata.is_interpolated

  @needs_gkeyll
  def test_read_interpolated_gfile_with_comp(self):
    grid, values, gdata = utils.read_interpolated_gfile(
        os.path.join(DATA, f"{GK_NAME}-geo_int_jacobtot_inv.gkyl"),
        poly_order=1,
        basis_type="serendipity",
        comp=0)
    assert gdata.num_comps == 1
