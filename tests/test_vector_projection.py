"""Analytic vector projections on nonorthogonal 2x/3x affine geometry."""
from pathlib import Path
import shutil

import numpy as np
import pytest
from click.testing import CliRunner

import postgkyl as pg
from postgkyl import gpython, operations
from postgkyl.cli.app import cli
from postgkyl.diagnostics.gk.load_quantity import load_quantity
from postgkyl.diagnostics.gk.registry import gk_quant_registry
from generate_test_data import drift_projection_reference

pytestmark = pytest.mark.skipif(not gpython.available(),
                                reason="requires Gkeyll")
DATA = Path(__file__).parent / "test_data" / "generated"
DIRECTIONS = ("n0", "n1", "n2", "R", "Z", "phi")


def _prefix(ndim, kind="cart"):
  return f"drift_{kind}" if ndim == 3 else f"drift_2x_{kind}"


def _read(prefix, suffix):
  return pg.load(str(DATA / f"{prefix}-{suffix}.gkyl"))


def _expected(vector, direction, ndim):
  """Independent physical dot product and explicit tensor Gauss projection."""
  cells, _, _, _, basis, _, normals, positions, _, _, _ = drift_projection_reference(
      ndim)
  if direction.startswith("n"):
    unit = normals[int(direction[1])]
  elif direction == "Z":
    unit = np.array([0., 0., 1.])
  else:
    radius = np.hypot(positions[..., 0], positions[..., 1])
    cosine, sine = positions[..., 0] / radius, positions[..., 1] / radius
    unit = np.stack((cosine, sine, np.zeros_like(cosine)), axis=-1)
    if direction == "phi":
      unit = np.stack((-sine, cosine, np.zeros_like(cosine)), axis=-1)
  vector = np.broadcast_to(vector, (*cells, 2**ndim, 3))
  return np.einsum("...q,qm->...m", np.sum(vector * unit, axis=-1),
                   basis).reshape(*cells, 2**ndim)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("kind", ["cart", "cyl"])
@pytest.mark.parametrize("basis", ["contravariant", "covariant", "cartesian"])
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_general_vector_projection(ndim, kind, basis, direction):
  """All representations recover a known physical vector, coefficient by coefficient."""
  prefix = _prefix(ndim, kind)
  vector = _read(prefix, "vector_" + basis)
  points = drift_projection_reference(ndim)[3]
  scalar = points[..., 0] + (.3 * points[..., 2] if ndim == 2 else 0.)
  physical = np.stack(
      (1 + .2 * points[..., 0], -.7 + .1 * points[..., 2], .4 + .05 * scalar),
      axis=-1)
  result = operations.project_vector(
      [vector.select(comp=i) for i in range(3)],
      basis=basis,
      direction=direction,
      tangents=_read(prefix, "geo_int_dxdz"),
      duals=_read(prefix, "geo_int_dzdx"),
      normals=_read(prefix, "geo_int_normals"),
      positions=_read(prefix, "geo_int_mapc2p"),
      position_system="cartesian" if kind == "cart" else "cylindrical")
  np.testing.assert_allclose(result.values,
                             _expected(physical, direction, ndim),
                             rtol=2e-12,
                             atol=2e-13)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("form", ["nodal", "quad", "interpolated"])
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_point_representations(ndim, form, direction):
  """Point-value inputs stay in their representation and use pointwise geometry."""
  prefix = _prefix(ndim)

  def read(suffix):
    field = _read(prefix, suffix)
    return field.interpolate() if form == "interpolated" else field.represent(
        to=form)

  vector, positions = read("vector_cartesian"), read("geo_int_mapc2p")
  components = [vector.select(comp=i) for i in range(3)]
  result = operations.project_vector(components,
                                     basis="cartesian",
                                     direction=direction,
                                     positions=positions,
                                     position_system="cartesian",
                                     normals=read("geo_int_normals"))
  xyz = [positions.select(comp=i).values for i in range(3)]
  if direction.startswith("n"):
    unit = drift_projection_reference(ndim)[6][int(direction[1])]
  elif direction == "Z":
    unit = (0., 0., 1.)
  else:
    radius = np.hypot(xyz[0], xyz[1])
    unit = (xyz[0] / radius, xyz[1] / radius,
            0.) if direction == "R" else (-xyz[1] / radius, xyz[0] / radius, 0.)
  expected = sum(component.values * unit[i]
                 for i, component in enumerate(components))
  np.testing.assert_allclose(result.values, expected, rtol=2e-13, atol=2e-13)
  assert result.backend == vector.backend
  assert result.ctx.get("value_form") == vector.ctx.get("value_form")
  assert result is not components[2]


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("kind", ["cart", "cyl"])
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("quantity", [
    "E_field_cov", "E_field_contra", "dB_perp_cov", "dB_perp_contra",
    "B_tot_cov", "B_tot_contra"
])
def test_field_diagnostics(ndim, kind, direction, quantity):
  """E=-grad(phi), dB=grad(Apar) cross b, and Btot=B*b+dB in physical units."""
  _, _, _, _, _, tangents, _, _, bmag, _, _ = drift_projection_reference(ndim)
  duals = np.linalg.inv(tangents)
  gradient = duals[0] + (.3 * duals[2] if ndim == 2 else 0.)
  if quantity.startswith("E_field"):
    physical = -.4 * gradient
    units = "V/m"
  else:
    physical = .6 * np.cross(gradient, [0., 0., 1.])
    if quantity.startswith("B_tot"):
      physical = physical + bmag[..., None] * np.array([0., 0., 1.])
    units = "T"
  result, = load_quantity(quantity,
                          None,
                          _prefix(ndim, kind),
                          "0",
                          path=str(DATA),
                          direction=direction)
  np.testing.assert_allclose(result.values,
                             _expected(physical, direction, ndim),
                             rtol=2e-12,
                             atol=2e-13)
  assert result.ctx["projection_units"] == units
  assert f"({units})" in result.get_label()


@pytest.mark.parametrize("quantity", ["B_tot_cov", "B_tot_contra"])
def test_equilibrium_fallback(tmp_path, quantity):
  """Missing Apar retains the existing equilibrium-only fallback under projection."""
  prefix = _prefix(2)
  for source in DATA.glob(prefix + "-geo_int_*.gkyl"):
    shutil.copy2(source, tmp_path / source.name)
  result, = load_quantity(quantity,
                          None,
                          prefix,
                          None,
                          path=str(tmp_path),
                          direction="Z")
  bmag = drift_projection_reference(2)[8]
  np.testing.assert_allclose(result.values,
                             _expected(bmag[..., None] * np.array([0., 0., 1.]),
                                       "Z", 2),
                             rtol=2e-12,
                             atol=2e-13)


def test_projection_contract():
  """Reject incomplete vectors and incompatible geometry instead of interpolating."""
  vector = _read("drift_cart", "vector_cartesian")
  components = [vector.select(comp=i) for i in range(3)]
  kwargs = dict(basis="cartesian", direction="Z")
  with pytest.raises(ValueError, match="three scalar"):
    operations.project_vector(components[:2], **kwargs)
  with pytest.raises(ValueError, match="basis must"):
    operations.project_vector(components, basis="unknown", direction="Z")
  with pytest.raises(ValueError, match="direction"):
    operations.project_vector(components, basis="cartesian", direction="0")
  for basis, geometry in (("contravariant", "tangents"), ("covariant",
                                                          "duals")):
    with pytest.raises(ValueError, match=geometry):
      operations.project_vector(components, basis=basis, direction="Z")
  with pytest.raises(ValueError, match="1 fields"):
    operations.project_vector([vector] * 3, **kwargs)
  bad = components[0].clone()
  bad.grid[0][:] += 1
  with pytest.raises(ValueError, match="different grids"):
    operations.project_vector([bad, *components[1:]], **kwargs)
  with pytest.raises(ValueError, match="different value_forms"):
    operations.project_vector(
        [components[0].represent(to="quad"), *components[1:]], **kwargs)
  with pytest.raises(ValueError, match="different backends"):
    operations.project_vector([components[0].interpolate(), *components[1:]],
                              **kwargs)
  with pytest.raises(ValueError, match="normals"):
    operations.project_vector(components, basis="cartesian", direction="n0")
  positions = _read("drift_cart", "geo_int_mapc2p")
  with pytest.raises(ValueError, match="position_system"):
    operations.project_vector(components,
                              basis="cartesian",
                              direction="R",
                              positions=positions)
  with pytest.raises(ValueError, match="R > 0"):
    operations.project_vector(components,
                              basis="cartesian",
                              direction="phi",
                              positions=positions * 0,
                              position_system="cylindrical")


RENAMES = {
    "ExB_vel": "ExB_vel_contra",
    "gradB_vel": "gradB_vel_contra",
    "diamag_vel": "diamag_vel_contra",
    "flux_particle_ExB": "flux_particle_ExB_contra",
    "flux_energy_ExB": "flux_energy_ExB_contra",
    "flux_particle_dB": "flux_particle_dB_contra",
    "flux_energy_dB": "flux_energy_dB_contra",
    "flux_particle": "flux_particle_contra",
    "flux_energy": "flux_energy_contra",
    "inv_L_n": "inv_L_n_cov",
    "inv_L_T": "inv_L_T_cov",
}


@pytest.mark.parametrize("old,new", RENAMES.items())
def test_clean_renames(old, new):
  """Only the explicit basis name is registered; old CLI names fail clearly."""
  assert not gk_quant_registry.has(old)
  quantity = gk_quant_registry.get(new)
  assert quantity is not None
  assert quantity.component_basis == ("covariant" if new.endswith("_cov") else
                                      "contravariant")
  result = CliRunner().invoke(cli, [
      "gk_load_q", "-q", old, "-n", "drift_cart", "-s", "ion", "-p",
      str(DATA), "-f", "0", "-d", "0"
  ])
  assert result.exit_code != 0
  assert "Invalid value for '--quantity'" in result.output
  assert old in result.output


def test_registry_projection_scope_and_geo_name():
  """Capability metadata enables fields/drifts, leaving packed geometry unchanged."""
  enabled = {
      q
      for q in gk_quant_registry.list()
      if gk_quant_registry.get(q).projection_label
  }
  assert enabled == {
      "ExB_vel_contra", "gradB_vel_contra", "diamag_vel_contra", "E_field_cov",
      "E_field_contra", "dB_perp_cov", "dB_perp_contra", "B_tot_cov",
      "B_tot_contra"
  }
  geometry = gk_quant_registry.get("geo_int_b_i")
  assert geometry.component_basis == "covariant"
  assert geometry.projection_label is None
  result, = load_quantity("geo_int_b_i",
                          None,
                          "drift_cart",
                          None,
                          path=str(DATA))
  np.testing.assert_array_equal(result.values,
                                _read("drift_cart", "geo_int_b_i").values)
  for name in ("particle_D", "heat_chi", "particle_D_gB", "heat_chi_gB"):
    quantity = gk_quant_registry.get(name)
    if quantity is not None:
      assert quantity.component_basis is None


@pytest.mark.parametrize("quantity", [
    "E_field_cov", "E_field_contra", "dB_perp_cov", "dB_perp_contra",
    "B_tot_cov", "B_tot_contra"
])
def test_field_cli(quantity):
  result = CliRunner().invoke(cli, [
      "gk_load_q", "-q", quantity, "-n", "drift_2x_cart", "-p",
      str(DATA), "-s", "ion", "-f", "0", "-r", "dir=phi", "info"
  ])
  assert result.exit_code == 0, result.output
