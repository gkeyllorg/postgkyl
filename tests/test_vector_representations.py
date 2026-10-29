"""Independent metric and Cartesian-vector references for all GK vector families."""
from pathlib import Path
import shutil
import numpy as np
import pytest
from click.testing import CliRunner

import postgkyl as pg
from postgkyl import gpython, operations
from postgkyl.cli.app import cli, COMMANDS
from postgkyl.diagnostics.gk.load_quantity import load_quantity, _quantity_catalog
from postgkyl.diagnostics.gk.registry import gk_quant_registry
from postgkyl.diagnostics.gk import quantities as ff
from postgkyl.diagnostics.gk.vector_projection import fetch_vector
from generate_test_data import drift_projection_reference

DATA = Path(__file__).parent / "test_data" / "generated"
REPS = ("covar", "contra", "cyl", "cart", "norm")
VECTOR_NAMES = [
    "ExB_vel", "gradB_vel", "diamag_vel", "E_field", "dB_perp", "B_tot",
    "geo_int_b_i", "inv_L_n", "inv_L_T", "flux_particle_ExB",
    "flux_particle_dB", "flux_particle", "flux_energy_ExB", "flux_energy_dB",
    "flux_energy"
]
needs_gkeyll = pytest.mark.skipif(not gpython.available(),
                                  reason="requires Gkeyll")


def _read(prefix, suffix):
  return pg.load(str(DATA / f"{prefix}-{suffix}.gkyl"))


def _expected(cartesian, rep, direction, ndim):
  """Project analytic Cartesian samples using independently constructed basis vectors."""
  cells, _, _, _, modes, matrix, normals, positions, _, _, _ = drift_projection_reference(
      ndim)
  cartesian = np.broadcast_to(cartesian, (*cells, 2**ndim, 3))
  if rep == "covar":
    target = matrix[:, direction]
  elif rep == "contra":
    target = np.linalg.inv(matrix)[direction]
  elif rep == "norm":
    target = normals[direction]
  elif rep == "cart" or direction == 2:
    target = np.eye(3)[direction]
  else:
    radius = np.hypot(positions[..., 0], positions[..., 1])
    cosine, sine = positions[..., 0] / radius, positions[..., 1] / radius
    target = np.stack((cosine, sine, np.zeros_like(cosine)), axis=-1)
    if direction == 1:
      target = np.stack((-sine, cosine, np.zeros_like(cosine)), axis=-1)
  samples = np.sum(cartesian * target, axis=-1)
  return np.einsum("...q,qm->...m", samples, modes).reshape(*cells, 2**ndim)


def _cartesian_reference(quantity, ndim):
  """Known fields for n=2+.2*f, u=.3, T=4, B=2+.1*f, phi=1+.4*f."""
  _, _, _, points, _, matrix, _, _, bmag, cross, _ = drift_projection_reference(
      ndim)
  f = points[..., 0] + (.3 * points[..., 2] if ndim == 2 else 0.)
  density = 2 + .2 * f
  gradient = np.linalg.inv(matrix)[0] + (.3 * np.linalg.inv(matrix)[2]
                                         if ndim == 2 else 0.)
  equilibrium_unit = np.array([0., 0., 1.])
  exb = (.4 / bmag)[..., None] * cross
  perturbation = .6 * np.cross(gradient, equilibrium_unit)
  if quantity == "ExB_vel": return exb
  if quantity == "gradB_vel": return (.4 / bmag**2)[..., None] * cross
  if quantity == "diamag_vel":
    return (.8 / (density * bmag))[..., None] * cross
  if quantity.startswith("E_field"): return -.4 * gradient
  if quantity.startswith("dB_perp"): return perturbation
  if quantity.startswith("B_tot"):
    return bmag[..., None] * equilibrium_unit + perturbation
  if quantity == "geo_int_b_i": return equilibrium_unit
  if quantity == "inv_L_n": return (-.2 / density)[..., None] * gradient
  if quantity == "inv_L_T": return gradient * 0
  particle = density[..., None] * exb
  flutter = (.3 * density / bmag)[..., None] * perturbation
  if quantity.startswith("flux_energy"):
    particle = particle * .5 * (.3**2 + 12)
    flutter = flutter * .5 * (.3**3 + 6) / .3
  if quantity.endswith("_ExB"): return particle
  if quantity.endswith("_dB"): return flutter
  return particle + flutter


@needs_gkeyll
@pytest.mark.parametrize("quantity,ndim",
                         [(q, d) for q in VECTOR_NAMES for d in (2, 3)])
@pytest.mark.parametrize("rep", REPS)
@pytest.mark.parametrize("direction", [0, 1, 2])
def test_all_vector_families(quantity, ndim, rep, direction):
  """Check all coefficients, including mixed components and the third flux component."""
  prefix = "repr_cart" if ndim == 3 else "repr_2x_cart"
  out, = load_quantity(quantity,
                       "ion",
                       prefix,
                       "0",
                       path=str(DATA),
                       direction=direction,
                       rep=rep)
  np.testing.assert_allclose(out.values,
                             _expected(_cartesian_reference(quantity, ndim),
                                       rep, direction, ndim),
                             rtol=3e-11,
                             atol=4e-13)
  assert out.ctx["vector_representation"] == rep
  assert out.ctx["projection_direction"] == direction
  if rep in ("cyl", "cart", "norm"):
    assert out.ctx["projection_units"] == gk_quant_registry.get(
        quantity).unit_vector_units


@needs_gkeyll
@pytest.mark.parametrize("basis", ["covariant", "contravariant", "cartesian"])
@pytest.mark.parametrize("rep", REPS)
@pytest.mark.parametrize("direction", [0, 1, 2])
@pytest.mark.parametrize("ndim", [2, 3])
def test_general_basis_conversion(basis, rep, direction, ndim):
  """An affine vector in a nonorthogonal map verifies raising/lowering every index."""
  prefix = "drift_cart" if ndim == 3 else "drift_2x_cart"
  vector = _read(prefix, "vector_" + basis)
  points = drift_projection_reference(ndim)[3]
  f = points[..., 0] + (.3 * points[..., 2] if ndim == 2 else 0.)
  cartesian = np.stack(
      (1 + .2 * points[..., 0], -.7 + .1 * points[..., 2], .4 + .05 * f),
      axis=-1)
  out = operations.project_vector([vector.select(comp=i) for i in range(3)],
                                  basis=basis,
                                  rep=rep,
                                  direction=direction,
                                  metric_cov=_read(prefix, "geo_int_g_ij"),
                                  metric_contra=_read(prefix, "geo_int_gij"),
                                  tangents=_read(prefix, "geo_int_dxdz"),
                                  duals=_read(prefix, "geo_int_dzdx"),
                                  normals=_read(prefix, "geo_int_normals"),
                                  positions=_read(prefix, "geo_int_mapc2p"),
                                  position_system="cartesian")
  np.testing.assert_allclose(out.values,
                             _expected(cartesian, rep, direction, ndim),
                             rtol=2e-12,
                             atol=3e-13)


@needs_gkeyll
@pytest.mark.parametrize("rep", REPS)
@pytest.mark.parametrize("direction", [0, 1, 2])
@pytest.mark.parametrize("ndim", [2, 3])
def test_nonconstant_temperature_log_gradient(rep, direction, ndim):
  """A nonzero temperature gradient checks reduced derivatives and weak inversion."""
  prefix = "drift_cart" if ndim == 3 else "drift_2x_cart"
  temperature = _read(prefix, "field_0")
  geometry = {
      key: _read(prefix, "geo_int_" + suffix)
      for key, suffix in (("metric_contra", "gij"), ("duals", "dzdx"),
                          ("normals", "normals"), ("positions", "mapc2p"))
  }
  geometry["position_system"] = "cartesian"
  out = fetch_vector(ff.fetch_inv_L_T_cov, [temperature],
                     gk_quant_registry.get("inv_L_T"),
                     dir=direction,
                     rep=rep,
                     vector_geometry=geometry)
  _, _, _, points, _, matrix, _, _, _, _, _ = drift_projection_reference(ndim)
  f = points[..., 0] + (.3 * points[..., 2] if ndim == 2 else 0.)
  gradient = np.linalg.inv(matrix)[0] + (.3 * np.linalg.inv(matrix)[2]
                                         if ndim == 2 else 0.)
  cartesian = (-.4 / (1 + .4 * f))[..., None] * gradient
  np.testing.assert_allclose(out.values,
                             _expected(cartesian, rep, direction, ndim),
                             rtol=3e-12,
                             atol=3e-13)


@needs_gkeyll
@pytest.mark.parametrize("quantity", VECTOR_NAMES)
def test_default_rep_matches_explicit_native(quantity):
  args = (quantity, "ion", "repr_cart", "0")
  default, = load_quantity(*args, path=str(DATA), direction=1)
  native = "covar" if gk_quant_registry.get(
      quantity).component_basis == "covariant" else "contra"
  explicit, = load_quantity(*args, path=str(DATA), direction=1, rep=native)
  np.testing.assert_array_equal(default.values, explicit.values)


@needs_gkeyll
def test_geometry_requirements_and_packed_default(tmp_path):
  prefix = "repr_cart"
  for suffix in ("field_0", "geo_int_gij", "geo_int_b_i"):
    source = DATA / f"{prefix}-{suffix}.gkyl"
    shutil.copy2(source, tmp_path / source.name)
  # Native E needs only phi; covariant-to-contravariant needs only g^ij.
  for rep in (None, "covar", "contra"):
    out, = load_quantity("E_field",
                         None,
                         prefix,
                         "0",
                         path=str(tmp_path),
                         direction=0,
                         rep=rep)
    assert out.ctx["value_form"] == "modal"
  with pytest.raises(FileNotFoundError, match="geo_int_dzdx"):
    load_quantity("E_field",
                  None,
                  prefix,
                  "0",
                  path=str(tmp_path),
                  direction=0,
                  rep="cart")
  packed, = load_quantity("geo_int_b_i", None, prefix, None, path=str(tmp_path))
  np.testing.assert_array_equal(packed.values,
                                _read(prefix, "geo_int_b_i").values)
  with pytest.raises(ValueError, match="direction="):
    load_quantity("geo_int_b_i",
                  None,
                  prefix,
                  None,
                  path=str(tmp_path),
                  rep="covar")
  with pytest.raises(FileNotFoundError, match="geo_int_g_ij"):
    load_quantity("ExB_vel",
                  "ion",
                  prefix,
                  "0",
                  path=str(tmp_path),
                  direction=0,
                  rep="covar")


@pytest.mark.parametrize("quantity",
                         ["M0", "geo_int_gij", "particle_D", "heat_chi"])
def test_nonvector_rep_rejected(quantity):
  with pytest.raises(ValueError, match="not a vector"):
    load_quantity(quantity, "ion", "absent", "0", rep="contra", direction=0)


@pytest.mark.parametrize("direction",
                         ["R", "Z", "phi", "n0", "n1", "n2", -1, 3, 1.5])
def test_named_and_invalid_directions_rejected(direction):
  with pytest.raises(ValueError, match="Choose direction="):
    load_quantity("ExB_vel", "ion", "absent", "0", direction=direction)


def test_catalog_defaults_and_short_options():
  for name, description in _quantity_catalog():
    quantity = gk_quant_registry.get(name)
    assert not description.startswith(("Covariant", "Contravariant"))
    if quantity.component_basis:
      assert description.endswith(f"(default: {quantity.component_basis})")
    else:
      assert "(default:" not in description
  commands = {c.name: c for c in COMMANDS}
  for name in ("load", "gk_load_quantity"):
    options = {p.name: p for p in commands[name].params}
    assert options["options"].opts == ["--options", "-o"]
    assert "read_options" not in options
  assert next(p for p in commands["gk_load_quantity"].params
              if p.name == "rep").opts == ["--rep", "-r"]


@needs_gkeyll
def test_python_and_cli_options_precedence():
  kwargs = dict(path=str(DATA),
                options={
                    "dir": "0",
                    "rep": "norm"
                },
                direction=2,
                rep="cyl")
  out, = load_quantity("ExB_vel", "ion", "repr_2x_cart", "0", **kwargs)
  np.testing.assert_allclose(out.values, 0, rtol=0, atol=3e-13)
  assert out.ctx["vector_representation"] == "cyl"
  assert out.ctx["projection_direction"] == 2
  result = CliRunner().invoke(cli, [
      "gk_load_q", "-q", "ExB_vel", "-s", "ion", "-n", "repr_2x_cart", "-p",
      str(DATA), "-f", "0", "-o", "dir=0", "-o", "rep=norm", "-d", "2", "-r",
      "cyl", "info"
  ])
  assert result.exit_code == 0, result.output
  with pytest.raises(TypeError, match="read_options"):
    load_quantity("ExB_vel", "ion", "absent", read_options={})
  with pytest.raises(TypeError, match="read_options"):
    pg.load("absent", read_options={})
  for cmd in (["load", "--read_options",
               "x=y"], ["load", "-r",
                        "x=y"], ["gk_load_q", "--read_options",
                                 "dir=0"], ["gk_load_q", "-r", "dir=0"]):
    assert CliRunner().invoke(cli, cmd).exit_code != 0
  # Reader options still pass through general load's renamed interface.
  field = pg.load(str(DATA / "drift_cart-field_0.gkyl"), options={})
  assert field.num_dims == 3
  result = CliRunner().invoke(cli, [
      "load", "-f",
      str(DATA / "drift_cart-field_0.gkyl"), "-o", "comp=0", "info"
  ])
  assert result.exit_code == 0, result.output


@needs_gkeyll
@pytest.mark.parametrize("form", ["nodal", "quad", "interpolated"])
@pytest.mark.parametrize("source,rep", [("covariant", "contra"),
                                        ("contravariant", "covar")])
def test_metric_conversion_point_representations(form, source, rep):
  """Index conversion preserves point layouts and recovers known affine fields."""

  def read(suffix):
    field = _read("drift_2x_cart", suffix)
    return field.interpolate() if form == "interpolated" else field.represent(
        to=form)

  vector = read("vector_" + source)
  expected = read("vector_" +
                  ("contravariant" if rep == "contra" else "covariant"))
  for direction in range(3):
    out = operations.project_vector([vector.select(comp=i) for i in range(3)],
                                    basis=source,
                                    rep=rep,
                                    direction=direction,
                                    metric_cov=read("geo_int_g_ij"),
                                    metric_contra=read("geo_int_gij"))
    np.testing.assert_allclose(out.values,
                               expected.select(comp=direction).values,
                               rtol=2e-13,
                               atol=3e-13)
    assert out.backend == vector.backend
    assert out.ctx.get("value_form") == vector.ctx.get("value_form")


@needs_gkeyll
@pytest.mark.parametrize("rep", REPS)
def test_fluctuating_flux_conversion(rep):
  """Binormally uniform data have zero turbulent flux in every representation."""
  for direction in range(3):
    out, = load_quantity("flux_particle",
                         "ion",
                         "repr_cart",
                         "0",
                         path=str(DATA),
                         direction=direction,
                         rep=rep,
                         fluct="y")
    np.testing.assert_allclose(out.values, 0, rtol=0, atol=3e-13)


@needs_gkeyll
def test_metric_contract_errors():
  vector = _read("drift_cart", "vector_contravariant")
  components = [vector.select(comp=i) for i in range(3)]
  with pytest.raises(ValueError, match="metric_cov"):
    operations.project_vector(components,
                              basis="contravariant",
                              rep="covar",
                              direction=0)
  with pytest.raises(ValueError, match="6 fields"):
    operations.project_vector(components,
                              basis="contravariant",
                              rep="covar",
                              direction=0,
                              metric_cov=vector)
  with pytest.raises(ValueError, match="Choose rep"):
    operations.project_vector(components,
                              basis="contravariant",
                              rep="invalid",
                              direction=0)
  with pytest.raises(ValueError, match="Choose rep"):
    load_quantity("ExB_vel",
                  "ion",
                  "absent",
                  "0",
                  options={
                      "rep": "invalid",
                      "dir": "0"
                  })


@pytest.mark.parametrize("quantity,rep,units", [
    ("ExB_vel", "covar", "covariant component units"),
    ("E_field", "contra", "contravariant component units"),
    ("E_field", "cart", "V/m"),
    ("inv_L_T", "norm", "1/m"),
])
def test_labels_follow_requested_representation(quantity, rep, units):
  """Direct label requests also describe the output basis and its units."""
  label = gk_quant_registry.get(quantity).get_label(direction=0, rep=rep)
  assert label.endswith(f"({units})")


@needs_gkeyll
@pytest.mark.parametrize("quantity",
                         [q for q in VECTOR_NAMES if q.startswith("flux_")])
@pytest.mark.parametrize("fluct", ["y", "yz"])
@pytest.mark.parametrize("rep", REPS)
@pytest.mark.parametrize("direction", [0, 1, 2])
def test_2x_flux_fluctuations(quantity, fluct, rep, direction):
  """Nonzero parallel correlations, from independent tensor Gauss averages.

  The affine fixture has constant J. Two-point Gauss integration is exact
  for the products of retained p1 fields; reciprocal fields are represented
  by their independently generated Gauss projections.
  """
  cells, _, _, points, _, matrix, _, _, bmag, cross, _ = drift_projection_reference(
      2)
  density = 2 + .2 * (points[..., 0] + .3 * points[..., 2])

  def parallel_fluctuation(samples):
    # Gauss node order is (sx,sz); average both z cells and z nodes at each x.
    samples = np.broadcast_to(samples, (*cells, 4, 3))
    shaped = samples.reshape(*cells, 2, 2, 3)
    mean = shaped.mean(axis=(1, 3), keepdims=True)
    return (shaped - mean).reshape(*cells, 4, 3)

  moment = parallel_fluctuation(density[..., None])
  exb = moment * parallel_fluctuation((.4 / bmag)[..., None] * cross)
  gradient = np.linalg.inv(matrix)[0] + .3 * np.linalg.inv(matrix)[2]
  flutter = .3 * moment * parallel_fluctuation(
      (.6 / bmag)[..., None] * np.cross(gradient, [0., 0., 1.]))
  if quantity.startswith("flux_energy"):
    exb *= .5 * (.3**2 + 12)
    flutter *= .5 * (.3**3 + 6) / .3
  cartesian = exb if quantity.endswith(
      "_ExB") else flutter if quantity.endswith("_dB") else exb + flutter
  out, = load_quantity(quantity,
                       "ion",
                       "repr_2x_cart",
                       "0",
                       path=str(DATA),
                       direction=direction,
                       rep=rep,
                       fluct=fluct)
  np.testing.assert_allclose(out.values,
                             _expected(cartesian, rep, direction, 2),
                             rtol=3e-11,
                             atol=4e-13)


@needs_gkeyll
@pytest.mark.parametrize("fluct", ["y", "yz"])
def test_2x_fluctuation_uses_nonuniform_weight(fluct):
  """Weight w=2+.1*f shifts the parallel mean of n=2+.2*f, f=x+.3*z."""
  cells, _, _, points, basis, _, _, _, weight, _, _ = drift_projection_reference(
      2)
  density = 2 + .2 * (points[..., 0] + .3 * points[..., 2])
  n = density.reshape(*cells, 2, 2)
  w = weight.reshape(*cells, 2, 2)
  mean = (n * w).sum(axis=(1, 3), keepdims=True) / w.sum(axis=(1, 3),
                                                         keepdims=True)
  samples = (n - mean).reshape(*cells, 4)
  expected = np.einsum("...q,qm->...m", samples, basis)
  out = ff._maybe_fluct(_read("repr_2x_cart", "ion_M0_0"),
                        _read("repr_2x_cart", "geo_int_bmag"),
                        "test_weighted_flux",
                        fluct=fluct)
  np.testing.assert_allclose(out.values, expected, rtol=3e-12, atol=3e-13)


@needs_gkeyll
def test_2x_flux_cli():
  """The CLI accepts 2x flutter flux with a parallel fluctuation average."""
  result = CliRunner().invoke(cli, [
      "gk_load_q", "-q", "flux_particle_dB", "-s", "ion", "-n", "repr_2x_cart",
      "-p",
      str(DATA), "-f", "0", "-r", "cyl", "-d", "0", "--fluct", "yz", "info"
  ])
  assert result.exit_code == 0, result.output
