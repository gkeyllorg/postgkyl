"""Independent physical references for RPN and multi-dataset composition.

The polynomial fixtures use exact Legendre integrals in generate_test_data.py.
Their two 1D fields are f=2+x and g=4-x/4 on [-1,2]; the 2D fields
multiply these by 2+2y and 4-y/2 respectively on y in [1/4,9/4].
"""

from pathlib import Path

import numpy as np
import pytest

import postgkyl as pg

GEN = Path(__file__).parent / "test_data" / "generated"
needs_gkeyll = pytest.mark.skipif(not pg.gpython.available(),
                                  reason="requires the compiled Gkeyll bridge")
ROUND_OFF = dict(rtol=2e-13, atol=2e-13)


@needs_gkeyll
@pytest.mark.parametrize("form", ["modal", "nodal", "quad"])
def test_rpn_integral_counts_physical_fields_and_preserves_exact_measure(form):
  data = pg.load(GEN / "polynomial_1d_ms_p1.gkyl").represent(to=form)
  result = pg.evaluate("f 0 int", data)
  np.testing.assert_allclose(result.values, [7.5, 11.625], **ROUND_OFF)
  assert result.num_dims == 0
  assert result.num_comps == 2
  assert result.ctx.get("value_form") is None
  mean = pg.evaluate("f 0 avg", data)
  np.testing.assert_allclose(mean.values, [2.5, 3.875], **ROUND_OFF)


@needs_gkeyll
def test_rpn_partial_integral_retains_the_surviving_polynomial():
  data = pg.load(GEN / "polynomial_2d_ms_p1.gkyl")
  reduced = pg.evaluate("f 0 int", data)
  assert reduced.num_dims == 1
  values = reduced.interpolate(num_interp=3)
  y = (values.grid[0][:-1] + values.grid[0][1:]) / 2
  expected = np.column_stack((7.5 * (2 + 2 * y), 11.625 * (4 - y / 2)))
  np.testing.assert_allclose(values.values, expected, **ROUND_OFF)


@needs_gkeyll
@pytest.mark.parametrize("form", ["nodal", "quad"])
def test_rpn_dot_unpacks_nodes_before_contracting_physical_components(form):
  data = pg.load(GEN / "polynomial_1d_ms_p1.gkyl").represent(to=form)
  result = pg.evaluate("f f dot", data)
  x = result.grid[0]
  assert result.values.shape == (len(x), 1)
  expected = (2 + x)**2 + (4 - x / 4)**2
  np.testing.assert_allclose(result.values[:, 0], expected, **ROUND_OFF)
  assert result.ctx.get("value_form") is None


@needs_gkeyll
@pytest.mark.parametrize("form", ["nodal", "quad"])
@pytest.mark.parametrize("token", ["div", "curl"])
def test_rpn_rejects_stencils_on_packed_discontinuous_nodes(form, token):
  data = pg.load(GEN / "polynomial_1d_ms_p1.gkyl").represent(to=form)
  with pytest.raises(ValueError, match="unpacked point grid"):
    pg.evaluate(f"f {token}", data)


@needs_gkeyll
def test_rpn_addition_of_independent_loads_has_analytic_values():
  filename = GEN / "polynomial_2d_ms_p1.gkyl"
  a, b = pg.load(filename).interpolate(), pg.load(filename).interpolate()
  a.ctx["nested"] = {"array": [np.arange(6).reshape(2, 3)]}
  b.ctx["nested"] = {"array": [np.arange(6).reshape(2, 3)]}
  result = pg.evaluate("f0 f1 +", a, b)
  x, y = np.meshgrid(*[(g[:-1] + g[1:]) / 2 for g in result.grid],
                     indexing="ij")
  expected = 2 * np.stack(
      ((2 + x) * (2 + 2 * y), (4 - x / 4) * (4 - y / 2)), axis=-1)
  np.testing.assert_allclose(result.values, expected, **ROUND_OFF)
  assert "_load_metadata" not in result.ctx
  np.testing.assert_array_equal(result.ctx["nested"]["array"][0],
                                np.arange(6).reshape(2, 3))


@needs_gkeyll
def test_weighted_average_checks_collocation_and_accepts_one_weight_field():
  data = pg.load(GEN / "polynomial_1d_ms_p1.gkyl")
  weight = data.select(comp=0)
  result = data.average([0], weight=weight).interpolate()
  # int(f^2)=21, int(f*g)=28.5, int(f)=7.5.
  np.testing.assert_allclose(result.values, [[2.8, 3.8]] * 2, **ROUND_OFF)
  shifted = weight.clone()
  shifted.grid = [shifted.grid[0] + 10]
  with pytest.raises(ValueError, match="different grids"):
    data.average([0], weight=shifted)


@needs_gkeyll
@pytest.mark.parametrize("scale,offset,shift", [(1e-10, 0., 1e-9),
                                                (1., 1e9, 0.1)])
def test_grid_collocation_is_independent_of_units_and_origin(
    scale, offset, shift):
  data = pg.load(GEN / "polynomial_1d_ms_p1.gkyl")
  data.grid = [data.grid[0] * scale + offset]
  weight = data.select(comp=0)
  weight.grid = [weight.grid[0] + shift]
  with pytest.raises(ValueError, match="different grids"):
    data.average([0], weight=weight)
  with pytest.raises(ValueError, match="different grids"):
    data.select(comp=0) * weight


@needs_gkeyll
def test_native_average_rejects_nonuniform_cells_before_uniform_grid_kernel():
  data = pg.load(GEN / "polynomial_1d_ms_p1.gkyl")
  data.grid[0][1] += 0.25
  with pytest.raises(ValueError, match="uniform Cartesian cell edges"):
    data.average([0], weight=data.select(comp=0))


@needs_gkeyll
def test_native_integral_resolves_current_coordinates_instead_of_cached_bounds(
):
  data = pg.load(GEN / "polynomial_1d_ms_p1.gkyl")
  data.grid[0][:] *= 3
  # The reference-cell polynomial is unchanged; tripling every physical cell
  # width triples each integral. Mutating coordinates must not use old bounds.
  np.testing.assert_allclose(data.integrate(), [22.5, 34.875], **ROUND_OFF)


@needs_gkeyll
def test_native_uniform_grid_accepts_coordinate_roundoff_at_large_origin():
  data = pg.load(GEN / "polynomial_1d_ms_p1.gkyl")
  data.grid = [np.linspace(1e9, 1e9 + 1, 4)]
  # The coordinate differences are not bitwise equal, but the grid is exactly
  # reconstructed from its bounds. In xi=x-1e9, f=1+3*xi, g=4.25-0.75*xi.
  assert np.ptp(np.diff(data.grid[0])) > 1e-8
  np.testing.assert_allclose(data.integrate(), [2.5, 3.875], **ROUND_OFF)
  average = data.average([0], weight=data.select(comp=0)).interpolate()
  # int(f*f)/int(f)=2.8; int(f*g)/int(f)=3.8, invariant under this affine map.
  np.testing.assert_allclose(average.values, [[2.8, 3.8]] * 2, **ROUND_OFF)


def test_rpn_gradient_and_integral_respect_nonuniform_point_coordinates():
  x = np.array([-1., -0.25, 0.5, 2.])
  data = pg.GData().push([x], (x * x)[:, None])
  derivative = pg.evaluate("f grad", data)
  np.testing.assert_allclose(derivative.values[:, 0], 2 * x, **ROUND_OFF)
  linear = pg.GData().push([x], (2 + x)[:, None])
  np.testing.assert_allclose(
      pg.evaluate("f 0 int", linear).values, [7.5], **ROUND_OFF)


def test_rpn_axis_scaling_does_not_mutate_the_source_geometry():
  x = np.array([-1., 0., 2.])
  data = pg.GData().push([x.copy()], (x * x)[:, None])
  result = pg.evaluate("f 0 3 scale_zi_axis", data)
  np.testing.assert_array_equal(data.grid[0], x)
  np.testing.assert_array_equal(result.grid[0], 3 * x)


@needs_gkeyll
def test_collect_preserves_collocated_samples_and_falls_back_from_missing_time(
):
  first = pg.load(GEN / "polynomial_1d_ms_p1.gkyl").interpolate()
  first.ctx.update(time=None, frame=3)
  second = first * 2
  second.ctx.update(time=None, frame=7)
  result = pg.collect(first, second)
  np.testing.assert_array_equal(result.grid[0], [3, 7])
  x = (first.grid[0][:-1] + first.grid[0][1:]) / 2
  field = np.column_stack((2 + x, 4 - x / 4))
  np.testing.assert_allclose(result.values, np.stack((field, 2 * field)),
                             **ROUND_OFF)
  second.grid[0] += 10
  with pytest.raises(ValueError, match="different grids"):
    pg.collect(first, second)
