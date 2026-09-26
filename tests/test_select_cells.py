"""Selection copies stored cells and samples without evaluating the field."""
from pathlib import Path

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from postgkyl.numerics import idx_parser


@pytest.mark.parametrize("positions,coordinate,expected", [
    ([0., 1., 3.], 0.6, 1),
    ([0., 1., 3.], 2., 1),
    ([0., 1., 3.], -10., 0),
    ([0., 1., 3.], 10., 2),
    ([3., 1., 0.], 0.6, 1),
    ([2.], 100., 0),
])
def test_nearest_existing_sample(positions, coordinate, expected):
  assert idx_parser(coordinate, np.array(positions), nodal=True) == expected


@pytest.fixture
def modal():
  if not gpython.available():
    pytest.skip("needs Gkeyll")
  return pg.load(Path(__file__).parent / "test_data/generated/3d_ms_p1.gkyl")


@pytest.mark.parametrize("selector,index", [
    (1, 1),
    (np.int64(1), 1),
    (-1, 3),
    (0.4, 1),
    ("0.4", 1),
    (0.5, 1),
    (-100., 0),
    (100., 3),
])
def test_modal_scalar_preserves_coefficients_and_basis(modal, selector, index):
  selected = modal.select(z1=selector)
  assert selected.num_dims == 3
  assert selected.backend == "gkyl"
  assert selected.ctx["value_form"] == "modal"
  assert selected.ctx["basis_type"] == modal.ctx["basis_type"]
  assert selected.ctx["poly_order"] == modal.ctx["poly_order"]
  np.testing.assert_array_equal(selected.values, modal.values[:,
                                                              index:index + 1])
  np.testing.assert_array_equal(selected.grid[1],
                                modal.grid[1][index:index + 2])
  np.testing.assert_array_equal(selected.ctx["cells"], [4, 1, 4])
  np.testing.assert_allclose(
      selected.interpolate().values,
      modal.interpolate().values[:, 2 * index:2 * index + 2])
  assert modal.values.shape == (4, 4, 4, 8)


@pytest.mark.parametrize("selector,expected", [
    (":", slice(0, 4)),
    ("1:", slice(1, 4)),
    (":-1", slice(0, 3)),
    ("-2:", slice(2, 4)),
    ("0.4:0.9", slice(1, 3)),
])
def test_modal_ranges(modal, selector, expected):
  result = modal.select(z1=selector)
  np.testing.assert_array_equal(result.values, modal.values[:, expected])
  np.testing.assert_array_equal(result.grid[1],
                                modal.grid[1][expected.start:expected.stop + 1])


def test_fields_keep_complete_coefficient_blocks(modal):
  np.testing.assert_array_equal(modal.select(comp=0).values, modal.values)
  doubled = modal.select(comp="0,0")
  np.testing.assert_array_equal(doubled.values,
                                np.concatenate([modal.values] * 2, axis=-1))
  np.testing.assert_array_equal(doubled.select(comp="1:").values, modal.values)
  np.testing.assert_allclose(doubled.interpolate().values[..., 1],
                             modal.interpolate().values[..., 0])


def test_repeated_and_inplace_selection(modal):
  expected = modal.select(z1=1).select(z2=-1)
  result = modal.select(z1=1, z2=-1, inplace=True, tag="selected")
  assert result is modal
  assert result.tag == "selected"
  np.testing.assert_array_equal(result.values, expected.values)
  np.testing.assert_array_equal(result.ctx["lower"], [0., 0.25, 0.75])
  np.testing.assert_array_equal(result.ctx["upper"], [1., 0.5, 1.])


@pytest.mark.parametrize("kwargs,error", [
    ({
        "z1": 4
    }, IndexError),
    ({
        "z1": -5
    }, IndexError),
    ({
        "z1": "2:2"
    }, ValueError),
    ({
        "z1": "3:1"
    }, ValueError),
    ({
        "z1": np.nan
    }, ValueError),
    ({
        "z1": np.inf
    }, ValueError),
    ({
        "comp": "1:1"
    }, ValueError),
    ({
        "comp": 1
    }, IndexError),
])
def test_invalid_selection_preserves_input(modal, kwargs, error):
  values = modal.values.copy()
  with pytest.raises(error):
    modal.select(**kwargs, inplace=True)
  np.testing.assert_array_equal(modal.values, values)


def test_cli_modal_selection_retains_dimension(modal):
  from click.testing import CliRunner
  from postgkyl.cli.app import cli
  result = CliRunner().invoke(cli, [
      str(Path(__file__).parent / "test_data/generated/3d_ms_p1.gkyl"),
      "select", "--z1", "0.4", "info"
  ])
  assert result.exit_code == 0, result.output
  assert "Number of dimensions: 3" in result.output


def test_interpolated_multicomponent_selection_copies_nearest_point(modal):
  field = modal.select(comp="0,0").interpolate()
  selected = field.select(z1=0.22, z2=-1)
  np.testing.assert_array_equal(selected.values, field.values[:, 1:2, -1:])
  assert selected.values.shape == (8, 1, 1, 2)


def test_nonuniform_edges_snap_to_centers():
  # At x=1.1 the containing cell is index 1, but the nearest center is 0.5.
  assert idx_parser(1.1, np.array([0., 1., 5.])) == 0
