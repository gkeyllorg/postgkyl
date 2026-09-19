"""Cell-center DG selection preserves the polynomial on surviving axes."""

from pathlib import Path

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython

pytestmark = pytest.mark.skipif(not gpython.available(), reason="needs Gkeyll")
FIELD = Path(__file__).parent / "test_data/generated/select_2d_tensor_p2.gkyl"


@pytest.fixture
def data():
  return pg.load(FIELD)


@pytest.mark.parametrize("selector,cell", [
    (0, 0),
    (-1, 3),
    (np.int64(1), 1),
    (0.0, 0),
    (1.9, 1),
    (2.0, 1),
    (-100.0, 0),
    (100.0, 3),
    ("2", 2),
    ("2.0", 1),
])
def test_scalar_evaluates_at_center_and_removes_dimension(data, selector, cell):
  result = data.select(z0=selector)
  assert result.backend == "gkyl"
  assert result.ctx["value_form"] == "modal"
  assert result.num_dims == 1
  np.testing.assert_array_equal(result.grid[0], data.grid[1])
  points = np.array([[-0.7], [0.2], [0.8]])
  donor_basis = gpython.basis.eval_matrix("tensor", 2, 2, np.c_[np.zeros(3),
                                                                points])
  target_basis = gpython.basis.eval_matrix(result.ctx["basis_type"], 1, 2,
                                           points)
  expected = np.einsum("pk,cfk->cpf", donor_basis,
                       data.values[cell].reshape(3, 2, 9))
  actual = np.einsum("pk,cfk->cpf", target_basis,
                     result.values.reshape(3, 2, 3))
  np.testing.assert_allclose(actual, expected, atol=1e-12)
  assert data.num_dims == 2


@pytest.mark.parametrize("selector,expected", [
    ("1:3", slice(1, 3)),
    (":", slice(0, 4)),
    ("1:", slice(1, 4)),
    (":-1", slice(0, 3)),
    ("-2:", slice(2, 4)),
    ("0.6:2.6", slice(0, 2)),
])
def test_ranges_preserve_whole_cells(data, selector, expected):
  result = data.select(z0=selector)
  assert result.num_dims == 2
  np.testing.assert_array_equal(result.values, data.values[expected])
  np.testing.assert_array_equal(result.grid[0],
                                data.grid[0][expected.start:expected.stop + 1])
  assert result.ctx["basis_type"] == "tensor"
  np.testing.assert_array_equal(result.ctx["cells"], result.values.shape[:-1])
  np.testing.assert_allclose(
      result.interpolate().values,
      data.interpolate().values[expected.start * 3:expected.stop * 3])


@pytest.mark.parametrize("comp,fields", [(1, [1]), (-1, [1]), ("0,1", [0, 1]),
                                         ("1,0", [1, 0]), ("1:", [1])])
def test_components_are_complete_fields(data, comp, fields):
  result = data.select(comp=comp)
  expected = data.values.reshape(4, 3, 2,
                                 9)[...,
                                    fields, :].reshape(4, 3,
                                                       len(fields) * 9)
  np.testing.assert_array_equal(result.values, expected)
  np.testing.assert_allclose(result.interpolate().values,
                             data.interpolate().values[..., fields])


def test_mixed_selection_and_inplace(data):
  expected = data.select(z0=1).select(z0="1:", comp=1)
  result = data.select(z0=1,
                       z1="1:",
                       comp=1,
                       inplace=True,
                       tag="slice",
                       label="Center slice")
  assert result is data
  assert result.tag == "slice"
  assert result.label == "Center slice"
  np.testing.assert_allclose(result.values, expected.values)
  np.testing.assert_array_equal(result.grid[0], [1., 2., 3.])
  np.testing.assert_array_equal(result.ctx["lower"], [1.])
  np.testing.assert_array_equal(result.ctx["upper"], [3.])


def test_all_dimensions_evaluate_to_constant_dummy_cell(data):
  result = data.select(z0=1, z1=-1)
  basis = gpython.basis.eval_matrix("tensor", 2, 2, np.zeros((1, 2)))[0]
  expected = data.values[1, -1].reshape(2, 9) @ basis
  np.testing.assert_allclose(result.interpolate().values,
                             expected[None, :],
                             atol=1e-12)
  np.testing.assert_array_equal(result.grid[0], [0., 1.])


@pytest.mark.parametrize("kwargs,error", [
    ({
        "z0": 4
    }, IndexError),
    ({
        "z0": -5
    }, IndexError),
    ({
        "z0": "2:2"
    }, ValueError),
    ({
        "z0": "3:1"
    }, ValueError),
    ({
        "z0": np.nan
    }, ValueError),
    ({
        "z0": np.inf
    }, ValueError),
    ({
        "comp": "1:1"
    }, ValueError),
    ({
        "comp": 2
    }, IndexError),
])
def test_invalid_selection_does_not_mutate(data, kwargs, error):
  original = data.values.copy()
  with pytest.raises(error):
    data.select(**kwargs, inplace=True)
  np.testing.assert_array_equal(data.values, original)


def test_cli_float_selector_removes_dimension():
  from click.testing import CliRunner
  from postgkyl.cli.app import cli

  result = CliRunner().invoke(
      cli, [str(FIELD), "select", "--z0", "0.0", "--comp", "1", "info"])
  assert result.exit_code == 0, result.output
  assert "Number of dimensions: 1" in result.output
