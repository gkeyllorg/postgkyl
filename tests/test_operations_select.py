"""Modal DG selection preserves whole cells and their coordinate axes."""

from pathlib import Path

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython
from generate_test_data import generate_all

pytestmark = pytest.mark.skipif(not gpython.available(),
                                reason="needs compiled Gkeyll")
FIELD = Path(__file__).parent / "test_data/generated/select_2d_tensor_p2.gkyl"


@pytest.fixture
def data():
  return pg.load(FIELD)


def test_selection_fixture_generated_from_scratch(tmp_path):
  generate_all(tmp_path)
  data = pg.load(tmp_path / FIELD.name)
  assert data.values.shape == (4, 3, 18)
  assert data.ctx["basis_type"] == "tensor"
  assert data.ctx["poly_order"] == 2
  np.testing.assert_array_equal(data.grid[0], np.arange(5))
  np.testing.assert_array_equal(data.grid[1], np.arange(4))


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
  expected = data.values[1:2, 1:, 9:].copy()
  result = data.select(z0=1,
                       z1="1:",
                       comp=1,
                       inplace=True,
                       tag="slice",
                       label="Cell slice")
  assert result is data
  assert result.tag == "slice"
  assert result.label == "Cell slice"
  np.testing.assert_array_equal(result.values, expected)
  np.testing.assert_array_equal(result.grid[0], [1., 2.])
  np.testing.assert_array_equal(result.grid[1], [1., 2., 3.])
  np.testing.assert_array_equal(result.num_cells, [1, 2])
  np.testing.assert_array_equal(result.ctx["lower"], [1., 1.])
  np.testing.assert_array_equal(result.ctx["upper"], [2., 3.])


def test_selecting_all_axes_preserves_one_complete_cell(data):
  result = data.select(z0=1, z1=-1)
  assert result.num_dims == 2
  np.testing.assert_array_equal(result.num_cells, [1, 1])
  np.testing.assert_array_equal(result.values, data.values[1:2, -1:])
  np.testing.assert_array_equal(result.grid[0], [1., 2.])
  np.testing.assert_array_equal(result.grid[1], [2., 3.])
  np.testing.assert_allclose(result.interpolate().values,
                             data.interpolate().values[3:6, -3:])


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
