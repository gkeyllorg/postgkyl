"""Gkeyll geometry file conventions and coordinate decoding."""
import numpy as np
import pytest
from postgkyl.io import geometry as file_geometry


def test_gauss_nodes_are_ordered_inside_each_cell():
  nodes = file_geometry._gauss_nodes(np.array([0.0, 2.0, 4.0]))
  assert nodes.shape == (4, )
  assert np.all(np.diff(nodes) > 0.0)
  assert nodes[0] > 0.0 and nodes[-1] < 4.0


def test_geometry_components_support_cartesian_and_rz_layouts():
  mapc2p = {"geometry_type": file_geometry._MAPC2P_IDX}
  values = np.array([[[3.0, 4.0, 2.0], [0.0, 2.0, 5.0]]])
  major_r, vert_z, phi = file_geometry.geometry_components(
      values, mapc2p, "map.gkyl")
  np.testing.assert_allclose(major_r, [[5.0, 2.0]])
  np.testing.assert_allclose(vert_z, [[2.0, 5.0]])
  assert phi.shape == (1, 2)

  rz = {"geometry_type": 1}
  values_3d = np.ones((2, 2, 2, 3))
  major_r, vert_z, phi = file_geometry.geometry_components(
      values_3d, rz, "rz.gkyl")
  assert major_r.shape == vert_z.shape == phi.shape == (2, 2, 2)


@pytest.mark.parametrize(("ctx", "values", "message"), [
    ({
        "geometry_type": file_geometry._MAPC2P_IDX
    }, np.ones((2, 2, 2)), "at least 3 Cartesian"),
    ({
        "geometry_type": 1
    }, np.ones(2), "at least 2 R/Z"),
])
def test_geometry_components_reject_short_layouts(ctx, values, message):
  with pytest.raises(ValueError, match=message):
    file_geometry.geometry_components(values, ctx, "bad.gkyl")
