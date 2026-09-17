"""Linear tensor-grid resampling with linear extrapolation outside the grid."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def resample_grid(values: np.ndarray, src_coords: list[np.ndarray],
                  dst_coords: list[np.ndarray]) -> np.ndarray:
  """Linearly resample ``values`` between tensor-product coordinate grids."""
  mesh = np.meshgrid(*dst_coords, indexing="ij")
  return RegularGridInterpolator(tuple(src_coords),
                                 values,
                                 bounds_error=False,
                                 fill_value=None)(tuple(mesh))
