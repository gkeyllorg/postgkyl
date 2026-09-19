"""Rotation matrix aligning the x-axis with a given vector."""

from __future__ import annotations

import numpy as np


def rotation_matrix(vector: np.ndarray) -> np.ndarray:
  """Calculate a 3x3 rotation matrix whose first row is ``vector``'s direction.

  Args:
    vector: A 3-component vector (nonzero in every component).

  Returns:
    3x3 rotation matrix (NumPy array).
  """
  rot = np.zeros((3, 3))
  norm = np.abs(vector)
  k = vector / norm  # direction unit vector

  # normalization
  norm2 = np.sqrt(k[1] * k[1] + k[2] * k[2])
  norm3 = np.sqrt((k[1] * k[1] + k[2] * k[2])**2 + k[0] * k[0] * k[1] * k[1] +
                  k[0] * k[0] * k[2] * k[2])

  rot[0, :] = k
  rot[1, 0] = 0
  rot[1, 1] = -k[2] / norm2
  rot[1, 2] = k[1] / norm2
  rot[2, 0] = (k[1] * k[1] + k[2] * k[2]) / norm3
  rot[2, 1] = -k[0] * k[1] / norm3
  rot[2, 2] = -k[0] * k[2] / norm3

  return rot
