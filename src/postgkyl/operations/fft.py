"""The ``fft`` verb -- Fourier transform / power spectral density."""

from __future__ import annotations

from typing import TYPE_CHECKING

from postgkyl import numerics

if TYPE_CHECKING:
  from postgkyl.gdatastate.gdatastate import GDataState


def fft(data: "GDataState",
        *,
        psd: bool = False,
        iso: bool = False,
        inplace: bool = False,
        tag: str | None = None,
        label: str | None = None):
  """Fourier transform (or power spectral density) of field-domain data.

  Wraps ``numerics.fft``: each component is transformed over the spatial
  axes (dummy axes of length <= 2 are squeezed out first). Supports 1D, 2D,
  and 3D data. ``numerics.fft`` reads its sample spacing straight off the
  grid array's own length, so a nodal (edge) grid -- one entry longer than
  the value count, the usual post-``.interpolate()`` shape -- is first collapsed
  to cell centers (matching values); a grid that already matches (e.g. a
  dynvector's) is passed through unchanged.

  Args:
    data: the dataset to transform; must be NumPy-backed (call ``.interpolate()``
      first on native modal data).
    psd: when True, return the power spectral density ``|FT|^2`` over the
      positive frequencies only.
    iso: when True (only meaningful for 2D/3D data with ``psd=True``), bin
      the PSD into a 1D isotropic spectrum over the polar wavenumber
      magnitude.
    inplace: mutate and return ``data`` instead of a new dataset.
    tag: optional tag for the returned dataset.
    label: optional label for the returned dataset.

  Returns:
    A dataset whose grid is the frequency/wavenumber axis (axes) and whose
    values are the transform, PSD, or isotropic spectrum.

  Raises:
    ValueError: if ``data`` is native modal (gkyl-backed), or if isotropic
      binning is requested for data that is not 2D/3D.
  """
  if data.backend == "gkyl":
    raise ValueError(
        "fft operates on interpolated (NumPy) values; call .interpolate() first "
        "-- Fourier transforming raw DG coefficients would mix basis functions."
    )
  grid, values = data.grid, data.values
  num_cells = values.shape[:-1]
  if any(grid[d].shape[0] == num_cells[d] + 1 for d in range(len(grid))):
    grid = numerics.nodal_to_cell_centered_grid(grid, num_cells)
  freq, ft_values = numerics.fft(grid, values, psd=psd, iso=iso)
  return data._result(freq, ft_values, inplace=inplace, tag=tag, label=label)
