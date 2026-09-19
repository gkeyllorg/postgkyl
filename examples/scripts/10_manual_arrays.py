"""Copy grids and values, manipulate the arrays, and wrap a field for plotting."""

import numpy as np
import postgkyl as pg
from _example_paths import TEST_DATA, prepare_output_dir

OUTPUT_DIR = prepare_output_dir()
modal = pg.load(TEST_DATA / "generated/wave_surface_000.gkyl")
# On modal data, these numbers are coefficients, not physical field samples.
coefficients = modal.values.copy()
assert coefficients.shape[-1] == 1  # this generated example uses p0

field = modal.interpolate()
grid = [axis.copy() for axis in field.grid]
values = field.values.copy()
np.testing.assert_array_equal(np.asarray(field), values)

# This interpolated mesh has edges. Pair its cell centers with field values;
# for data already located at nodes, use the grid coordinates directly.
coordinates = [
    0.5 * (axis[1:] + axis[:-1]) if len(axis) == values.shape[dim] + 1 else axis
    for dim, axis in enumerate(grid)
]
x, y = np.meshgrid(*coordinates, indexing="ij")
assert x.shape == y.shape == values.shape[:-1]

# NumPy edits to these copies do not change the original dataset.
manual_values = (values - 1.0) / 0.6
np.testing.assert_allclose(manual_values[..., 0],
                           np.cos(x) * np.cos(y),
                           atol=1e-14)
np.testing.assert_array_equal(field.values, values)

# push accepts coordinates and values. Preserve metadata explicitly when
# constructing a new point-value dataset after a manual calculation.
manual = field.clone()
manual.push(grid, manual_values)
manual.plot(title="Normalized density perturbation",
            xlabel="x",
            ylabel="y",
            clabel="(n - 1) / 0.6",
            no_show=True,
            saveas=OUTPUT_DIR / "10_manual.png",
            dpi=100)
