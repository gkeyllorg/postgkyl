"""Generated fixtures explain their physical meaning and construction on load."""

import numpy as np
import pytest

import postgkyl as pg
from postgkyl import gpython, io
from generate_test_data import generate_all


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
  directory = tmp_path_factory.mktemp("described_data")
  generate_all(directory)
  return directory


@pytest.fixture(params=[io.GkylReader, io.GkylCReader])
def reader(request, monkeypatch):
  if request.param is io.GkylCReader and not gpython.available():
    pytest.skip("compiled Gkeyll unavailable")
  # Native field reading falls back to Python for dynvectors, as in production.
  monkeypatch.setattr(io, "_READERS", {
      "test": request.param,
      "fallback": io.GkylReader
  })


def test_every_generated_file_explains_its_data(reader, generated):
  paths = sorted(generated.glob("*.gkyl"))
  assert paths
  for path in paths:
    data = pg.load(str(path))
    info = data.info(all=True)
    for key in ("description", "analytic_function", "generation_method"):
      assert isinstance(data.ctx.get(key), str), (path.name, key)
      assert data.ctx[key].strip(), (path.name, key)
      assert f"{key}:" in info, (path.name, key)


def test_described_dynvector_preserves_times_and_values(generated):
  data = pg.load(str(generated / "exponential_energy.gkyl"))
  times = np.linspace(0.0, 10.0, 101)
  np.testing.assert_allclose(data.get_grid()[0], times)
  np.testing.assert_allclose(data.get_values()[:, 0],
                             1e-6 * np.exp(0.4 * times))
  assert data.ctx["analytic_function"] == "E(t)=1e-6*exp(0.4*t)."
