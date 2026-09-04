"""Smoke tests + architecture contract for the postgkyl library.

Run:  PYTHONPATH=src pytest tests/test_postgkyl.py -v
"""

import ast
import collections
import os
import sys

import numpy as np
import pytest

# Make src/ importable without an install.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)  # dedup harmless across the shared test session

import matplotlib
matplotlib.use("Agg")

import postgkyl as pg  # noqa: E402

DATA = os.path.join(ROOT, "tests", "test_data")
F1 = os.path.join(DATA, "rt_gk_tcv_iwl_adapt_source_1x2v_p1-ion_HamiltonianMoments_250.gkyl")
F2D = os.path.join(DATA, "generated", "2d_ms_p1.gkyl")
F_GKHYBRID = os.path.join(DATA, "rt_gk_tcv_iwl_1x2v_p1-elc_250.gkyl")


def test_load_metadata():
  d = pg.load(F1)
  assert d.num_dims == 1
  assert d.ctx["basis_type"] == "serendipity"
  assert d.ctx["poly_order"] == 1
  assert not d.is_interpolated          # raw modal data
# end


def test_golden_script_1d():
  g = pg.load(F1).interpolate().select(comp=0)
  assert g.is_interpolated
  assert g.num_comps == 1
  assert g.num_dims == 1
  assert g.values.shape[0] == 48         # 24 cells * (p+1=2) interpolation points
  assert type(g).__name__ == "GData"    # subclass propagated through verbs
  fig = g.plot(show=False)
  assert fig is not None
# end


def test_golden_script_2d():
  g = pg.load(F2D).interpolate().select(comp=0)
  assert g.num_dims == 2
  assert g.values.shape == (16, 16, 1)
  assert g.plot(show=False) is not None
# end


def test_plot_has_one_canonical_callable():
  from postgkyl import operations, render
  from postgkyl.gdata import verbs

  assert pg.plot is render.plot
  assert pg.plot is operations.plot
  assert pg.plot is pg.GData.plot
  assert pg.plot is pg.GDataGroup.plot
  assert pg.plot is verbs.plot
# end


def test_plotly_has_one_canonical_callable():
  from postgkyl import operations, render

  assert pg.plotly is render.plotly
  assert pg.plotly is operations.plotly
  assert pg.plotly is pg.GData.plotly
# end


def test_pyvista_has_one_canonical_callable():
  from postgkyl import operations, render

  assert pg.pyvista is render.pyvista
  assert pg.pyvista is operations.pyvista
  assert pg.pyvista is pg.GData.pyvista
# end


def test_arithmetic_and_ufunc():
  a = pg.load(F1).interpolate().select(comp=0)
  b = pg.load(F1).interpolate().select(comp=0)
  assert isinstance(a + b, pg.GData)
  assert isinstance(a * 2.0, pg.GData)
  assert isinstance(2.0 * a, pg.GData)          # reflected
  mag = np.sqrt(a ** 2 + b ** 2)                # ufunc keeps it a GData
  assert isinstance(mag, pg.GData)
  assert np.allclose(mag.values, np.sqrt(a.values ** 2 + b.values ** 2))
  assert np.asarray(a).shape == a.values.shape  # __array__
# end


def test_capability_guardrails_on_modal_data():
  """Modal data supports the Gkeyll verbs; everything NumPy-shaped refuses."""
  a = pg.load(F1)
  with pytest.raises(ValueError):
    np.sqrt(a)                                   # general ufunc: no modal meaning
  # end
  with pytest.raises(ValueError):
    np.asarray(a)                                # coefficients are not point values
  # end
  with pytest.raises(ValueError):
    a.select(comp=0)                                # slicing would mix basis functions
  # end
  with pytest.raises(ValueError):
    _ = a + a.interpolate()                           # mixed modal + field domains
  # end
# end


# --------------------------------------------------------------------------
# The modal domain: DG operations running inside Gkeyll (REFACTOR_GKEYLL_FFI.md)
# --------------------------------------------------------------------------
from postgkyl import gpython  # noqa: E402

needs_gkeyll = pytest.mark.skipif(not gpython.available(),
    reason="no compiled Gkeyll (libg0core.so) found")


@needs_gkeyll
def test_load_lands_in_the_modal_domain():
  d = pg.load(F1)
  assert d.backend == "gkyl"                     # native gkyl_array storage
  assert d.native is not None
  assert d.values.shape == (24, 6)                # read-only view for inspection
  assert not d.values.flags.writeable
  g = d.interpolate()                                 # the one-way bridge
  assert g.backend == "numpy"                    # ...to a by-value NumPy array
  assert g.values.flags.writeable
# end


@needs_gkeyll
def test_shim_handshake():
  """The compiled gpython shim pairs with this postgkyl (GKEYLL_C_SHIM.md).

  There are no struct layouts to guard anymore -- the C compiler checked the
  whole contract when gpython.c built. What remains testable at runtime is the
  version handshake plus a behavioral probe through the shim."""
  g0 = gpython.require()
  assert g0.api_version() == g0.GPYTHON_API_VERSION
  b = gpython.basis.get_basis("serendipity", 2, 1)
  assert (b.ndim, b.poly_order, b.num_basis) == (2, 1, 4)
  assert b.id == "serendipity"
# end


@needs_gkeyll
def test_gkhybrid_basis_loads_and_interpolates():
  """A real 1x2v gyrokinetic distribution file (gkhybrid basis) round-trips
  through the modal -> field bridge, exactly like a serendipity/tensor file."""
  d = pg.load(F_GKHYBRID)
  assert d.ctx["basis_type"] == "gkhybrid"
  assert d.ctx["poly_order"] == 1
  assert d.num_dims == 3                          # 1x2v
  assert d.values.shape[-1] == 12                 # gkhybrid 1x2v num_basis
  g = d.interpolate()
  assert g.backend == "numpy"
  assert g.values.shape == (64, 32, 16, 1)        # (p+1=2) interpolation points/cell
# end


@needs_gkeyll
def test_interpolation_matrix_matches_analytic_basis():
  """Matrices built from Gkeyll's eval() match the normalized Legendre basis."""
  m = gpython.basis.interpolation_matrix("serendipity", 1, 1, 2)   # points z = -+1/2
  expect = np.array([[1 / np.sqrt(2), -np.sqrt(3.0 / 2.0) / 2],
                     [1 / np.sqrt(2), +np.sqrt(3.0 / 2.0) / 2]])
  assert np.allclose(m, expect)
  m2 = gpython.basis.interpolation_matrix("serendipity", 1, 2, 3)  # p2, points -+2/3, 0
  z = np.array([-2.0 / 3.0, 0.0, 2.0 / 3.0])
  assert np.allclose(m2[:, 2], 2.371708245126285 * z ** 2 - 0.7905694150420951)
# end


@needs_gkeyll
def test_weak_algebra_identities():
  """div(mul(a, b), b) == a -- Gkeyll's weak kernels are exact inverses."""
  a, b = pg.load(F1), pg.load(F1)
  back = (a * b / b).interpolate().values
  ref = a.interpolate().values
  for f in (0, 2):  # density and T; field 1 (u_par) is identically ~0 -> 0/0
    scale = np.abs(ref[..., f]).max()
    assert np.abs(back[..., f] - ref[..., f]).max() / scale < 1e-12
  # end
# end


@needs_gkeyll
def test_modal_linear_ops_commute_with_interpolate():
  """interpolate is linear: modal +,-,scalar* agree with their NumPy counterparts."""
  a, b = pg.load(F1), pg.load(F1)
  assert np.allclose((a + b).interpolate().values, a.interpolate().values + b.interpolate().values)
  assert np.allclose((a - b).interpolate().values, 0.0)
  assert np.allclose((2.5 * a).interpolate().values, 2.5 * a.interpolate().values)
  assert np.allclose((-a).interpolate().values, -(a.interpolate().values))
  assert np.allclose((a ** 2).interpolate().values, (a * a).interpolate().values)
  shifted = (a + 1.0e18).interpolate().values - a.interpolate().values
  assert np.allclose(shifted, 1.0e18, rtol=1e-6)
# end


def _make_modal(grid, cells, basis_type, poly_order, coeffs):
  """A bare modal GData, built in-memory rather than from a file -- for
  exercising the conf x phase cross-multiply path with grids we control."""
  d = pg.GData()
  d.ctx.update(basis_type=basis_type, poly_order=poly_order, value_form="modal",
      cells=np.array(cells))
  d.push(grid, gpython.array.GkylArray.from_numpy(coeffs))
  return d
# end


@needs_gkeyll
def test_conf_phase_mul_is_automatic_and_commutative():
  """``conf * phase`` and ``phase * conf`` both dispatch to the cross-basis
  gkyl_dg_mul_conf_phase_op_range path with no separate method needed -- the
  API picks the lower-dimensional operand as the conf side automatically.
  Multiplying by a spatially-uniform conf field of true value 1 is an exact
  identity on the phase side (no weak-projection truncation), so this is a
  correctness check, not just a "did it run" smoke test."""
  conf_edges = [np.linspace(0.0, 1.0, 4)]                       # 3 cells
  phase_edges = [np.linspace(0.0, 1.0, 4), np.linspace(-1.0, 1.0, 5)]  # 3x4

  cbasis = gpython.basis.get_basis("serendipity", 1, 1)
  pbasis = gpython.basis.get_basis("hybrid", 2, 1)
  cop = np.zeros((3, cbasis.num_basis))
  cop[:, 0] = np.sqrt(2.0)                                      # value 1
  rng = np.random.default_rng(11)
  pop = rng.normal(size=(12, pbasis.num_basis))

  conf = _make_modal(conf_edges, [3], "serendipity", 1, cop)
  phase = _make_modal(phase_edges, [3, 4], "hybrid", 1, pop)

  out1 = conf * phase
  out2 = phase * conf
  assert isinstance(out1, pg.GData) and out1.num_dims == 2
  np.testing.assert_allclose(out1.values.reshape(12, 6), pop)
  np.testing.assert_allclose(out2.values.reshape(12, 6), pop)
# end


@needs_gkeyll
def test_conf_phase_mul_rejects_non_mul_ops_and_mismatched_grids():
  conf = _make_modal([np.linspace(0.0, 1.0, 4)], [3], "serendipity", 1,
      np.zeros((3, 2)))
  phase = _make_modal(
      [np.linspace(0.0, 1.0, 4), np.linspace(-1.0, 1.0, 5)], [3, 4],
      "hybrid", 1, np.zeros((12, 6)))
  with pytest.raises(ValueError, match="only '\\*' is defined"):
    conf / phase
  # end
  with pytest.raises(ValueError, match="only '\\*' is defined"):
    conf + phase
  # end

  mismatched = _make_modal(
      [np.linspace(0.0, 2.0, 4), np.linspace(-1.0, 1.0, 5)], [3, 4],
      "hybrid", 1, np.zeros((12, 6)))
  with pytest.raises(ValueError, match="not the same simulation"):
    conf * mismatched
  # end
# end


def _relerr(x, y):
  x, y = np.asarray(x, float), np.asarray(y, float)
  return np.abs(x - y).max() / np.abs(y).max()
# end


@needs_gkeyll
def test_value_form_round_trips():
  """modal <-> nodal is exact; modal <-> quad is exact for num_quad >= p+1."""
  a = pg.load(F1)
  n = a.to_nodal()
  assert n.ctx["value_form"] == "nodal"
  assert n.backend == "gkyl"                     # never leaves the native domain
  assert _relerr(n.to_modal().values, a.values) < 1e-14
  q = a.to_quad()
  assert (q.ctx["value_form"], q.ctx["num_quad"]) == ("quad", 2)
  assert _relerr(q.to_modal().values, a.values) < 1e-14
  # nodal -> quad composes through modal
  assert _relerr(n.to_quad().to_modal().values, a.values) < 1e-14
  # nodal values are the field evaluated at the basis node_list points
  m2n = gpython.basis.modal_to_nodal_matrix("serendipity", 1, 1)
  manual = np.einsum("pk,cfk->cfp", m2n,
      np.asarray(a.values).reshape(24, 3, 2)).reshape(24, 6)
  assert np.allclose(n.values, manual)
# end


@needs_gkeyll
def test_apply_pointwise_via_quadrature():
  """.apply(fn): modal -> quad -> fn -> modal, exact where quadrature is."""
  a = pg.load(F1)
  assert _relerr(a.apply(lambda v: v).values, a.values) < 1e-13
  # p=1: p+1 Gauss points integrate the square exactly -> matches the weak kernel
  assert _relerr(a.apply(np.square).values, (a * a).values) < 1e-13
  chained = a.apply(np.abs).apply(np.sqrt)       # stays modal + gkyl-native
  assert chained.backend == "gkyl"
  assert chained.ctx.get("value_form", "modal") == "modal"
  with pytest.raises(ValueError):
    a.apply(lambda v: v.sum(axis=-1))            # fn must act pointwise
  # end
# end


@needs_gkeyll
def test_conversions_are_always_explicit():
  """No implicit value_form change, ever (REFACTOR_GKEYLL_FFI.md §3b)."""
  a = pg.load(F1)
  n, q = a.to_nodal(), a.to_quad()
  with pytest.raises(ValueError):
    _ = a + n                                    # mixed value_forms
  # end
  with pytest.raises(ValueError):
    _ = np.add(n, q)                             # mixed reps through a ufunc
  # end
  with pytest.raises(ValueError):
    q.interpolate()                                   # interp needs modal
  # end
  assert n.integrate() is not None               # point values integrate in-place in form
  with pytest.raises(ValueError):
    np.sqrt(a)                                   # ufuncs have no modal meaning
  # end
  with pytest.raises(ValueError):
    np.asarray(a)                                # coefficients are not values
  # end
  with pytest.raises(ValueError):
    a.plot(show=False)                           # coefficients are not plottable
  # end
# end


@needs_gkeyll
def test_pointwise_numpy_on_point_values():
  """NumPy math is exact on nodal/quad data and stays native, in-value_form."""
  a = pg.load(F1)
  n, q = a.to_nodal(), a.to_quad()
  s = np.sqrt(np.abs(n))                         # ufunc on nodal
  assert (s.backend, s.ctx["value_form"]) == ("gkyl", "nodal")
  assert np.allclose(s.values, np.sqrt(np.abs(np.asarray(n.values))))
  assert np.allclose((n ** 2).values, np.asarray(n.values) ** 2)
  assert np.allclose((q * q).values, np.asarray(q.values) ** 2)
  # pointwise-at-quad then one projection == the weak kernel (p1 exactness)
  assert _relerr((q * q).to_modal().values, (a * a).values) < 1e-13
  # chain at the points, project once -- identical to the one-shot .apply()
  fn = lambda v: np.sqrt(np.abs(v))
  assert _relerr(np.sqrt(np.abs(q)).to_modal().values, a.apply(fn).values) < 1e-15
  assert np.asarray(n).shape == (24, 6)          # __array__ allowed on points
# end


@needs_gkeyll
def test_plot_point_values_directly():
  """Nodal/quad datasets plot at their true point locations."""
  a = pg.load(F1)
  assert a.to_nodal().plot(show=False) is not None
  assert a.to_quad().plot(show=False) is not None
  b = pg.load(F2D)
  assert b.to_quad().plot(show=False) is not None
  assert b.to_nodal().plot(show=False) is not None   # p1 corners: tensor set
  p2 = pg.load(os.path.join(DATA, "generated", "2d_ms_p2.gkyl"))
  with pytest.raises(ValueError):
    p2.to_nodal().plot(show=False)               # non-tensor node set -> to_quad
  # end
# end


@needs_gkeyll
def test_linear_ops_valid_in_any_value_form():
  """+ - and scalar ops act pointwise in nodal/quad and agree with modal."""
  a = pg.load(F1)
  n = a.to_nodal()
  assert _relerr((2 * n - n + n).to_modal().values, (2 * a).values) < 1e-13
  assert _relerr((n + 5.0e17).to_modal().values, (a + 5.0e17).values) < 1e-13
# end


@needs_gkeyll
def test_values_view_pins_native_memory():
  """Regression: `dataset.values` on a temporary must stay valid after GC."""
  import gc
  a = pg.load(F1)
  expected = a.values.copy()
  v = pg.load(F1).values                         # dataset is garbage immediately
  got = (2 * pg.load(F1).to_nodal()).to_modal().values  # temporaries galore
  gc.collect()
  assert np.array_equal(v, expected)
  assert _relerr(got, 2 * expected) < 1e-13
# end


@needs_gkeyll
def test_integrate_via_gkeyll():
  """pg-level integrate == the coefficient-space formula (exact for DG)."""
  a = pg.load(F1)
  result = a.integrate()
  v = a.values                                   # (cells, nfields*num_basis) view
  dx = float((a.bounds[1][0] - a.bounds[0][0]) / a.num_cells[0])
  nb = 2                                         # serendipity 1D p1
  manual = np.array([v[:, f * nb].sum() * dx / np.sqrt(2.0)
                     for f in range(v.shape[-1] // nb)])
  assert np.allclose(result, manual)
  assert np.all(a.integrate(op="abs") >= np.abs(result) * (1 - 1e-12))
  point_result = a.interpolate().integrate()
  assert np.allclose(point_result, result)
# end


def test_write_roundtrip(tmp_path):
  a = pg.load(F1).interpolate().select(comp=0)
  out = a.save(str(tmp_path / "rt.gkyl"))
  back = pg.load(out)
  assert np.allclose(back.values, a.values)
# end


def test_info_returns_string(capsys):
  s = pg.load(F1).info()
  assert "Number of components" in s
# end


def test_cli_chained(tmp_path):
  """The chained CLI: bare filename -> load, interp, sel, plot --saveas."""
  from click.testing import CliRunner
  from postgkyl.cli.app import cli

  out = tmp_path / "cli.png"
  result = CliRunner().invoke(cli, [
      F1, "interp", "sel", "--comp", "0", "plot", "--show", "False",
      "--saveas", str(out)])
  assert result.exit_code == 0, result.output
  assert out.exists()
# end


def test_cli_abbreviation_and_info():
  """`interp`/`sel` resolve by unique-prefix abbreviation."""
  from click.testing import CliRunner
  from postgkyl.cli.app import cli

  result = CliRunner().invoke(cli, [F1, "interp", "sel", "--comp", "0", "info"])
  assert result.exit_code == 0, result.output
  assert "interpolated" in result.output
# end


# --------------------------------------------------------------------------
# Architecture contract: the layering is a strict, cycle-free DAG.
# --------------------------------------------------------------------------
_ALLOWED = {
    "cli_spec": set(),                                # frozen CLI metadata; dependency-free leaf
    "gpython":    set(),                                # the foreign floor (only ctypes owner)
    "numerics": set(),
    "dg":     {"gpython"},                              # interpolation bridge + modal ops -> kernels
    "io":     {"gpython", "numerics", "cli_spec"}, # C-native reader -> gkyl_array_rio;
                                                      # writer reuses the pure-math leaf
                                                      # (nodal_to_cell_centered_grid for the
                                                      # vtk writer) instead of duplicating
                                                      # it -- numerics has 0 internal imports,
                                                      # so this cannot create a cycle (layer 04-io)
    "gdatastate": {"io", "gpython", "dg"},              # state plus the shared native
                                                      # point-value materialization bridge
    "render": {"gdatastate", "numerics", "cli_spec"},
    "operations": {"gdatastate", "dg", "io", "numerics", "render", "cli_spec"}, # data transformations:
                                                      # the physics verbs (moments/agyro/
                                                      # current/energetics/rotate/
                                                      # transform_frame/laguerre) moved up
                                                      # into diagnostics, folded with the
                                                      # models/ array math they delegated to;
                                                      # flat modules are equation-blind core
                                                      # verbs; domain subpackages (currently
                                                      # gyrokinetics) own transformations that
                                                      # need domain geometry without deriving
                                                      # a physical conclusion
    "diagnostics": {"gdatastate", "operations", "numerics", "gdata", "render", "io", "cli_spec"}, # added by
                                                      # 10-diagnostics.md: equation-
                                                      # specific compositions grouped under
                                                      # gk/vm/pkpm/mom;
                                                      # their modules wrap core
                                                      # verbs and state -- none of gdatastate/operations/
                                                      # numerics imports upward, so this
                                                      # cannot create a cycle; "gdata" added by
                                                      # 12-diagnostics-loaders.md: the
                                                      # gk/pkpm loaders build on
                                                      # pg.load/GData (modal arithmetic,
                                                      # .interpolate()) to read simulation output
                                                      # -- gdata imports only gdatastate/operations/io, none
                                                      # of which import diagnostics, so this
                                                      # still cannot create a cycle; "render"
                                                      # pre-authorized by 13-diagnostics-
                                                      # programs.md for future program-scale
                                                      # diagnostics that may want render's
                                                      # generic plot() -- as of this layer's
                                                      # landing, none of the six program
                                                      # modules (energy_balance, particle_
                                                      # balance, nodes, trajectory, enstrophy,
                                                      # ke_dke) actually import it, each
                                                      # building its own bespoke figure
                                                      # directly with matplotlib instead;
                                                      # render imports only gdatastate/numerics,
                                                      # neither of which imports diagnostics,
                                                      # so this cannot create a cycle whether
                                                      # or not the edge is ever exercised
    "gdata":  {"gdatastate", "operations", "io", "cli_spec"},
    "":       {"gdata", "operations", "render", "io", "gdatastate",
               "diagnostics", "gpython", "_version", "cli_spec"}, # facade:
                                                      # pure re-export of public names;
                                                      # "gdatastate" is group_blocks, the
                                                      # multiblock-family partition, which
                                                      # lives beside flatten_datasets in the
                                                      # container layer that owns collections
                                                      # of datasets;
                                                      # "diagnostics" added by
                                                      # 12-diagnostics-loaders.md, which
                                                      # explicitly authorizes the facade
                                                      # re-exporting the gk namespace for
                                                      # pg.gk.load_quantity(...);
                                                      # "_version" is __init__.py's own import
                                                      # of _version.py's version_report (`pgkyl
                                                      # --version`'s commit/build-info report),
                                                      # re-exported like any other facade name;
                                                      # "gpython" is _version.py's own edge (it
                                                      # reads gpython.available()/build_info())
                                                      # -- both source files sit in the same ""
                                                      # layer, so both edges are checked here
    "cli":    {"", "cli_spec"},                       # top surface: facade + frozen metadata
}
_LAYERS = set(_ALLOWED)


def _layer(path, pkg_root):
  parts = os.path.relpath(path, pkg_root).split(os.sep)
  if len(parts) > 1:
    return parts[0]
  # end
  module = os.path.splitext(parts[0])[0]
  return module if module in _LAYERS else ""
# end


def _import_targets(node):
  if isinstance(node, ast.Import):
    for n in node.names:
      if n.name == "postgkyl" or n.name.startswith("postgkyl."):
        t = n.name.split(".")
        yield t[1] if len(t) > 1 else ""
      # end
    # end
  # end
  elif isinstance(node, ast.ImportFrom):
    if node.level:
      return
    # end
    mod = node.module or ""
    if mod == "postgkyl":
      for n in node.names:
        yield n.name if n.name in _LAYERS else ""
      # end
    # end
    elif mod.startswith("postgkyl."):
      yield mod.split(".")[1]
    # end
  # end
# end


def _build_edges(pkg_root=None):
  pkg_root = pkg_root or os.path.join(SRC, "postgkyl")
  edges = collections.defaultdict(set)
  violations = []
  for dp, _, files in os.walk(pkg_root):
    for f in files:
      if not f.endswith(".py"):
        continue
      # end
      p = os.path.join(dp, f)
      src = _layer(p, pkg_root)
      for node in ast.walk(ast.parse(open(p, encoding="utf-8").read(), p)):
        for tgt in _import_targets(node):
          if tgt == src:
            continue
          # end
          edges[src].add(tgt)
          if tgt not in _ALLOWED.get(src, set()):
            violations.append(f"{os.path.relpath(p, pkg_root)} [{src or 'facade'}] -> [{tgt or 'facade'}]")
          # end
        # end
      # end
    # end
  # end
  return edges, violations
# end


def test_facade_is_pure_reexport():
  """__init__.py must define no functions/classes -- only re-export names."""
  facade = os.path.join(SRC, "postgkyl", "__init__.py")
  tree = ast.parse(open(facade).read(), facade)
  defs = [n.name for n in tree.body
          if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
  assert not defs, f"facade should be pure re-export, but defines: {defs}"
# end


def test_import_contract_no_violations():
  _, violations = _build_edges()
  assert not violations, "layer contract violations:\n" + "\n".join(violations)
# end


def _foreign_floor_offenders(pkg_root):
  offenders = []
  for dp, _, files in os.walk(pkg_root):
    for f in files:
      if not f.endswith(".py"):
        continue
      # end
      p = os.path.join(dp, f)
      in_gpython = _layer(p, pkg_root) == "gpython"
      for node in ast.walk(ast.parse(open(p, encoding="utf-8").read(), p)):
        names = []
        if isinstance(node, ast.Import):
          names = [n.name for n in node.names]
        # end
        elif isinstance(node, ast.ImportFrom):
          names = [node.module or ""] + (
              [n.name for n in node.names] if node.level or "." in (node.module or "")
              or (node.module or "") == "postgkyl" else [])
        # end
        for name in names:
          root = name.split(".")[0]
          if root == "ctypes":
            offenders.append(f"{os.path.relpath(p, pkg_root)}: ctypes")
          # end
          if ("_gpython" in name.split(".") or name == "_gpython") and not in_gpython:
            offenders.append(f"{os.path.relpath(p, pkg_root)}: _gpython")
          # end
        # end
      # end
    # end
  # end
  return offenders
# end


def test_foreign_floor_confined_to_gpython():
  """The foreign world is the compiled ``_gpython`` extension, importable only
  under gpython/ -- and ctypes appears nowhere at all: the C contract is enforced
  by the compiler when the gpython shim builds, never re-declared in Python
  (GKEYLL_C_SHIM.md)."""
  pkg_root = os.path.join(SRC, "postgkyl")
  offenders = _foreign_floor_offenders(pkg_root)
  assert not offenders, f"foreign floor leaked above gpython/: {offenders}"
# end


def _find_cycles(edges):
  color = collections.defaultdict(int)
  cycles = []

  def dfs(u, stack):
    color[u] = 1
    for w in edges.get(u, ()):
      if color[w] == 1:
        cycles.append(stack + [w])
      # end
      elif color[w] == 0:
        dfs(w, stack + [w])
      # end
    # end
    color[u] = 2
  # end

  for n in list(edges):
    if color[n] == 0:
      dfs(n, [n])
    # end
  # end
  return cycles
# end


def test_import_graph_is_acyclic():
  edges, _ = _build_edges()
  cycles = _find_cycles(edges)
  assert not cycles, f"import cycle(s): {cycles}"
# end


# --------------------------------------------------------------------------
# The self-checks above only ever see a *compliant* tree in this repo (that
# is the point). These drive their violation/cycle/offender branches
# directly, against a small throwaway fake package tree, without touching
# the real source.
# --------------------------------------------------------------------------
def _write_module(pkg_root, layer, name, body):
  d = os.path.join(pkg_root, layer) if layer else pkg_root
  os.makedirs(d, exist_ok=True)
  with open(os.path.join(d, name), "w") as fh:
    fh.write(body)
  # end
# end


def test_build_edges_flags_a_disallowed_import(tmp_path):
  pkg_root = str(tmp_path / "postgkyl")
  _write_module(pkg_root, "badlayer", "mod.py", "import postgkyl.operations\n")
  _, violations = _build_edges(pkg_root)
  assert any("badlayer" in v and "operations" in v for v in violations)
# end


def test_build_edges_classifies_a_flat_leaf_module(tmp_path):
  pkg_root = str(tmp_path / "postgkyl")
  _write_module(pkg_root, "", "cli_spec.py", "import postgkyl.operations\n")
  _, violations = _build_edges(pkg_root)
  assert any("cli_spec.py [cli_spec] -> [operations]" in v for v in violations)
# end


def test_import_graph_detects_a_real_cycle(tmp_path):
  pkg_root = str(tmp_path / "postgkyl")
  _write_module(pkg_root, "layer_a", "mod.py", "import postgkyl.layer_b\n")
  _write_module(pkg_root, "layer_b", "mod.py", "import postgkyl.layer_a\n")
  edges, _ = _build_edges(pkg_root)
  cycles = _find_cycles(edges)
  assert cycles, "expected the fake layer_a <-> layer_b cycle to be detected"
# end


def test_foreign_floor_offenders_flags_ctypes_and_gpython_outside_gpython(tmp_path):
  pkg_root = str(tmp_path / "postgkyl")
  _write_module(pkg_root, "badlayer", "uses_ctypes.py", "import ctypes\n")
  _write_module(pkg_root, "badlayer", "uses_gpython.py", "from postgkyl.gpython import _gpython\n")
  offenders = _foreign_floor_offenders(pkg_root)
  assert any(o.endswith(": ctypes") for o in offenders)
  assert any(o.endswith(": _gpython") for o in offenders)
# end
