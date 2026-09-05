"""Postgkyl module for testing the gk-load-quantity command.

This exercises ``gk_load_quantity`` against *every* quantity registered in
``gk_quant_registry``. Because the on-disk .gkyl writer does not persist the DG
metadata (poly_order, basis_type, mass) needed by the fetch functions, the test
does not rely on real simulation output. Instead it

  1. creates empty marker files with the exact names each quantity's sources are
     discovered by (so ``choose_source`` finds a valid source combination), and
  2. monkeypatches the ``GData`` constructor used inside ``gkquantity`` so that
     loading a source returns a small, self-consistent synthetic DG dataset.

The fetch functions (and their gkylsoft-backed DG operators) then run for real.
Quantities whose computation requires the compiled gkylsoft library are skipped
when that library is unavailable.
"""
import os

import click
import numpy as np
import pytest

import postgkyl.commands as cmd
import postgkyl.utils.gk_quantities.gkquantity as gkquantity
from postgkyl.data import GData
from postgkyl.pgkyl import cli
from postgkyl.utils.gk_quantities.registry import gk_quant_registry

# Synthetic DG dataset parameters: 1D, p1 serendipity (num_basis = 2), six
# physical components so that fetch functions selecting up to component 5 work
# (the metric g_ij is the widest source, with g_11,g_12,g_13,g_22,g_23,g_33).
_POLY_ORDER = 1
_BASIS_TYPE = "serendipity"
_NUM_BASIS = 2
_NUM_PHYS_COMPS = 6
_NUM_CELLS = 4

# Probe whether the gkylsoft DG-operator library is available. Quantities whose
# fetch functions use it (e.g. press, beta, ExB_vel) are skipped if it is not.
try:
  from postgkyl.tools.gkeyll_dg_ops import GkeyllDGops
  GkeyllDGops()
  _DGOPS_AVAILABLE = True
except Exception:  # noqa: BLE001 - any failure means the lib is unusable here
  _DGOPS_AVAILABLE = False

# Extra command options required by specific quantities (beyond the per-component
# selection that every vector quantity needs, which is added automatically below).
_EXTRA_OPTS = {}

# Species names used in the test. Multi-species quantities (e.g. the sound speed)
# combine an electron species with one or more ion species, so they are requested
# with the whole list; the electron species is identified by its negative charge.
_ELC_SPECIES = "elc"
_ION_SPECIES = "ion"

# Value of the 0th (constant) modal basis function in 1D: the cell average of a
# DG field is its 0th coefficient times _PSI0.
_PSI0 = 2.0**-0.5

# Define the data used to test the handling of GK distribution functions.
_TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
_DISTF_REAL = {
  "name": "rt_gk_tcv_iwl_1x2v_p1",
  "species": "elc",
  "frame": 250,
}


def _extra_opts_for(quant) -> str | None:
  """Build the '--extra' string a quantity needs to be fetched in the test."""
  opts = []
  if quant.is_vector:
    opts.append("dir=0")  # Vector quantities need a component selection.
  if quant.name in _EXTRA_OPTS:
    opts.append(_EXTRA_OPTS[quant.name])
  return ",".join(opts) if opts else None


def _make_synthetic_gdata(*args, **kwargs) -> GData:
  """Return a small, self-consistent constant-valued DG dataset.

  Each physical component is a positive constant (only the cell-average modal
  coefficient is nonzero), which keeps the DG multiply/invert operations
  well-defined. Every source is served the same synthetic values, which is
  enough to drive the fetch functions; only the charge is read back out of the
  file name, so that multi-species quantities (which tell electrons from ions
  by the sign of the charge) see a genuine electron species.
  """
  values = np.zeros((_NUM_CELLS, _NUM_BASIS * _NUM_PHYS_COMPS))
  for comp in range(_NUM_PHYS_COMPS):
    # Distinct positive cell-average per component (1/sqrt(2) is the value of
    # the 0th modal serendipity basis function).
    values[:, comp * _NUM_BASIS] = (comp + 2) * np.sqrt(2.0)

  file_name = str(args[0]) if args else ""
  charge = -1.0 if f"-{_ELC_SPECIES}_" in file_name else 1.0

  grid = [np.linspace(0.0, 1.0, _NUM_CELLS + 1)]
  gdata = GData(ctx={"poly_order": _POLY_ORDER, "basis_type": _BASIS_TYPE,
                     "mass": 1.0, "charge": charge})
  gdata.push(grid, values)
  return gdata


def _make_synthetic_gdata_no_attrs(*args, **kwargs) -> GData:
  """Synthetic data whose files carry no mass/charge attributes.

  Used to drive the '--extra' fallback, which only kicks in when the attribute
  is absent from the file context.
  """
  gdata = _make_synthetic_gdata(*args, **kwargs)
  gdata.ctx.pop("mass", None)
  gdata.ctx.pop("charge", None)
  return gdata


def _collect_source_files(quant, path: str, name: str, species: str, frame: int) -> set[str]:
  """Recursively collect the file names every source combination would look for."""
  files: set[str] = set()
  for combo in quant.source:
    for src in combo:
      if isinstance(src, str):
        files.add(quant._src_file_name(path, name, species, src, frame))
      else:
        files |= _collect_source_files(src, path, name, species, frame)
  return files


class TestGkLoadQuantity:
  """Test that gk-load-quantity can load every registered quantity."""

  name = "gktest"
  species = "ion"
  frame = 0

  def _make_ctx(self):
    ctx = click.core.Context(cli)
    ctx.obj = {"data": cmd.DataSpace(), "verbose": False}
    return ctx

  @pytest.mark.parametrize("quantity", gk_quant_registry.list())
  def test_load_quantity(self, quantity, tmp_path, monkeypatch):
    if quantity == "distf":
      self._check_distf_real()
      return

    quant = gk_quant_registry.get(quantity)
    path = str(tmp_path)

    # A multi-species quantity needs an electron and an ion species to combine.
    species = f"{_ELC_SPECIES},{_ION_SPECIES}" if quant.is_multi_species else self.species

    # Create empty marker files for every source so source discovery succeeds.
    for species_name in species.split(","):
      for file_name in _collect_source_files(quant, path, self.name, species_name, self.frame):
        open(file_name, "w").close()

    # Serve synthetic DG data whenever a source file is "loaded".
    monkeypatch.setattr(gkquantity, "GData", _make_synthetic_gdata)

    ctx = self._make_ctx()
    try:
      ctx.invoke(
        cmd.gk_load_quantity,
        quantity=quantity,
        name=self.name,
        species=species,
        frame=str(self.frame),
        path=path,
        extra=_extra_opts_for(quant),
      )
    except (RuntimeError, FileNotFoundError, OSError) as err:
      if not _DGOPS_AVAILABLE:
        pytest.skip(f"'{quantity}' requires the gkylsoft DG library: {err}")
      raise

    assert ctx.obj["data"].get_num_datasets() >= 1, (
      f"gk-load-quantity produced no dataset for quantity '{quantity}'")

  def _load(self, ctx, quantity, path, species, extra=None):
    """Invoke gk-load-quantity for a quantity, skipping if the DG lib is absent."""
    quant = gk_quant_registry.get(quantity)
    for species_name in species.split(","):
      for file_name in _collect_source_files(quant, path, self.name, species_name, self.frame):
        open(file_name, "w").close()

    try:
      ctx.invoke(
        cmd.gk_load_quantity,
        quantity=quantity,
        name=self.name,
        species=species,
        frame=str(self.frame),
        path=path,
        extra=extra,
      )
    except (RuntimeError, FileNotFoundError, OSError) as err:
      if not _DGOPS_AVAILABLE:
        pytest.skip(f"'{quantity}' requires the gkylsoft DG library: {err}")
      raise

  @pytest.mark.parametrize("quantity", ["B_tot", "B_tot_dual"])
  def test_the_total_field_falls_back_to_the_equilibrium_without_apar(
      self, quantity, tmp_path, monkeypatch):
    """An electrostatic run has no apar file, so the second source combination
    must be picked and the total field must come back as the equilibrium one.

    Only the geo files are laid down here, so the fallback is the only
    combination that can resolve; it carries no frame, which also exercises the
    frame-less path through the nested geo sources.
    """
    if not _DGOPS_AVAILABLE:
      pytest.skip(f"'{quantity}' requires the gkylsoft DG library")

    monkeypatch.setattr(gkquantity, "GData", _make_synthetic_gdata)
    quant = gk_quant_registry.get(quantity)
    path = str(tmp_path)

    # Every source of the fallback combination, and nothing else: no apar.
    for src in quant.source[-1]:
      for file_name in _collect_source_files(src, path, self.name, self.species, self.frame):
        open(file_name, "w").close()

    combo_idx, frames = quant.get_avail_source(path, self.name, self.species, None)
    assert combo_idx == len(quant.source) - 1, "the no-apar combination must be the one chosen"
    assert frames == [None], "a geo-only combination carries no frame"

    ctx = self._make_ctx()
    ctx.invoke(cmd.gk_load_quantity, quantity=quantity, name=self.name,
               species=self.species, frame=None, path=path, extra="dir=2")
    assert ctx.obj["data"].get_num_datasets() >= 1

  def test_multi_species_quantity_yields_a_single_dataset(self, tmp_path, monkeypatch):
    """A multi-species quantity combines its species into one dataset.

    Per-species quantities produce one dataset per requested species; a
    multi-species one (the sound speed) must instead fold them all into a
    single dataset, which is the whole reason for the separate fetch path.
    """
    monkeypatch.setattr(gkquantity, "GData", _make_synthetic_gdata)
    species = f"{_ELC_SPECIES},{_ION_SPECIES}"

    ctx = self._make_ctx()
    self._load(ctx, "c_s", str(tmp_path), species)
    assert ctx.obj["data"].get_num_datasets() == 1, (
      "the sound speed must combine both species into one dataset")

    # The same two species through a per-species quantity give one dataset each.
    ctx = self._make_ctx()
    self._load(ctx, "temp", str(tmp_path), species)
    assert ctx.obj["data"].get_num_datasets() == 2, (
      "a per-species quantity must still produce one dataset per species")

  def test_per_species_extra_array_reaches_each_species(self, tmp_path, monkeypatch):
    """'--extra mass=1,2' must give species #0 mass 1 and species #1 mass 2.

    This drives the whole chain end to end: the command parses the array and
    tags each species with its index, and _get_ctx_val picks the entry. temp
    from the Maxwellian moments is mass*<component 2>, so the two datasets must
    come out differing by exactly the mass ratio.
    """
    monkeypatch.setattr(gkquantity, "GData", _make_synthetic_gdata_no_attrs)

    ctx = self._make_ctx()
    self._load(ctx, "temp", str(tmp_path), f"{_ELC_SPECIES},{_ION_SPECIES}", extra="mass=1,2")

    assert ctx.obj["data"].get_num_datasets() == 2
    # Several species are tagged per species, in the order they were requested.
    first = ctx.obj["data"].get_dataset(0, tag=f"default_{_ELC_SPECIES}").get_values()
    second = ctx.obj["data"].get_dataset(0, tag=f"default_{_ION_SPECIES}").get_values()
    assert np.allclose(second, 2.0*first), (
      "the second species must be computed with the second mass of the array")

  @pytest.mark.parametrize("extra", [
    "mass=1,2,dir=0",   # Pairs separated by commas, as --extra has always been written.
    "mass=1,2 dir=0",   # Pairs separated by spaces.
    "mass=1,2  dir=0",  # Extra whitespace.
  ])
  def test_extra_pairs_may_be_separated_by_commas_or_spaces(self, extra, tmp_path, monkeypatch):
    monkeypatch.setattr(gkquantity, "GData", _make_synthetic_gdata_no_attrs)

    ctx = self._make_ctx()
    self._load(ctx, "temp", str(tmp_path), f"{_ELC_SPECIES},{_ION_SPECIES}", extra=extra)

    assert ctx.obj["data"].get_num_datasets() == 2
    first = ctx.obj["data"].get_dataset(0, tag=f"default_{_ELC_SPECIES}").get_values()
    second = ctx.obj["data"].get_dataset(0, tag=f"default_{_ION_SPECIES}").get_values()
    assert np.allclose(second, 2.0*first), (
      f"'--extra {extra}' must give the two species masses 1 and 2")

  def test_extra_overrides_the_file_attributes(self, tmp_path, monkeypatch):
    """'--extra mass=' must win over the mass stored in the output files.

    The synthetic files carry mass=1, so asking for mass=1,2 must change the
    result for both species: temp from the Maxwellian moments is
    mass*<component 2>.
    """
    monkeypatch.setattr(gkquantity, "GData", _make_synthetic_gdata)  # mass=1 in ctx.
    species = f"{_ELC_SPECIES},{_ION_SPECIES}"

    ctx = self._make_ctx()
    self._load(ctx, "temp", str(tmp_path), species)
    from_files = ctx.obj["data"].get_dataset(0, tag=f"default_{_ION_SPECIES}").get_values()

    ctx = self._make_ctx()
    self._load(ctx, "temp", str(tmp_path), species, extra="mass=1,2")
    overridden = ctx.obj["data"].get_dataset(0, tag=f"default_{_ION_SPECIES}").get_values()

    assert np.allclose(overridden, 2.0*from_files), (
      "--extra mass= must override the mass attribute stored in the files")

  def test_scalar_extra_applies_to_every_species(self, tmp_path, monkeypatch):
    """A single '--extra mass=2' must be shared by every species."""
    monkeypatch.setattr(gkquantity, "GData", _make_synthetic_gdata_no_attrs)

    ctx = self._make_ctx()
    self._load(ctx, "temp", str(tmp_path), f"{_ELC_SPECIES},{_ION_SPECIES}", extra="mass=2")

    assert ctx.obj["data"].get_num_datasets() == 2
    first = ctx.obj["data"].get_dataset(0, tag=f"default_{_ELC_SPECIES}").get_values()
    second = ctx.obj["data"].get_dataset(0, tag=f"default_{_ION_SPECIES}").get_values()
    assert np.allclose(second, first)

  def test_multi_species_extra_array_reaches_nested_sources(self, tmp_path, monkeypatch):
    """Every species' nested sources must use that species' own array entry.

    c_s(kind=thermo) needs each species' temperature, and temp from the
    Maxwellian moments is mass*<component 2>. So if the sources of every species
    were resolved with the same '--extra mass=' entry, the ion temperature would
    be built from the electron mass. Only pinning the expected value catches
    that: comparing two runs would not, because the mass also enters the
    denominator through a correctly-indexed path.
    """
    monkeypatch.setattr(gkquantity, "GData", _make_synthetic_gdata_no_attrs)

    mass_e, mass_i = 1.0, 4.0
    ctx = self._make_ctx()
    self._load(ctx, "c_s", str(tmp_path), f"{_ELC_SPECIES},{_ION_SPECIES}",
               extra=f"kind=thermo,mass={mass_e},{mass_i},charge=-1,1")

    # The synthetic data gives component c the cell average (c+2), so every
    # species has n = 2 (component 0) and temp = mass*4 (component 2).
    dens, temp_e, temp_i = 2.0, 4.0*mass_e, 4.0*mass_i
    expected = np.sqrt((1.0*dens*temp_e + 3.0*dens*temp_i)/(dens*mass_i))

    values = ctx.obj["data"].get_dataset(0).get_values()
    assert np.isclose(values[0, 0]*_PSI0, expected, rtol=1e-10), (
      "each species' temperature must be built from its own '--extra mass=' entry")

  def test_multi_species_quantity_needs_a_species_list(self, tmp_path, monkeypatch):
    """Asking for the sound speed without species must say so, not crash oddly."""
    monkeypatch.setattr(gkquantity, "GData", _make_synthetic_gdata)

    ctx = self._make_ctx()
    with pytest.raises(ValueError, match="needs a species list"):
      ctx.invoke(
        cmd.gk_load_quantity,
        quantity="c_s",
        name=self.name,
        species=None,
        frame=str(self.frame),
        path=str(tmp_path),
        extra=None,
      )

  def test_sound_speed_kinds_differ(self, tmp_path, monkeypatch):
    """Both --extra kind= values must run and give genuinely different answers."""
    monkeypatch.setattr(gkquantity, "GData", _make_synthetic_gdata)
    species = f"{_ELC_SPECIES},{_ION_SPECIES}"

    values = {}
    for kind in ("ion_acoustic", "thermo"):
      ctx = self._make_ctx()
      self._load(ctx, "c_s", str(tmp_path), species, extra=f"kind={kind}")
      assert ctx.obj["data"].get_num_datasets() == 1
      values[kind] = ctx.obj["data"].get_dataset(0).get_values().copy()

    assert not np.allclose(values["ion_acoustic"], values["thermo"]), (
      "the two sound-speed definitions should not coincide for this data")

  def _check_distf_real(self):
    """
    Test the distf function with real data present in the test_data directory.
    """
    ctx = self._make_ctx()
    try:
      ctx.invoke(
        cmd.gk_load_quantity,
        quantity="distf",
        name=_DISTF_REAL["name"],
        species=_DISTF_REAL["species"],
        frame=str(_DISTF_REAL["frame"]),
        path=_TEST_DATA_DIR,
      )
    except (RuntimeError, FileNotFoundError, OSError) as err:
      if not _DGOPS_AVAILABLE:
        pytest.skip(f"'distf' requires the gkylsoft DG library: {err}")
      raise

    assert ctx.obj["data"].get_num_datasets() >= 1, (
      "gk-load-quantity produced no dataset for quantity 'distf'")
