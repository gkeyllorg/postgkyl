"""The GUI's processing chain, as plain functions the notebook wires to widgets.

A chain is a tuple of :class:`Step` records, each one ``PostgkylSession``
call (a ``pgkyl`` command and its options). :func:`run` executes the steps in
a session and :func:`python_script` renders the same steps as session Python,
so the Python the GUI shows, and the command line the session records, are
what it computed.

Nothing here imports marimo: the notebook only turns widget values into a
chain and figures into pixels.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import glob
import inspect
import io as _bytes_io
import json
import os
import re

import numpy as np

import postgkyl as pg
from postgkyl import io
from postgkyl.cli import PostgkylSession

__all__ = [
    "Output", "Step", "GridInfo", "scan_outputs", "list_simulations",
    "weight_file", "pick_frames", "apply", "run", "processed", "probe",
    "plot_step", "python_script", "frame_spec", "figure_png", "make_movie",
    "Settings", "TRANSFORMS", "parse_options", "build_chain", "quantity_frames",
    "save_state", "load_state"
]

# Fluctuations and averages of gyrokinetic fields are weighted by the
# configuration-space Jacobian, written beside the simulation output.
_WEIGHT_STEM = "geo_int_jacobgeo"


# --------------------------------------------------------------- discovery
@dataclass(frozen=True)
class Output:
  """A plottable file family in a directory.

  Attributes:
    label: Display name, the quantity without the simulation prefix unless
      that would be ambiguous.
    files: File name for each frame, in frame order (one entry, frame
      ``None``, for a frame-less file such as geometry).
    sim: The simulation prefix (``<sim>[_b<N>]``) the file belongs to.
  """

  label: str
  files: tuple[tuple[int | None, str], ...]
  sim: str

  @property
  def frames(self) -> list[int]:
    """The frame numbers, empty for a frame-less file."""
    return [frame for frame, _ in self.files if frame is not None]

  def file_name(self, frame: int | None) -> str:
    """The file holding ``frame`` (any frame for a frame-less file)."""
    if not self.frames:
      return self.files[0][1]
    return dict(self.files)[frame]


def scan_outputs(directory: str) -> dict[str, Output]:
  """Group a directory's ``.gkyl`` files into outputs, by Gkeyll's naming
  convention (:func:`postgkyl.io.parse_output_name`).

  Files sharing a stem form one output with one file per frame; files with
  no frame number (geometry, integrated diagnostics) are single outputs.
  """
  directory = os.path.expanduser(directory.strip())
  if not os.path.isdir(directory):
    return {}
  families: dict[str, dict] = {}
  for file_name in sorted(glob.glob(os.path.join(directory, "*.gkyl"))):
    name = io.parse_output_name(file_name)
    if name.restart:
      continue
    key = name.stem if name.frame is not None else f"{name.stem} (static)"
    family = families.setdefault(key, {
        "name": name,
        "files": {},
        "static": name.frame is None
    })
    family["files"][name.frame] = file_name

  # Label by quantity; fall back to the full stem when simulations clash.
  labels = {
      key: (family["name"].quantity or family["name"].stem) +
      (" (static)" if family["static"] else "")
      for key, family in families.items()
  }
  clashes = Counter(labels.values())
  outputs: dict[str, Output] = {}
  for key, family in families.items():
    label = labels[key] if clashes[labels[key]] == 1 else key
    files = sorted(family["files"].items(),
                   key=lambda item: (item[0] is not None, item[0] or 0))
    outputs[label] = Output(label=label,
                            files=tuple(files),
                            sim=os.path.basename(family["name"].prefix))
  return outputs


def list_simulations(directory: str) -> list[str]:
  """The simulation prefixes (``<sim>[_b<N>]``) of a directory's outputs."""
  return sorted({output.sim for output in scan_outputs(directory).values()})


def weight_file(directory: str, sim: str) -> str | None:
  """The Jacobian weight file of ``sim`` in ``directory``, if written."""
  path = os.path.join(os.path.expanduser(directory.strip()),
                      f"{sim}-{_WEIGHT_STEM}.gkyl")
  return path if os.path.isfile(path) else None


def quantity_frames(directory: str, sim: str, quantity: str,
                    species: str | None) -> list[int]:
  """Frames at which ``quantity`` can be computed for ``species`` (every
  listed species for a multi-species quantity); empty when none can."""
  quant = pg.gk.gk_quant_registry.get(quantity)
  path = os.path.expanduser(directory.strip()).rstrip("/") + "/"
  species_list = ([sp.strip() for sp in species.split(",")
                   if sp.strip()] if species else [None])
  try:
    if quant.is_multi_species:
      _, frames = quant.get_avail_source_multi(path, sim, species_list, None)
    else:
      _, frames = quant.get_avail_source(path, sim, species_list[0], None)
  except FileNotFoundError:
    return []
  return [frame for frame in frames if frame is not None]


def pick_frames(frames: list[int], text: str) -> list[int] | None:
  """The frames a range text selects among the available ``frames``, read as
  ``load --frame`` reads it (:func:`postgkyl.io.select_frames`): ``':'``
  all, ``'::2'`` every other frame number, ``'-10:'`` the last ten, ``'-1'``
  the last one. ``None`` for blank text.

  Raises:
    ValueError: the text is malformed or selects a frame that is not
      available.
  """
  text = (text or "").strip()
  if not text:
    return None
  try:
    picked = io.select_frames(text, frames)
  except ValueError as exc:
    raise ValueError(f"frame range '{text}': {exc}") from None
  missing = sorted(set(picked) - set(frames))
  if missing:
    raise ValueError(f"frame range '{text}' selects frame(s) {missing}, "
                     f"which are not among the {len(frames)} available.")
  return picked


def frame_spec(frames: list[int], available: list[int]) -> str:
  """The shortest frame specification selecting exactly ``frames`` among the
  ``available`` ones (``':'``, ``'-10:'``, ``'100:200:2'``, ...), falling
  back to the comma-separated frames."""
  frames, available = list(frames), sorted(available)
  candidates = [",".join(map(str, frames))]
  if available:
    candidates += [":", f"-{len(frames)}:"]
    steps = {b - a for a, b in zip(frames, frames[1:])}
    if len(steps) == 1:
      (step, ) = steps
      every = f":{step}" if step != 1 else ""
      candidates += [
          f"{start}:{stop}{every}" for start in ("", str(frames[0]))
          for stop in ("", str(frames[-1] + 1))
      ]

  def selects(spec):
    try:
      return io.select_frames(spec, available) == frames
    except ValueError:
      return False

  return min((spec for spec in candidates if selects(spec)),
             key=len,
             default=candidates[0])


# ------------------------------------------------------------------- chains
@dataclass(frozen=True)
class Step:
  """One ``PostgkylSession`` call: a ``pgkyl`` command and its options."""

  verb: str
  options: tuple[tuple[str, object], ...] = ()

  @classmethod
  def of(cls, verb: str, **options) -> "Step":
    """A step with the given options, ``None`` values left out."""
    return cls(verb, tuple((k, v) for k, v in options.items() if v is not None))

  @property
  def kwargs(self) -> dict:
    return dict(self.options)


# The weight file joins the session under this tag; ``activate`` then sets it
# aside, so later commands skip it while ``--weight`` still finds it.
_WEIGHT_TAG = "weight"
# The tag the loaders give the data they load.
_DATA_TAG = inspect.signature(pg.load).parameters["tag"].default


def apply(session: PostgkylSession, step: Step) -> PostgkylSession:
  """Make ``step``'s call on ``session``."""
  return getattr(session, step.verb)(**step.kwargs)


def run(steps: tuple[Step, ...]) -> PostgkylSession:
  """A session, opening no window, that has made every call of ``steps``."""
  session = PostgkylSession(no_show=True)
  for step in steps:
    apply(session, step)
  return session


def processed(steps: tuple[Step, ...]) -> pg.GDataGroup:
  """The datasets ``steps`` produce, as one group."""
  return pg.GDataGroup(list(run(steps).datasets))


def plot_step(plot_options: dict, datasets) -> Step:
  """The ``plot`` call drawing every one of ``datasets`` on one figure.

  ``pgkyl`` gives each dataset (e.g. each frame) its own figure;
  ``--figure 0`` overlays them, which is what the GUI shows.
  """
  options = dict(plot_options)
  if len(datasets) > 1:
    options.setdefault("figure", 0)
  return Step.of("plot", **options)


def python_script(steps: tuple[Step, ...]) -> str:
  """The session Python making the calls :func:`run` makes."""
  calls = [
      f"s.{step.verb}({', '.join(f'{k}={v!r}' for k, v in step.options)})"
      for step in steps
  ]
  return "\n".join(
      ["from postgkyl.cli import PostgkylSession", "", "s = PostgkylSession()"
       ] + calls)


# --------------------------------------------------------- GUI settings
TRANSFORMS = ("interpolate", "local_poly", "map_to_rz", "extract_flux_surface",
              "none")
_GEOMETRY_TRANSFORMS = ("map_to_rz", "extract_flux_surface")
_FLUCT_DIMS = {"y": "1", "yz": "1,2"}


@dataclass(frozen=True)
class Settings:
  """Every choice the GUI offers, independent of the widgets showing it.

  Attributes:
    directory: Data directory.
    mode: ``"file"`` (plot an output) or ``"quantity"``
      (``pg.gk.load_quantity``).
    frames: Frames to load; ``(None,)`` for a frame-less file.
    available: Every frame of the quantity in ``"quantity"`` mode, so the
      frames are written as the shortest range that selects them.
    output: The output plotted in ``"file"`` mode.
    sim: Simulation prefix of the loaded data.
    quantity: Registered quantity of ``"quantity"`` mode.
    species: Species, or comma-separated species, of ``"quantity"`` mode.
    direction: Vector component of ``"quantity"`` mode.
    options: Further keyword options of the GK loader.
    fluct: ``"y"``/``"yz"``: plot the fluctuation about that average (3x data).
    weight: Jacobian file weighting fluctuations and averages, if any.
    source_ndim: Dimensions of the data as loaded; fluctuations about
      ``y`` or ``(y, z)`` need 3x ``(x, y, z)`` data.
    ndim: Dimensions of the data the averages and selections index (the
      probed layout, :func:`probe`).
    average: Dimensions averaged over, before the transform.
    transform: One of :data:`TRANSFORMS`.
    num_interp: Points per cell of ``interpolate``/``local_poly``.
    mapc2p: Mapping file of the geometry transforms (default: found).
    phi_tor: Toroidal angle (radians) of ``map_to_rz``.
    nz_interp: Points per cell along z of the geometry transforms.
    x_idx: Radial cell of ``extract_flux_surface``.
    select: ``(dimension, coordinate)`` pairs, the coordinate a cell index
      on a curvilinear dimension.
    comp: Component to keep.
    collect: Stack the frames along time.
  """

  directory: str
  mode: str
  frames: tuple
  available: tuple[int, ...] = ()
  output: Output | None = None
  sim: str | None = None
  quantity: str | None = None
  species: str | None = None
  direction: int | None = None
  options: tuple[tuple[str, object], ...] = ()
  fluct: str = "none"
  weight: str | None = None
  source_ndim: int = 0
  ndim: int = 0
  average: tuple[int, ...] = ()
  transform: str = "interpolate"
  num_interp: int | None = None
  mapc2p: str | None = None
  phi_tor: float = 0.0
  nz_interp: int | None = None
  x_idx: int = 0
  select: tuple[tuple[int, float], ...] = ()
  comp: int | None = None
  collect: bool = False


def parse_options(text: str) -> dict:
  """Keyword options from ``'key=value key2=v1,v2'`` text: numbers are
  converted, and several comma-separated values become a list (one per
  species)."""
  options = {}
  for pair in re.split(r"[,\s]+(?=[^\s,=]+=)", (text or "").strip()):
    if not pair:
      continue
    key, sep, value = pair.partition("=")
    if not sep or not key.strip():
      raise ValueError(f"option '{pair}' is not key=value")
    values = [_number(v.strip()) for v in value.split(",") if v.strip()]
    options[key.strip()] = values[0] if len(values) == 1 else values
  return options


def _number(text: str):
  for kind in (int, float):
    try:
      return kind(text)
    except ValueError:
      pass
  return text


def build_chain(settings: Settings,
                *,
                probe_only: bool = False) -> tuple[Step, ...]:
  """The steps computing ``settings``' data: load, fluctuation, average,
  transform, select, collect. With ``probe_only`` they stop after the
  transform and load the first frame only, which is what the selection
  sliders are sized from.

  A weighted fluctuation or average first loads the weight, then activates
  the data alone, so the weight is never transformed or plotted.

  Raises:
    ValueError: the settings combine choices that do not apply together.
  """
  s = settings
  frames = s.frames[:1] if probe_only else s.frames
  numbered = [f for f in frames if f is not None]
  if s.mode == "file":
    loads = _file_loads(s.output, frames)
  elif s.mode == "quantity":
    loads = [
        Step.of("gk_load_quantity",
                quantity=s.quantity,
                species=s.species,
                name=s.sim,
                frame=frame_spec(numbered, s.available) if numbered else None,
                path=s.directory,
                direction=s.direction,
                **dict(s.options))
    ]
  else:
    raise ValueError(f"unknown mode {s.mode!r}")

  weight = _WEIGHT_TAG if s.weight else None
  steps: list[Step] = []
  if s.fluct != "none":
    if s.source_ndim != 3:
      raise ValueError(
          f"fluctuations about the {s.fluct} average need 3x (x, y, z) "
          f"configuration-space data, not {s.source_ndim}-D data.")
    steps.append(
        Step.of("fluctuation", dims=_FLUCT_DIMS[s.fluct], weight=weight))
  if probe_only:
    steps.extend(_transform_steps(s))
  else:
    average = sorted(s.average)
    if average and s.transform in _GEOMETRY_TRANSFORMS:
      raise ValueError(f"averaging does not apply before {s.transform}, "
                       "which needs the full configuration space.")
    dims = ",".join(map(str, average))
    if average and len(average) == s.ndim:
      # One mean per frame, kept as a dataset so collect makes a trace.
      steps.append(Step.of("average", dims=dims, weight=weight,
                           as_dataset=True))
    else:
      if average:
        steps.append(Step.of("average", dims=dims, weight=weight))
      steps.extend(_transform_steps(s))
      steps.extend(_select_steps(s, average=average))
    if s.collect and len(frames) > 1:
      steps.append(Step.of("collect"))

  if any(step.kwargs.get("weight") for step in steps):
    loads = [
        Step.of("load", file_name=s.weight, tag=_WEIGHT_TAG), *loads,
        Step.of("activate", tags=[_DATA_TAG])
    ]
  return tuple(loads + steps)


def _file_loads(output: Output, frames) -> list[Step]:
  """Load ``frames`` of ``output``: one load of the output's file pattern
  and a frame specification when that reads exactly these files, else one
  load per file."""
  files = [output.file_name(f) for f in frames]
  if len(files) > 1:
    name = io.parse_output_name(files[0])
    pattern = os.path.join(glob.escape(name.directory),
                           f"{glob.escape(name.stem)}_[0-9]*.gkyl")
    spec = frame_spec(frames, output.frames)
    if io.select_frame_files(glob.glob(pattern), spec) == files:
      return [Step.of("load", file_name=pattern, frame=spec)]
  return [Step.of("load", file_name=f) for f in files]


def _transform_steps(s: Settings) -> tuple[Step, ...]:
  if s.transform == "interpolate":
    return (Step.of("interpolate", num_interp=s.num_interp), )
  if s.transform == "local_poly":
    return (Step.of("local_poly", npoints=s.num_interp), )
  if s.transform == "map_to_rz":
    return (Step.of("map_to_rz",
                    mapc2p=s.mapc2p,
                    phi_tor=s.phi_tor,
                    nz_interp=s.nz_interp), )
  if s.transform == "extract_flux_surface":
    return (Step.of("extract_flux_surface",
                    mapc2p=s.mapc2p,
                    x_idx=s.x_idx,
                    nz_interp=s.nz_interp), )
  if s.transform == "none":
    return ()
  raise ValueError(f"unknown transform {s.transform!r}")


def _select_steps(s: Settings, *, average) -> tuple[Step, ...]:
  """Select at each coordinate; averaging has removed earlier dimensions."""
  options = {}
  for d, value in s.select:
    shifted = d - sum(1 for a in average if a < d)
    options[f"z{shifted}"] = value
  if s.comp is not None:
    options["comp"] = s.comp
  return (Step.of("select", **options), ) if options else ()


# ------------------------------------------------------------------ probing
@dataclass(frozen=True)
class GridInfo:
  """Selectable layout of a chain's result.

  Attributes:
    lower, upper: Coordinate range per dimension: the logical coordinate
      of a mapped dimension that records one (after ``map_to_rz``, minor
      radius and poloidal angle), else the cell-index range of a
      curvilinear dimension.
    cells: Values per dimension.
    curvilinear: Whether each dimension has no 1-D coordinate at all, so
      it is selected by cell index.
    num_fields: Physical fields (components) per point.
  """

  lower: tuple[float, ...]
  upper: tuple[float, ...]
  cells: tuple[int, ...]
  curvilinear: tuple[bool, ...]
  num_fields: int

  @property
  def ndim(self) -> int:
    return len(self.cells)


def probe(steps: tuple[Step, ...]) -> GridInfo:
  """The layout of the first dataset ``steps`` produce."""
  data = run(steps).datasets[0]
  if data.ctx.get("basis_type") and not data.is_interpolated:
    fields = data.interpolate().num_comps  # Count fields, not coefficients.
  else:
    fields = data.num_comps
  lower, upper, curvilinear = [], [], []
  cells = tuple(int(n) for n in data.values.shape[:-1])
  logical = data.ctx.get("logical_grid") or []
  for d, coord in enumerate(data.grid):
    coord = np.asarray(coord)
    if coord.ndim > 1 and d < len(logical) and logical[d] is not None:
      coord = np.asarray(logical[d])  # select resolves values on it.
    if coord.ndim > 1:
      lower.append(0.0)
      upper.append(float(cells[d] - 1))
      curvilinear.append(True)
    else:
      lower.append(float(coord.min()))
      upper.append(float(coord.max()))
      curvilinear.append(False)
  return GridInfo(tuple(lower), tuple(upper), cells, tuple(curvilinear),
                  int(fields))


# --------------------------------------------------------------- rendering
def figure_png(figure, *, dpi: int = 110) -> bytes:
  """The PNG bytes of ``plot``'s result, then closed.

  ``plot`` returns a list when it draws one figure per multiblock family;
  the first is shown.
  """
  import matplotlib.pyplot as plt

  figures = figure if isinstance(figure, list) else [figure]
  try:
    buffer = _bytes_io.BytesIO()
    figures[0].savefig(buffer, format="png", dpi=dpi)
  finally:
    for each in figures:
      plt.close(each)
  return buffer.getvalue()


def make_movie(frames: list[pg.GDataGroup], file_name: str, *, fps: int,
               plot_options: dict, fixed_range: bool) -> str:
  """Animate one processed group per frame into ``file_name`` with
  ``pg.animate``. The plot options ``pg.animate`` also takes are forwarded;
  with ``fixed_range`` every frame shares the y (1-D) or colour (2-D) range,
  unless a limit is given. Every frame's title ends with its frame number
  and time. Returns ``file_name``."""
  accepted = inspect.signature(pg.animate).parameters
  options = {k: v for k, v in plot_options.items() if k in accepted}
  pg.animate(frames,
             saveas=file_name,
             fps=fps,
             no_show=True,
             variable_range=not fixed_range,
             stamp_title=True,
             **options)
  return file_name


# ---------------------------------------------------------------- state files
# A state file records every widget's value, to reopen the GUI as it was with
# ``pgkyl-gui --state <file>``. The version guards against reading a file
# whose widgets meant something else.
_STATE_VERSION = 1


def save_state(file_name: str, widgets: dict) -> str:
  """Write the GUI's widget values to ``file_name``; return it."""
  with open(file_name, "w") as f:
    json.dump({
        "postgkyl_gui_state": _STATE_VERSION,
        "widgets": widgets
    },
              f,
              indent=2)
  return file_name


def load_state(file_name: str) -> dict:
  """The widget values a :func:`save_state` file holds.

  Raises:
    OSError: the file cannot be read.
    ValueError: it is not a GUI state file of this version.
  """
  with open(file_name) as f:
    try:
      data = json.load(f)
    except json.JSONDecodeError as exc:
      raise ValueError(f"{file_name} is not a GUI state file: {exc}") from None
  if not (isinstance(data, dict) and data.get("postgkyl_gui_state")
          == _STATE_VERSION and isinstance(data.get("widgets"), dict)):
    raise ValueError(f"{file_name} is not a version-{_STATE_VERSION} GUI "
                     "state file")
  return data["widgets"]
