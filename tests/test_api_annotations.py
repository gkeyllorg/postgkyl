"""Python contracts remain truthful while CLI codecs narrow their input syntax."""

from typing import Annotated, get_args, get_origin, get_type_hints

from postgkyl import operations
from postgkyl.cli.compiler import compile_callable
from postgkyl.cli_spec import CliType
from postgkyl.gdatastate import GDataState


def test_map_annotation_accepts_loaded_dataset_with_filename_cli_codec():
  annotation = get_type_hints(operations.map, include_extras=True)["mapping"]
  assert get_origin(annotation) is Annotated
  python_type, *metadata = get_args(annotation)
  assert GDataState in get_args(python_type)
  assert str in get_args(python_type)
  assert CliType(str) in metadata
  command = compile_callable(operations.map)
  mapping = next(p for p in command.parameters if p.name == "mapping")
  assert mapping.argument


def test_select_annotation_preserves_numeric_python_selectors():
  hints = get_type_hints(operations.select, include_extras=True)
  for name in ("comp", "z0", "z1", "z2", "z3", "z4", "z5"):
    python_type, *metadata = get_args(hints[name])
    assert {int, float, str, type(None)} == set(get_args(python_type))
    assert CliType(str | None) in metadata


def test_public_receiver_types_resolve_in_the_owning_module():
  for function in (operations.interpolate, operations.local_poly,
                   operations.average, operations.magsq, operations.fft,
                   operations.mask, operations.fit, operations.integrate):
    hints = get_type_hints(function, include_extras=True)
    assert hints["data"] is GDataState
