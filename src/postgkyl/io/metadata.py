"""Resolve Gkeyll metadata once and retain its sources for inspection."""

from copy import deepcopy

import msgpack


def encode_structural_metadata(metadata: dict) -> dict:
  """Encode integer axis identities using MessagePack's string map keys."""
  result = dict(metadata)
  if "mapped_axes" in result:
    result["mapped_axes"] = {
        str(axis): offset
        for axis, offset in result["mapped_axes"].items()
    }
  return result


def unpack_gkyl_metadata(packed: bytes) -> object:
  """Read shared structural metadata, including legacy integer map keys.

  ``mapped_axes`` always has integer identities in memory. New files encode
  these as strings for strict MessagePack readers; older Postgkyl files
  stored integer keys directly. Decoding the keys cannot recover coordinates
  omitted by those older files.
  """
  if not packed:
    return {}
  metadata = msgpack.unpackb(packed, strict_map_key=False)
  if isinstance(metadata, dict) and "mapped_axes" in metadata:
    metadata["mapped_axes"] = {
        int(axis): int(offset)
        for axis, offset in metadata["mapped_axes"].items()
    }
  return metadata


def resolve_gkyl_metadata(ctx: dict, metadata: object, *,
                          basis_type: str | None, poly_order: int | None,
                          value_form: str | None) -> None:
  """Apply header metadata, overrides, and the legacy modal assumption.

  ``_load_metadata`` is a snapshot of the source file and load decisions,
  not provenance for the current values after subsequent operations.
  Both binary readers use the same precedence and record original key names.
  """
  provenance = ctx.setdefault("_load_metadata", {})
  provenance["file_metadata"] = deepcopy(metadata)
  has_basis = False
  if isinstance(metadata, dict):
    if metadata.get("mapped_axes") or metadata.get("grid_type") == "mapped":
      raise ValueError(
          "gkyl mapped metadata has no serialized coordinates; cannot "
          "reconstruct the physical grid")
    for key, value in metadata.items():
      if key in ("polyOrder", "poly_order"):
        ctx["poly_order"] = value
      elif key in ("basisType", "basis_type"):
        ctx["basis_type"] = value
        has_basis = True
      elif key != "_load_metadata":
        ctx[key] = value

  overrides = {
      key: value
      for key, value in (("basis_type", basis_type), ("poly_order", poly_order),
                         ("value_form", value_form)) if value is not None
  }
  ctx.update(overrides)
  provenance["overrides"] = overrides
  if (has_basis or basis_type is not None) and "value_form" not in ctx:
    ctx["value_form"] = "modal"
    provenance.setdefault(
        "defaults",
        {})["value_form"] = ("modal", "basis specified; value_form absent")
