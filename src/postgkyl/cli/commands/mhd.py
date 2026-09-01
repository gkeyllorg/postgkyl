"""Deprecated MHD variable dispatcher module path."""
from postgkyl.cli.commands import LEGACY_DISPATCHERS
command = next(item for item in LEGACY_DISPATCHERS if item.name == "mhd")
