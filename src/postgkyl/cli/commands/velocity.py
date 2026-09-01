"""Deprecated module path; the adapter invokes ``five-moment-velocity``."""
from postgkyl.cli.legacy import build_legacy_command
command = build_legacy_command("velocity")
