"""Execute a resolved managed Gkeyll build without rereading configuration."""

import os
from pathlib import Path
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from gkeyll_config import BuildPlan

ROOT = Path(__file__).resolve().parents[1]


def build_gkeyll(plan: "BuildPlan") -> None:
  """Build the selected stack, preserving existing sources and revisions."""
  if plan.mode != "managed":
    raise ValueError("Existing Gkeyll builds must be rebuilt by their owner")
  directory = plan.directory
  if not (directory / ".git").exists():
    subprocess.run(["sh", str(ROOT / "scripts/update_gkeyll.sh")], check=True)
  sparse = subprocess.run(
      ["git", "-C",
       str(directory), "config", "--bool", "core.sparseCheckout"],
      capture_output=True,
      text=True)
  if sparse.stdout.strip() == "true":
    subprocess.run([
        "git", "-C",
        str(directory), "sparse-checkout", "add", *(f"/{module}/"
                                                    for module in plan.modules)
    ],
                   check=True)

  cc = os.environ.get("CC") or "cc"
  lapack_lite = "yes" if plan.lapack_lite else "no"
  print(
      f"# Configuring gkeyll (CC={cc}, lapack-lite={lapack_lite}, app={plan.module})",
      flush=True)
  subprocess.run([
      "./configure", f"CC={cc}", f"--use-lapack-lite={lapack_lite}",
      f"--app={plan.module}"
  ],
                 cwd=directory,
                 check=True)
  arch_flags = os.environ.get("ARCH_FLAGS", "")
  jobs = os.environ.get("BUILD_JOBS") or str(max(1, (os.cpu_count() or 2) // 2))
  # Gkeyll's module makefiles resolve BUILD_DIR relative to the checkout.
  build_dir = plan.build_dir.relative_to(directory)
  print(f"# Building {plan.module} and prerequisites", flush=True)
  subprocess.run([
      "make", plan.module, "LIBRARY_TARGET=library", f"BUILD_DIR={build_dir}",
      f"ARCH_FLAGS={arch_flags}", f"-j{jobs}"
  ],
                 cwd=directory,
                 check=True)
  for module in plan.modules:
    library = plan.build_dir / module / f"libg0{module}.so"
    if not library.is_file():
      raise ValueError(f"expected {library} after build, but it is missing")
    print(f"# Built {library}", flush=True)


if __name__ == "__main__":
  from gkeyll_config import load_config

  try:
    build_gkeyll(load_config())
  except (ValueError, OSError, subprocess.CalledProcessError) as exc:
    sys.exit(f"error: {exc}")
