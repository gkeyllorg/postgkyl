"""Check an installed wheel and run native contracts outside the source tree."""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_installed_wheel():
  import postgkyl
  from postgkyl import gpython

  gpython.require()
  package = Path(postgkyl.__file__).resolve()
  assert "site-packages" in package.parts, package
  extension = gpython.lib_path()
  assert extension.with_name("libg0core.so").is_file(), extension
  info = gpython.build_info()
  assert info is not None and len(info["gkeyll_commit"]) == 40, info
  print(f"Checking installed native wheel: {extension}", flush=True)


def main():
  argparse.ArgumentParser(description=__doc__).parse_args()
  # Several tests add ../src to sys.path. Copying them ensures that even those
  # tests cannot accidentally import the checkout instead of the wheel.
  with tempfile.TemporaryDirectory(prefix="postgkyl-wheel-tests-") as directory:
    tests = Path(directory) / "tests"
    shutil.copytree(ROOT / "tests",
                    tests,
                    ignore=shutil.ignore_patterns("__pycache__", "generated"))
    shutil.copy2(ROOT / "pyproject.toml", Path(directory) / "pyproject.toml")
    shutil.copy2(__file__, tests / "test_installed_wheel.py")
    subprocess.run([
        sys.executable,
        "-m",
        "pytest",
        "--timeout=120",
        "-q",
        "tests/test_installed_wheel.py",
        "tests/test_gpython_array.py",
        "tests/test_gpython_basis.py",
        "tests/test_gpython_kernels.py",
        "tests/test_gpython_rio.py",
        "tests/test_analytic_observables.py",
        "tests/test_native_quadrature.py",
    ],
                   cwd=directory,
                   check=True)
  subprocess.run([sys.executable, "-m", "pip", "check"], check=True)


if __name__ == "__main__":
  main()
