"""Exercise clean-build revision selection without fetching or compiling."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
PIN = (ROOT / "scripts/gkeyll-revision").read_text().strip()
BRANCH_TIP = "a" * 40
OVERRIDE = "b" * 40


def _build_checkout(tmp_path, *, revision=None, dirty=False, pin=True):
  scripts = tmp_path / "scripts"
  scripts.mkdir()
  for name in ("build_gkeyll.sh", "gkeyll-branch"):
    shutil.copyfile(ROOT / "scripts" / name, scripts / name)
  if pin:
    (scripts / "gkeyll-revision").write_text(PIN + "\n")
  (scripts / "build_gpython.sh").write_text("#!/bin/sh\nexit 0\n")
  producer = tmp_path / "gkeyll"
  (producer / ".git").mkdir(parents=True)
  (producer / "build/core").mkdir(parents=True)
  (producer / "build/core/libg0core.so").write_bytes(b"test artifact")
  configure = producer / "configure"
  configure.write_text("#!/bin/sh\nexit 0\n")
  configure.chmod(0o755)
  binary_dir = tmp_path / "bin"
  binary_dir.mkdir()
  fake_git = binary_dir / "git"
  fake_git.write_text(f"#!{sys.executable}\n" + '''
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
root = Path(os.environ["TEST_BUILD_ROOT"])
with (root / "git_calls.jsonl").open("a") as stream:
    stream.write(json.dumps(args) + "\\n")
if args[0] == "-C":
    args = args[2:]
if args[0] == "diff" and os.environ.get("TEST_DIRTY") == "1":
    sys.exit(1)
if args[0] == "cat-file":
    sys.exit(1)  # Exercise the fetch for a pin older than the shallow tip.
if args[:2] == ["checkout", "--detach"]:
    (root / "detached_revision").write_text(args[2])
if args[0] == "rev-parse":
    detached = root / "detached_revision"
    if args[1] == "HEAD" and detached.exists():
        print(detached.read_text())
    else:
        print("a" * 40)
''')
  fake_git.chmod(0o755)
  fake_make = binary_dir / "make"
  fake_make.write_text("#!/bin/sh\nexit 0\n")
  fake_make.chmod(0o755)
  env = dict(os.environ,
             PATH=str(binary_dir) + os.pathsep + os.environ["PATH"],
             TEST_BUILD_ROOT=str(tmp_path),
             TEST_DIRTY="1" if dirty else "0")
  env.pop("GKEYLL_REVISION", None)
  if revision is not None:
    env["GKEYLL_REVISION"] = revision
  result = subprocess.run(["sh", str(scripts / "build_gkeyll.sh")],
                          cwd=tmp_path,
                          env=env,
                          text=True,
                          capture_output=True,
                          timeout=10)
  calls = [
      json.loads(line)
      for line in (tmp_path / "git_calls.jsonl").read_text().splitlines()
  ]
  return result, [args[2:] if args[0] == "-C" else args for args in calls]


@pytest.mark.parametrize("override,expected", [(None, PIN),
                                               (OVERRIDE, OVERRIDE)])
def test_clean_build_uses_immutable_revision_even_when_branch_advances(
    tmp_path, override, expected):
  result, calls = _build_checkout(tmp_path, revision=override)
  assert result.returncode == 0, result.stderr
  assert expected != BRANCH_TIP
  assert (tmp_path / "detached_revision").read_text() == expected
  assert ["fetch", "--depth", "1", "--filter=blob:none", "origin",
          expected] in calls
  assert ["checkout", "--detach", expected] in calls
  assert f"# Using Gkeyll {expected}" in result.stdout


@pytest.mark.parametrize(
    "revision", ["", "main", "abcdef0", "A" * 40, "g" * 40, "a" * 39, "a" * 41])
def test_mutable_or_malformed_revision_fails_before_fetch(tmp_path, revision):
  result, calls = _build_checkout(tmp_path, revision=revision)
  assert result.returncode != 0
  assert "full lowercase 40-character commit hash" in result.stderr
  assert not any(args[0] in ("fetch", "checkout", "merge") for args in calls)


def test_missing_pin_has_no_moving_branch_fallback(tmp_path):
  result, calls = _build_checkout(tmp_path, pin=False)
  assert result.returncode != 0
  assert "revision file is missing" in result.stderr
  assert not any(args[0] in ("fetch", "checkout", "merge") for args in calls)


def test_revision_pin_does_not_bypass_dirty_checkout_protection(tmp_path):
  result, calls = _build_checkout(tmp_path, dirty=True)
  assert result.returncode != 0
  assert "tracked modifications" in result.stderr
  assert not any(args[0] in ("fetch", "checkout", "merge") for args in calls)
