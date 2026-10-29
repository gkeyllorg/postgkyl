"""Separate native rebuilds from explicit upstream updates using a local remote."""

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]


def _git(directory, *args):
  return subprocess.check_output(
      ["git", "-C", str(directory), *args], text=True).strip()


@pytest.fixture
def selected_branch():
  return (ROOT / "scripts/gkeyll-branch").read_text().strip()


@pytest.fixture
def checkout(tmp_path, selected_branch):
  remote = tmp_path / "remote"
  remote.mkdir()
  _git(remote, "init", "-b", selected_branch)
  _git(remote, "config", "user.name", "Build test")
  _git(remote, "config", "user.email", "build@example.invalid")
  configure = remote / "configure"
  configure.write_text(
      "#!/bin/sh\nmkdir -p build/core\ntouch build/core/libg0core.so\n")
  configure.chmod(0o755)
  _git(remote, "add", "configure")
  _git(remote, "commit", "-m", "Initial source")
  workspace = tmp_path / "workspace"
  scripts = workspace / "scripts"
  scripts.mkdir(parents=True)
  for name in ("build_gkeyll.sh", "build_gkeyll.py", "update_gkeyll.sh",
               "gkeyll_config.py"):
    source = (ROOT / "scripts" / name).read_text()
    (scripts / name).write_text(
        source.replace("https://github.com/ammarhakim/gkeyll.git",
                       remote.as_uri()))
  (scripts / "gkeyll-branch").write_text(selected_branch + "\n")
  shutil.copyfile(ROOT / "scripts/gkeyll-modules", scripts / "gkeyll-modules")
  shutil.copyfile(ROOT / "postgkyl-build.toml",
                  workspace / "postgkyl-build.toml")
  binary_dir = tmp_path / "bin"
  binary_dir.mkdir()
  make = binary_dir / "make"
  make.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > make-args\n')
  make.chmod(0o755)
  env = dict(os.environ, PATH=str(binary_dir) + os.pathsep + os.environ["PATH"])
  for name in ("POSTGKYL_GKEYLL_MODULE", "POSTGKYL_GKEYLL_DIR",
               "POSTGKYL_BUILD_CONFIG"):
    env.pop(name, None)
  # A stale environment override must never restore commit pinning.
  env["GKEYLL_REVISION"] = _git(remote, "rev-parse", "HEAD")
  return workspace, remote, env


def _run(checkout, script="build_gkeyll.sh"):
  workspace, _, env = checkout
  return subprocess.run(["sh", str(workspace / "scripts" / script)],
                        cwd=workspace,
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=20)


@pytest.mark.parametrize("existing_branch", [None, "old-feature"])
def test_only_explicit_update_follows_selected_branch_as_remote_advances(
    checkout, existing_branch, selected_branch):
  workspace, remote, _ = checkout
  if existing_branch is not None:
    _git(remote, "branch", existing_branch)
    _git(workspace, "clone", "--depth", "1", "--branch", existing_branch,
         remote.as_uri(), "gkeyll")
  first = _run(checkout, "update_gkeyll.sh")
  assert first.returncode == 0, first.stderr
  producer = workspace / "gkeyll"
  initial = _git(producer, "rev-parse", "HEAD")
  _git(remote, "commit", "--allow-empty", "-m", "New upstream change")
  latest = _git(remote, "rev-parse", "HEAD")
  assert latest != initial
  second = _run(checkout)
  assert second.returncode == 0, second.stderr
  assert _git(producer, "rev-parse", "HEAD") == initial
  updated = _run(checkout, "update_gkeyll.sh")
  assert updated.returncode == 0, updated.stderr
  assert _git(producer, "rev-parse", "HEAD") == latest
  assert _git(producer, "symbolic-ref", "--short", "HEAD") == selected_branch
  assert _git(producer, "rev-parse", "--abbrev-ref",
              "@{upstream}") == f"origin/{selected_branch}"


@pytest.mark.parametrize("local_change", ["dirty", "commit"])
def test_build_preserves_local_changes_and_update_refuses_them(
    checkout, local_change):
  workspace, _, _ = checkout
  first = _run(checkout)
  assert first.returncode == 0, first.stderr
  producer = workspace / "gkeyll"
  configure = producer / "configure"
  changed = configure.read_text() + "# local change\n"
  configure.write_text(changed)
  if local_change == "commit":
    _git(producer, "add", "configure")
    _git(producer, "-c", "user.name=Build test", "-c",
         "user.email=build@example.invalid", "commit", "-m", "Local work")
  before = _git(producer, "rev-parse", "HEAD")
  result = _run(checkout)
  assert result.returncode == 0, result.stderr
  result = _run(checkout, "update_gkeyll.sh")
  assert result.returncode != 0
  expected = "tracked modifications" if local_change == "dirty" else "local commits"
  assert expected in result.stderr
  assert configure.read_text() == changed
  assert _git(producer, "rev-parse", "HEAD") == before


def test_rebuild_works_without_remote_and_keeps_local_branch(checkout):
  workspace, remote, _ = checkout
  first = _run(checkout)
  assert first.returncode == 0, first.stderr
  producer = workspace / "gkeyll"
  _git(producer, "checkout", "-b", "native-work")
  before = _git(producer, "rev-parse", "HEAD")
  remote.rename(remote.with_name("unavailable"))
  result = _run(checkout)
  assert result.returncode == 0, result.stderr
  assert _git(producer, "rev-parse", "HEAD") == before
  assert _git(producer, "symbolic-ref", "--short", "HEAD") == "native-work"


@pytest.mark.parametrize("module,expected", [
    ("core", ["core"]),
    ("moments", ["core", "moments"]),
    ("vlasov", ["core", "moments", "vlasov"]),
    ("gyrokinetic", ["core", "moments", "vlasov", "gyrokinetic"]),
    ("pkpm", ["core", "moments", "vlasov", "gyrokinetic", "pkpm"]),
])
def test_selected_stack_build(checkout, module, expected):
  workspace, remote, env = checkout
  configure = remote / "configure"
  configure.write_text(
      '#!/bin/sh\nprintf "%s\\n" "$@" > configure-args\n'
      'for module in core moments vlasov gyrokinetic pkpm; do\n'
      '  mkdir -p "build/$module" "build/postgkyl-full/$module"\n'
      '  touch "build/$module/libg0$module.so" "build/postgkyl-full/$module/libg0$module.so"\ndone\n'
  )
  _git(remote, "add", "configure")
  for name in ("core", "moments", "vlasov", "gyrokinetic", "pkpm"):
    (remote / name).mkdir()
    (remote / name / "source.c").touch()
    _git(remote, "add", name)
  _git(remote, "commit", "-m", "Module build fixture")
  env["POSTGKYL_GKEYLL_MODULE"] = module
  result = _run(checkout)
  assert result.returncode == 0, result.stderr
  args = (workspace / "gkeyll/configure-args").read_text().splitlines()
  assert f"--app={module}" in args
  assert f"--use-lapack-lite={'yes' if module == 'core' else 'no'}" in args
  built = [
      line.rsplit("libg0", 1)[1].removesuffix(".so")
      for line in result.stdout.splitlines() if line.startswith("# Built ")
  ]
  assert built == expected
  args = (workspace / "gkeyll/make-args").read_text().splitlines()
  assert args[0] == module
  assert "LIBRARY_TARGET=library" in args
  build = "build" if module == "core" else "build/postgkyl-full"
  assert f"BUILD_DIR={build}" in args
  for name in expected:
    assert (workspace / "gkeyll" / name / "source.c").exists()
  result = _run(checkout, "update_gkeyll.sh")
  assert result.returncode == 0, result.stderr
  for name in expected:
    assert (workspace / "gkeyll" / name / "source.c").exists()


def test_invalid_module_does_not_obtain_checkout(checkout):
  workspace, _, env = checkout
  env["POSTGKYL_GKEYLL_MODULE"] = "gyrokinetics"
  result = _run(checkout)
  assert result.returncode != 0
  assert "POSTGKYL_GKEYLL_MODULE" in result.stderr
  assert not (workspace / "gkeyll").exists()


def test_build_script_uses_configuration_file(checkout):
  workspace, _, env = checkout
  (workspace / "postgkyl-build.toml"
   ).write_text('[gkeyll]\nmode = "existing"\ndirectory = "/unused"\n')
  result = _run(checkout)
  assert result.returncode != 0
  assert "rebuilt by their owner" in result.stderr
  assert not (workspace / "gkeyll").exists()
  config = workspace / "alternate.toml"
  config.write_text('[gkeyll]\nmode = "managed"\nmodule = "core"\n')
  env["POSTGKYL_BUILD_CONFIG"] = str(config)
  result = _run(checkout)
  assert result.returncode == 0, result.stderr
  assert (workspace / "gkeyll/build/core/libg0core.so").is_file()


def test_updater_leaves_existing_gkeyll_alone(checkout):
  workspace, _, env = checkout
  shutil.copyfile(ROOT / "scripts/update_pgkyl.sh",
                  workspace / "scripts/update_pgkyl.sh")
  (workspace / "postgkyl-build.toml"
   ).write_text('[gkeyll]\nmode = "existing"\ndirectory = "/unused"\n')
  binary_dir = workspace / "commands"
  binary_dir.mkdir()
  git = binary_dir / "git"
  git.write_text('#!/bin/sh\nprintf "%s\\n" "$@" >> "$COMMAND_LOG"\n')
  git.chmod(0o755)
  python = binary_dir / "python"
  python.write_text('#!/bin/sh\ncase "$1" in\n'
                    '  */gkeyll_config.py) exec "$REAL_PYTHON" "$@" ;;\n'
                    '  *) printf "%s\\n" "$@" >> "$COMMAND_LOG" ;;\nesac\n')
  python.chmod(0o755)
  env.update(PATH=str(binary_dir) + os.pathsep + env["PATH"],
             PYTHON=str(python),
             REAL_PYTHON=sys.executable,
             COMMAND_LOG=str(workspace / "commands.log"))
  result = subprocess.run(
      ["bash", str(workspace / "scripts/update_pgkyl.sh")],
      env=env,
      capture_output=True,
      text=True,
      timeout=20)
  assert result.returncode == 0, result.stderr
  calls = (workspace / "commands.log").read_text().splitlines()
  assert calls[:4] == ["-C", str(workspace), "pull", "--ff-only"]
  assert calls[4:9] == ["-m", "pip", "install", "--editable", str(workspace)]
  assert not (workspace / "gkeyll").exists()
