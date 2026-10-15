"""Contracts of the native setuptools command, independent of compilation."""

import runpy
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import setuptools
from setuptools.dist import Distribution

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def default_build_config(monkeypatch, tmp_path):
  config = tmp_path / "default-build.toml"
  config.write_text('[gkeyll]\nmode = "managed"\nmodule = "core"\n')
  monkeypatch.setenv("POSTGKYL_BUILD_CONFIG", str(config))


@pytest.fixture
def setup_config(monkeypatch):
  config = {}
  monkeypatch.delenv("POSTGKYL_GKEYLL_DIR", raising=False)
  monkeypatch.delenv("POSTGKYL_GKEYLL_MODULE", raising=False)
  monkeypatch.setattr(setuptools, "setup",
                      lambda **kwargs: config.update(kwargs))
  monkeypatch.setenv("POSTGKYL_SKIP_GKEYLL_BUILD", "0")
  namespace = runpy.run_path(ROOT / "setup.py")
  return config, namespace


# Setuptools asks install_lib for output metadata even for PEP 660 builds.
@pytest.mark.filterwarnings(
    "ignore:setup.py install is deprecated:setuptools.warnings.SetuptoolsDeprecationWarning"
)
@pytest.mark.parametrize("editable", [False, True])
def test_native_outputs_include_library_and_provenance(setup_config, tmp_path,
                                                       editable):
  config, _ = setup_config
  distribution = Distribution({
      **config, "packages": ["postgkyl.gpython"],
      "package_dir": {
          "": "src"
      }
  })
  command = distribution.get_command_obj("build_ext")
  command.build_lib = str(tmp_path / "lib")
  command.editable_mode = editable
  command.ensure_finalized()
  outputs = command.get_outputs()
  for name in ("libg0core.so", "_build_info.json"):
    built = str(tmp_path / "lib/postgkyl/gpython" / name)
    assert built in outputs
    assert command.get_output_mapping()[built] == str(
        Path("src/postgkyl/gpython") / name)
  assert any(Path(path).name.startswith("_gpython.") for path in outputs)
  assert distribution.has_ext_modules()


def test_skip_build_produces_pure_python_distribution(monkeypatch):
  config = {}
  monkeypatch.setattr(setuptools, "setup",
                      lambda **kwargs: config.update(kwargs))
  monkeypatch.setenv("POSTGKYL_SKIP_GKEYLL_BUILD", "1")
  runpy.run_path(ROOT / "setup.py")
  assert not Distribution(config).has_ext_modules()


def test_skip_build_rejects_misspelled_value(monkeypatch):
  monkeypatch.setenv("POSTGKYL_SKIP_GKEYLL_BUILD", "true")
  with pytest.raises(ValueError, match="must be 0 or 1"):
    runpy.run_path(ROOT / "setup.py")


@pytest.mark.parametrize("headers,library", [
    ("core/zero", "build/core/libg0core.so"),
    ("include", "lib/libg0core.so"),
])
def test_external_build_is_selected_without_building(setup_config, monkeypatch,
                                                     tmp_path, headers,
                                                     library):
  directory = tmp_path / "external build"
  include = directory / headers
  core = directory / library
  include.mkdir(parents=True)
  core.parent.mkdir(parents=True)
  (include / "gkyl_gpython_core.h").touch()
  core.touch()
  monkeypatch.setenv("POSTGKYL_GKEYLL_DIR", str(directory))
  with patch("subprocess.run") as run:
    selected = setup_config[1]["_gkeyll_paths"](
        setup_config[1]["load_config"]())
  assert selected == (directory, include, core)
  run.assert_not_called()


@pytest.mark.parametrize("missing", ["header", "library", "both", "empty"])
def test_invalid_external_build_never_falls_back(setup_config, monkeypatch,
                                                 tmp_path, missing):
  include = tmp_path / "include"
  library = tmp_path / "lib"
  include.mkdir()
  library.mkdir()
  if missing == "header":
    (library / "libg0core.so").touch()
  if missing == "library":
    (include / "gkyl_gpython_core.h").touch()
  monkeypatch.setenv("POSTGKYL_GKEYLL_DIR",
                     "" if missing == "empty" else str(tmp_path))
  with patch("subprocess.run") as run:
    with pytest.raises(ValueError, match="POSTGKYL_GKEYLL_DIR"):
      setup_config[1]["_gkeyll_paths"](setup_config[1]["load_config"]())
  run.assert_not_called()


def test_default_build_uses_internal_checkout(setup_config, monkeypatch):
  _, namespace = setup_config
  plan = namespace["PLAN"]
  # Configuration changes after resolution must not change the build selection.
  monkeypatch.setenv("POSTGKYL_GKEYLL_MODULE", "pkpm")
  monkeypatch.setenv("POSTGKYL_BUILD_CONFIG", "/missing/config.toml")
  build = Mock()
  with patch.dict(namespace["_gkeyll_paths"].__globals__,
                  {"build_gkeyll": build}):
    selected = namespace["_gkeyll_paths"](plan)
  build.assert_called_once_with(plan)
  assert selected == (ROOT / "gkeyll", ROOT / "gkeyll/core/zero",
                      ROOT / "gkeyll/build/core/libg0core.so")


@pytest.mark.parametrize("module,expected", [
    ("core", ["core"]),
    ("moments", ["core", "moments"]),
    ("vlasov", ["core", "moments", "vlasov"]),
    ("gyrokinetic", ["core", "moments", "vlasov", "gyrokinetic"]),
    ("pkpm", ["core", "moments", "vlasov", "gyrokinetic", "pkpm"]),
])
def test_module_selection_links_highest_and_bundles_prerequisites(
    monkeypatch, module, expected):
  config = {}
  monkeypatch.setattr(setuptools, "setup", lambda **kw: config.update(kw))
  monkeypatch.setenv("POSTGKYL_GKEYLL_MODULE", module)
  monkeypatch.setenv("POSTGKYL_SKIP_GKEYLL_BUILD", "0")
  namespace = runpy.run_path(ROOT / "setup.py")
  assert namespace["PLAN"].modules == tuple(expected)
  monkeypatch.delenv("POSTGKYL_GKEYLL_DIR", raising=False)
  with patch.dict(namespace["_gkeyll_paths"].__globals__,
                  {"build_gkeyll": lambda plan: None}):
    _, _, core = namespace["_gkeyll_paths"](namespace["PLAN"])
  build = "build" if module == "core" else "build/postgkyl-full"
  assert core == ROOT / "gkeyll" / build / "core/libg0core.so"
  assert namespace["BUNDLE_FILES"] == tuple(
      f"libg0{name}.so" for name in expected) + ("_build_info.json", )
  extension, = config["ext_modules"]
  assert extension.libraries == [f"g0{module}"]
  assert extension.define_macros == [(f"POSTGKYL_HAVE_{name.upper()}", "1")
                                     for name in expected[1:]]


@pytest.mark.parametrize("module", ["", "gyrokinetics", "core,vlasov"])
def test_invalid_module_selection(monkeypatch, module):
  monkeypatch.setenv("POSTGKYL_GKEYLL_MODULE", module)
  with pytest.raises(ValueError, match="POSTGKYL_GKEYLL_MODULE"):
    runpy.run_path(ROOT / "setup.py")


@pytest.mark.parametrize("layout", ["checkout", "prefix"])
def test_optional_build_bundles_stack_and_cleans_stale_libraries(
    monkeypatch, tmp_path, layout):
  from setuptools.command.build_ext import build_ext

  config = {}
  monkeypatch.setattr(setuptools, "setup", lambda **kw: config.update(kw))
  monkeypatch.setenv("POSTGKYL_GKEYLL_MODULE", "moments")
  monkeypatch.setenv("POSTGKYL_GKEYLL_DIR", str(tmp_path / "producer"))
  monkeypatch.setenv("POSTGKYL_SKIP_GKEYLL_BUILD", "0")
  namespace = runpy.run_path(ROOT / "setup.py")
  producer = tmp_path / "producer"
  for module in ("core", "moments"):
    headers = producer / ("include" if layout == "prefix" else f"{module}/zero")
    library = producer / ("lib" if layout == "prefix" else f"build/{module}")
    headers.mkdir(parents=True, exist_ok=True)
    library.mkdir(parents=True, exist_ok=True)
    header = "gkyl_gpython_core.h" if module == "core" else "gkyl_gpython_moments.h"
    (headers / header).touch()
    (library / f"libg0{module}.so").write_text(module)
  command = namespace["BuildGkeyllExtension"](Distribution(config))
  command.build_lib = str(tmp_path / "output")
  command.extensions = config["ext_modules"]
  bundle = tmp_path / "output/postgkyl/gpython"
  bundle.mkdir(parents=True)
  (bundle / "libg0gyrokinetic.so").touch()
  monkeypatch.setattr(build_ext, "build_extensions", lambda self: None)
  monkeypatch.setattr("subprocess.run", lambda *a, **kw: None)
  with patch.dict(
      namespace["_build_info"].__globals__,
      {"_build_info": lambda _: {
          "gkeyll_modules": ["core", "moments"]
      }}):
    command.build_extensions()
  assert (bundle / "libg0core.so").read_text() == "core"
  assert (bundle / "libg0moments.so").read_text() == "moments"
  assert not (bundle / "libg0gyrokinetic.so").exists()
  assert (bundle / "_build_info.json").is_file()
  (library / "libg0moments.so").unlink()
  with pytest.raises(ValueError, match="missing.*libg0moments"):
    command.build_extensions()


@pytest.mark.parametrize("build_dir", ["build", "cuda-build"])
def test_configured_existing_checkout_uses_selected_build(
    setup_config, monkeypatch, tmp_path, build_dir):
  producer = tmp_path / "producer"
  headers = producer / "core/zero"
  library = producer / build_dir / "core/libg0core.so"
  headers.mkdir(parents=True)
  library.parent.mkdir(parents=True)
  (headers / "gkyl_gpython_core.h").touch()
  library.touch()
  # An explicit build-dir must win even when an installation also exists.
  (producer / "include").mkdir()
  (producer / "lib").mkdir()
  (producer / "include/gkyl_gpython_core.h").touch()
  (producer / "lib/libg0core.so").touch()
  config = tmp_path / "native.toml"
  config.write_text('[gkeyll]\nmode = "existing"\ndirectory = "producer"\n'
                    f'build-dir = "{build_dir}"\n')
  monkeypatch.setenv("POSTGKYL_BUILD_CONFIG", str(config))
  with patch("subprocess.run") as run:
    assert setup_config[1]["_gkeyll_paths"](
        setup_config[1]["load_config"]()) == (producer, headers, library)
  run.assert_not_called()


def test_installed_prefix_precedes_checkout_without_build_dir(
    setup_config, monkeypatch, tmp_path):
  producer = tmp_path / "gkylsoft/gkeyll"
  for header in ("include/gkyl_gpython_core.h",
                 "core/zero/gkyl_gpython_core.h"):
    path = producer / header
    path.parent.mkdir(parents=True)
    path.touch()
  for library in ("lib/libg0core.so", "build/core/libg0core.so"):
    path = producer / library
    path.parent.mkdir(parents=True)
    path.touch()
  config = tmp_path / "native.toml"
  config.write_text('[gkeyll]\nmode = "existing"\n'
                    'directory = "gkylsoft/gkeyll"\n')
  monkeypatch.setenv("POSTGKYL_BUILD_CONFIG", str(config))
  with patch("subprocess.run") as run:
    assert setup_config[1]["_gkeyll_paths"](
        setup_config[1]["load_config"]()) == (producer, producer / "include",
                                              producer / "lib/libg0core.so")
  run.assert_not_called()


def test_config_module_and_environment_precedence(monkeypatch, tmp_path):
  config = tmp_path / "native.toml"
  config.write_text('[gkeyll]\nmode = "managed"\nmodule = "pkpm"\n')
  monkeypatch.setenv("POSTGKYL_BUILD_CONFIG", str(config))
  monkeypatch.delenv("POSTGKYL_GKEYLL_MODULE", raising=False)
  monkeypatch.delenv("POSTGKYL_GKEYLL_DIR", raising=False)
  resolve = runpy.run_path(ROOT / "scripts/gkeyll_config.py")["load_config"]
  assert resolve().module == "pkpm"
  monkeypatch.setenv("POSTGKYL_GKEYLL_MODULE", "moments")
  monkeypatch.setenv("POSTGKYL_GKEYLL_DIR", str(tmp_path))
  result = resolve()
  assert result.module == "moments"
  assert result.mode == "existing"
  assert result.directory == tmp_path


@pytest.mark.parametrize("settings", [
    'mode = "unknown"',
    'mode = "existing"',
    'directory = "/tmp/gkeyll"',
    'build-dir = "cuda-build"',
    'modul = "pkpm"',
    'module = 42',
])
def test_invalid_build_configuration_fails(monkeypatch, tmp_path, settings):
  config = tmp_path / "native.toml"
  config.write_text("[gkeyll]\n" + settings + "\n")
  monkeypatch.setenv("POSTGKYL_BUILD_CONFIG", str(config))
  monkeypatch.delenv("POSTGKYL_GKEYLL_MODULE", raising=False)
  monkeypatch.delenv("POSTGKYL_GKEYLL_DIR", raising=False)
  resolve = runpy.run_path(ROOT / "scripts/gkeyll_config.py")["load_config"]
  with pytest.raises(ValueError):
    resolve()
