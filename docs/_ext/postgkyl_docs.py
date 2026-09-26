"""Copy standalone interactive figures beside their referring tutorial pages."""

from pathlib import Path
import shutil


def copy_interactive(app, exception):
  if exception is not None or app.builder.format != "html":
    return
  relative = Path(app.config.postgkyl_doc_root) / "interactive"
  source = Path(app.srcdir) / relative
  if source.is_dir():
    destination = Path(app.outdir) / relative
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.html"):
      shutil.copyfile(path, destination / path.name)


def setup(app):
  app.add_config_value("postgkyl_doc_root", ".", "html")
  app.connect("build-finished", copy_interactive)
  return {"parallel_read_safe": True, "parallel_write_safe": True}
