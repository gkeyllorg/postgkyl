"""Standalone preview of the same source subtree staged into gkyl-doc."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "_ext"))
project = "Postgkyl"
extensions = [
    "sphinx.ext.autodoc", "sphinx.ext.napoleon", "myst_parser", "postgkyl_docs"
]
html_theme = "furo"
root_doc = "index"
exclude_patterns = ["_inputs/**", "_ext/**"]
autodoc_typehints = "none"
nitpicky = False
