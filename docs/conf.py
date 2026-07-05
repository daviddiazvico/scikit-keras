import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "scikit-keras"
author = "David Diaz Vico"
copyright = "2017, David Diaz Vico"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True

templates_path = []

exclude_patterns = ["_build"]

html_theme = "pydata_sphinx_theme"

html_static_path = []

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
