# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------

project = "pycaret"
copyright = "2026 - present, PyCaret maintainers. MIT License"
author = "pycaret maintainers"
contributors = "https://github.com/sktime/pycaret/graphs/contributors"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

linkcheck_ignore = [
    # LinkedIn blocks requests from non-browser clients.
    r"https://www\.linkedin\.com/.*",
    # Base path for dataset downloads, not a servable page.
    r"https://raw\.githubusercontent\.com/pycaret/datasets/main/",
]

linkcheck_retries = 2
linkcheck_timeout = 30


napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_mock_imports = ["setup"]
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# Sort methods by the order they are found in the source files
autodoc_member_order = "bysource"

suppress_warnings = [
    "docutils",  # FIXME: Various docstrings currently raise warnings.
]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"

html_logo = "../images/logo.png"

html_show_sourcelink = False

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/sktime/pycaret",
            "icon": "fa-brands fa-github",
        },
    ],
}

master_doc = "index"
