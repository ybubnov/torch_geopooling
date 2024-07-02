# Project information
project = "Torch Geopooling"
author = "Yakau Bubnou"
copyright = f"2024-present, {author}"


# General configuration
need_sphinx = "4.4"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

# The documentation is in English language.
language = "en"

# Change the code syntax highlighting style to "default" (the same is used in
# Python standard library documentation).
pygments_style = "default"


# Options for HTML output
html_theme = "furo"
templates_path = ["_templates"]
