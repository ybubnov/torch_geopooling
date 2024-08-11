import inspect
import sys
import warnings
from contextlib import suppress
from pathlib import Path
from typing import Optional, List, Tuple, Any

import torch_geopooling


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
    "sphinx.ext.linkcode",
    "nbsphinx",
]

# The documentation is in English language.
language = "en"

# Change the code syntax highlighting style to "default" (the same is used in
# Python standard library documentation).
pygments_style = "default"


# Options for HTML output
html_theme = "furo"
templates_path = ["_templates"]


def linkcode_sourcefile(obj: Any) -> Optional[str]:
    fn: Optional[str] = None
    try:
        fn = inspect.getsourcefile(inspect.unwrap(obj))  # type: ignore
    except TypeError:
        # Consider object as a property.
        with suppress(AttributeError, TypeError):
            fn = inspect.getsourcefile(inspect.unwrap(obj.fget))  # type: ignore
    return fn


def linkcode_sourcelines(obj: Any) -> Tuple[Optional[List[str]], Optional[int]]:
    lines: Optional[List[str]] = None
    lineno: Optional[int] = None
    try:
        lines, lineno = inspect.getsourcelines(obj)  # type: ignore
    except TypeError:
        # Consider object as a property.
        with suppress(AttributeError, TypeError):
            lines, lineno = inspect.getsourcelines(obj.fget)  # type: ignore
    except OSError:
        pass
    return lines, lineno


# Based on numpy doc/source/conf.py
def linkcode_resolve(domain, info) -> Optional[str]:
    """Determine the GitHub URL corresponding to the Python object."""
    if domain != "py":
        return None

    submodule = sys.modules.get(info["module"])
    if submodule is None:
        return None

    obj = submodule
    for part in info["fullname"].split("."):
        try:
            with warnings.catch_warnings():
                # Accessing deprecated objects will generate noisy warnings
                warnings.simplefilter("ignore", FutureWarning)
                obj = getattr(obj, part)
        except AttributeError:
            return None

    fn = linkcode_sourcefile(obj)
    if not fn:
        return None

    lines, lineno = linkcode_sourcelines(obj)

    linespec = ""
    if lines and lineno:
        linespec = f"#L{lineno}-L{lineno + len(lines) - 1}"

    fnspec = Path(fn).relative_to(Path(torch_geopooling.__file__).parent)

    return (
        f"https://github.com/ybubnov/torch_geopooling/blob/"
        f"v{torch_geopooling.__version__}/torch_geopooling/{fnspec}{linespec}"
    )
