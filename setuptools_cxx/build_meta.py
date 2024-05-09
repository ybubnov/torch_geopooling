from setuptools import build_meta as _build_meta
from setuptools.build_meta import build_editable as _build_editable
from setuptools.build_meta import build_sdist as _build_sdist
from setuptools.build_meta import build_wheel as _build_wheel

from .build_ext import BuildExtBackend

__all__ = [
    "get_requires_for_build_sdist",  # noqa
    "get_requires_for_build_wheel",  # noqa
    "prepare_metadata_for_build_wheel",  # noqa
    "build_wheel",  # noqa
    "build_sdist",  # noqa
    "get_requires_for_build_editable",  # noqa
    "prepare_metadata_for_build_editable",  # noqa
    "build_editable",  # noqa
    "__legacy__",  # noqa
    "SetupRequirementsError",  # noqa
]


def build_wheel(*args, **kwargs):
    BuildExtBackend.prepare_build_environment()
    return _build_wheel(*args, **kwargs)


def build_sdist(*args, **kwargs):
    BuildExtBackend.prepare_build_environment()
    return _build_sdist(*args, **kwargs)


def build_editable(*args, **kwargs):
    BuildExtBackend.prepare_build_environment()
    return _build_editable(*args, **kwargs)


def __getattr__(name):
    if name == "build_wheel":
        return build_wheel
    elif name == "build_sdist":
        return build_sdist
    elif name == "build_editable":
        return build_editable
    return getattr(_build_meta, name)
