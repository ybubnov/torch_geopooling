from setuptools.build_meta import *

from .build_ext import BuildExtBackend


__all__ = [
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_wheel",
    "build_wheel",
    "build_sdist",
    "get_requires_for_build_editable",
    "prepare_metadata_for_build_editable",
    "build_editable",
    "__legacy__",
    "SetupRequirementsError",
]


_build_wheel = build_wheel

def build_wheel(*args, **kwargs):
    BuildExtBackend.prepare_build_environment()
    return _build_wheel(*args, **kwargs)


_build_sdist = build_sdist

def build_sdist(*args, **kwargs):
    BuildExtBackend.prepare_build_environment()
    return _build_sdist(*args, **kwargs)


_build_editable = build_editable

def build_editable(*args, **kwargs):
    BuildExtBackend.prepare_build_environment()
    return _build_editable(*args, **kwargs)
