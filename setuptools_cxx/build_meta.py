# Copyright (C) 2024, Yakau Bubnou
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from setuptools import build_meta as _build_meta
from setuptools.build_meta import *  # noqa
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
