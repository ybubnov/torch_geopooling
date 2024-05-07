from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Optional

from setuptools import setup
from torch.__config__ import parallel_info
from torch.utils import cpp_extension


class TorchParallelBackend(Enum):
    OPENMP = "OpenMP"
    NATIVE = "native thread pool"
    NATIVE_TBB = "native thread pool and TBB"

    @classmethod
    def library_backend(cls) -> Optional[TorchParallelBackend]:
        info = parallel_info()
        matches = re.findall(r"^ATen parallel backend: ([\ \w]+)$", info, flags=re.MULTILINE)
        if len(matches) == 0:
            return None

        return cls(matches.pop())


def cpp_parallel_extension(*args, **kwargs):
    backend = TorchParallelBackend.library_backend()

    compile_args = []
    if backend is TorchParallelBackend.OPENMP:
        compile_args = ["-DAT_PARALLEL_OPENMP", "-fopenmp"]
    elif backend is TorchParallelBackend.NATIVE:
        compile_args = ["-DAT_PARALLEL_NATIVE"]
    elif backend is TorchParallelBackend.NATIVE_TBB:
        compile_args = ["-DAT_PARALLEL_NATIVE_TBB"]

    extra_compile_args = kwargs.get("extra_compile_args", [])
    extra_compile_args += compile_args
    kwargs["extra_compile_args"] = extra_compile_args

    return cpp_extension.CppExtension(*args, **kwargs)


setup(
    name="torch_geopooling",
    version="1.0.0",
    description="The geospatial pooling modules for neural networks for PyTorch",
    url="https://github.com/ybubnov/torch_geopooling",
    author="Yakau Bubnou",
    author_email="girokompass@gmail.com",
    package_dir={"torch_geopooling": "torch_geopooling"},
    packages=["torch_geopooling"],
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
    ext_modules=[
        cpp_parallel_extension(
            name="torch_geopooling._C",
            sources=[
                "src/quadpool.cc",
                "src/quadtree.cc",
                "src/tile.cc",
                "torch_geopooling/__bind__/python_module.cc",
            ],
            include_dirs=[str(Path.cwd() / "include")],
            extra_compile_args=["-DFMT_HEADER_ONLY=1"],
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    tests_require=["pytest"],
    install_requires=["torch>=2.0.0"],
)
