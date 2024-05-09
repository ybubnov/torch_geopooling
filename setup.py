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
        print(info)
        matches = re.findall(r"^ATen parallel backend: ([\ \w]+)$", info, flags=re.MULTILINE)
        if len(matches) == 0:
            return None

        return cls(matches.pop())


class BuildExtension(cpp_extension.BuildExtension):
    def build_extensions(self):
        parallel_backend = TorchParallelBackend.library_backend()

        for extension in self.extensions:
            if parallel_backend == TorchParallelBackend.OPENMP:
                self._add_compile_flag(extension, "-DAT_PARALLEL_OPENMP")
            elif parallel_backend == TorchParallelBackend.NATIVE:
                self._add_compile_flag(extension, "-DAT_PARALLEL_NATIVE")
            elif parallel_backend == TorchParallelBackend.NATIVE_TBB:
                self._add_compile_flag(extension, "-DAT_PARALLEL_NATIVE_TBB")

        super().build_extensions()


setup(
    package_dir={"torch_geopooling": "torch_geopooling"},
    packages=["torch_geopooling"],
    ext_modules=[
        cpp_extension.CppExtension(
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
    cmdclass={"build_ext": BuildExtension},
)
