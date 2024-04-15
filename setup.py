from pathlib import Path
from setuptools import setup, Extension
from torch.utils import cpp_extension


setup(
    name="torch_geopooling",

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
            include_dirs=[
                str(Path.cwd() / "include"),
            ],
            libraries=["fmt", "torch", "c10"],
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
