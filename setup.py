from pathlib import Path

from setuptools import setup
from torch.utils import cpp_extension

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
        cpp_extension.CppExtension(
            name="torch_geopooling._C",
            sources=[
                "src/quadpool.cc",
                "src/quadtree.cc",
                "src/tile.cc",
                "torch_geopooling/__bind__/python_module.cc",
            ],
            include_dirs=[str(Path.cwd() / "include")],
            libraries=["torch", "c10"],
            extra_compile_args=["-DFMT_HEADER_ONLY=1"],
        ),
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    tests_require=["pytest"],
    install_requires=["torch>=2.0.0"],
    setup_requires=["torch>=2.0.0"],
)
