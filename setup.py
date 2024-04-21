from pathlib import Path
from setuptools import setup, Extension
from torch.utils import cpp_extension


setup(
    name="torch_geopooling",
    version="1.0.0",
    description="Geopooling modules for PyTorch framework",

    url="https://github.com/ybubnov/torch_geopooling",
    author="Yakau Bubnou",
    author_email="girokompass@gmail.com",

    package_dir={"torch_geopooling": "torch_geopooling"},
    packages=["torch_geopooling"],

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries",
        "Topic :: System :: Archiving :: Compression",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
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
