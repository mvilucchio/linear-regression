import setuptools
import pathlib

from linear_regression import (
    __pkgname__ as PKG_NAME,
    __version__ as VERSION,
    __author__ as AUTHOR,
    __license__ as LICENSE,
    __summary__ as SUMMARY,
    __url__ as URL,
)

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

setuptools.setup(
    name=PKG_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email="matteo.vilucchio@epfl.ch",
    description=SUMMARY,
    long_description=README,
    long_description_content_type="text/markdown",
    url=URL,
    license=LICENSE,
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "numba",
        "scipy",
    ],
)
