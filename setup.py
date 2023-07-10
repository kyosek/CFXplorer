from io import open
from os import path

from setuptools import find_packages, setup

DESCRIPTION = (
    "FOCUS is a python package for "
    "generating counterfactual explanations for a tree-based model"
)


# get __version__ from _version.py
ver_file = path.join("focus", "version.py")
with open(ver_file) as f:
    exec(f.read())

this_directory = path.abspath(path.dirname(__file__))


# read the contents of README.rst
def readme():
    with open(path.join(this_directory, "README.rst"), encoding="utf-8") as f:
        return f.read()


# read the contents of requirements.txt
with open(path.join(this_directory, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().splitlines()

setup(
    name="focus-cfe",
    version=__version__,
    author="Kyosuke Morita",
    author_email="kq441morita@gmail.com",
    description=DESCRIPTION,
    long_description=readme(),
    packages=find_packages(),
    install_requires=[],
    keywords=[
        "python",
        "counterfactual explanation",
        "binary classification",
        "machine learning",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
)
