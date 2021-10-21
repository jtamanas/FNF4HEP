from os import path
from setuptools import find_packages, setup

exec(open("nflows/version.py").read())

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="fnf4hep",
    version=__version__,
    description="Fair Normalizing Flows for HEP",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/jtamanas/FNF4HEP",
    license="MIT",
    install_requires=[
        "matplotlib",
        "numpy",
        "tensorboard",
        "torch",
        "tqdm",
    ],
    extras_requires={
        "dev": [
            "autoflake",
            "black",
            "flake8",
            "isort",
            "pytest",
            "pyyaml",
            "torchtestcase",
        ],
    },
    dependency_links=[],
)
