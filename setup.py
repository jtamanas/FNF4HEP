from os import path
from setuptools import find_packages, setup


setup(
    name="fnf4hep",
    version='0.0.0a',
    description="Fair Normalizing Flows for HEP",
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
