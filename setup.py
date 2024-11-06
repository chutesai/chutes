import os
from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md")) as infile:
    long_description = infile.read()

setup(
    name="chutes",
    version="0.0.17",
    description="Chtues development kit and CLI.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jondurbin/chutes",
    author="Jon Durbin",
    license_expression="MIT",
    packages=find_packages(),
    install_requires=[
        "aiohttp[speedups]>=3.10,<4",
        "backoff>=2.2,<3",
        "requests>=2.32",
        "loguru==0.7.2",
        "fastapi>=0.115",
        "uvicorn>=0.32.0",
        "pydantic>=2.9,<3",
        "pybase64>=1.4.0",
        "orjson>=3.10",
        "fickling==0.1.3",
        "setuptools>=0.75",
        "substrate-interface>=1.7.11",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "wheel",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
    ],
    scripts=["bin/chutes"],
)
