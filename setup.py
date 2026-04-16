"""AutoQRA package installer."""

from setuptools import find_packages, setup

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    install_requires = [
        line.strip() for line in f if line.strip() and not line.startswith("#")
    ]

setup(
    name="autoqra",
    version="1.0.0",
    description=(
        "AutoQRA: Joint per-layer quantization bit-width and LoRA rank "
        "optimization under a memory budget."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AutoQRA Authors",
    license="MIT",
    url="https://github.com/harrysyz99/autoqra",
    packages=find_packages(exclude=("tests", "tests.*", "scripts", "examples")),
    include_package_data=True,
    install_requires=install_requires,
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "autoqra=autoqra.cli:main",
        ],
    },
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
