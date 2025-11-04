#!/usr/bin/env python3
"""
Setup script for ResNet50_test
Educational Research Project - NOT for clinical use

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
"""

from setuptools import setup, find_packages
import os

# Constants
PROJECT_NAME = "resnet50-test"
VERSION = "1.0.0"
GITHUB_URL = "https://github.com/your-org/ResNet50_Test"
AUTHOR = "Research Team"
AUTHOR_EMAIL = "research@example.com"
DESCRIPTION = "Breast Density Exploration via ResNet-50 Embeddings"

# Classifiers
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Medical Science Apps.",
    "Topic :: Scientific/Engineering :: Image Processing",
    "Topic :: Scientific/Engineering :: Information Analysis",
]

# Keywords
KEYWORDS = [
    "medical-imaging",
    "deep-learning",
    "breast-density",
    "resnet50",
    "clustering",
    "semi-supervised-learning",
    "research",
    "education",
    "bi-rads",
    "mammography",
    "dicom",
    "embeddings",
    "unsupervised-learning",
]

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name=PROJECT_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=read_readme() + """

⚠️ DISCLAIMER: This is an EDUCATIONAL RESEARCH project.
It must NOT be used for clinical or medical diagnostic purposes.
No medical decision should be based on these results.

This project is intended exclusively for research and education purposes
in medical imaging processing and machine learning.
""",
    long_description_content_type="text/markdown",
    url=GITHUB_URL,
    project_urls={
        "Bug Reports": f"{GITHUB_URL}/issues",
        "Source": GITHUB_URL,
        "Documentation": f"{GITHUB_URL}/docs",
        "Research Paper": f"{GITHUB_URL}/reports",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=CLASSIFIERS,
    python_requires=">=3.11",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "black>=23.7.0",
            "ruff>=0.0.280",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "docs": [
            "sphinx>=7.1.0",
            "sphinx-rtd-theme>=1.3.0",
            "nbsphinx>=0.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "resnet50-test=scripts.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md", "*.txt"],
    },
    zip_safe=False,
    keywords=KEYWORDS,
    license="MIT",
    platforms=["any"],
    provides=["resnet50_test"],
    requires_python=">=3.11",
    data_files=[
        ("configs", ["configs/base.yaml"]),
        ("docs", ["README.md", "QUICKSTART.md"]),
    ],
    scripts=[
        "scripts/check_status.py",
    ],
    test_suite="tests",
    cmdclass={},
    options={
        "build_sphinx": {
            "source_dir": "docs",
            "build_dir": "docs/_build",
            "builder": "html",
        },
    },
    metadata_version="2.1",
)
