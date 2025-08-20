#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for CG-HCAN package.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="cg-hcan",
    version="1.0.0",
    author="Xiong et al.",
    author_email="userxyb@whu.edu.cn",
    description="Category-Guided Hierarchical Cross-Attention Network for Remote Sensing Segmentation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/WHUXyb/CG-HCAN",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "flake8>=3.9.0",
            "black>=21.0.0",
            "pre-commit>=2.15.0",
        ],
        "clip": [
            "clip-by-openai",
        ],
        "visualization": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cg-hcan-train=train:main",
            "cg-hcan-infer=infer:main",
            "cg-hcan-quick=scripts.quick_inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "cg_hcan": [
            "configs/*.py",
            "scripts/*.sh",
            "scripts/*.bat",
            "docs/*.md",
        ],
    },
    zip_safe=False,
    keywords=[
        "remote sensing",
        "image segmentation", 
        "hierarchical classification",
        "deep learning",
        "computer vision",
        "satellite imagery",
        "land use classification",
        "pytorch",
        "clip",
        "attention mechanism",
    ],
    project_urls={
        "Bug Reports": "https://github.com/WHUXyb/CG-HCAN/issues",
        "Source": "https://github.com/WHUXyb/CG-HCAN",
        "Documentation": "https://github.com/WHUXyb/CG-HCAN/blob/main/README.md",
        "Paper": "https://arxiv.org/abs/xxxx.xxxxx",
    },
)
