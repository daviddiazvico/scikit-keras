#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name="scikit-keras",
    packages=find_packages(),
    version="0.4.0",
    description="Scikit-learn-compatible Keras models",
    author="David Diaz Vico",
    author_email="david.diaz.vico@outlook.com",
    license="MIT",
    url="https://github.com/daviddiazvico/scikit-keras",
    project_urls={
        "Documentation": "https://daviddiazvico.github.io/scikit-keras/",
        "Source": "https://github.com/daviddiazvico/scikit-keras",
        "Tracker": "https://github.com/daviddiazvico/scikit-keras/issues",
    },
    keywords=["keras", "scikit-learn"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    python_requires=">=3.11",
    install_requires=["keras>=3", "scikit-learn>=1.4"],
    extras_require={"test": ["coverage", "pytest", "pytest-cov"], "docs": ["sphinx>=8", "pydata-sphinx-theme>=0.16"]},
)
