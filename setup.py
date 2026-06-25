#!/usr/bin/env python3

from setuptools import find_packages, setup

setup(
    name="scikit-keras",
    packages=find_packages(),
    version="0.3.0",
    description="Scikit-learn-compatible Keras models",
    author="David Diaz Vico",
    author_email="david.diaz.vico@outlook.com",
    url="https://github.com/daviddiazvico/scikit-keras",
    download_url="https://github.com/daviddiazvico/scikit-keras/archive/v0.3.0.tar.gz",
    keywords=["scikit-learn", "keras"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
    install_requires=["keras", "scikit-learn"],
    setup_requires=["pytest-runner"],
    tests_require=["coverage", "pytest", "pytest-cov"],
    test_suite="tests",
)
