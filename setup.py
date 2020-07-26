#!/usr/bin/env python3

from setuptools import find_packages, setup


setup(name='scikit-keras',
      packages=find_packages(),
      version='0.2.0',
      description='Scikit-learn-compatible Keras models',
      author='David Diaz Vico',
      author_email='david.diaz.vico@outlook.com',
      url='https://github.com/daviddiazvico/scikit-keras',
      download_url='https://github.com/daviddiazvico/scikit-keras/archive/v0.2.0.tar.gz',
      keywords=['scikit-learn', 'keras'],
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.7',
                   'Programming Language :: Python :: 3.8',
                   'Programming Language :: Python :: 3.9',
                   'Programming Language :: Python :: 3.10'],
      install_requires=['scikit-learn', 'tensorflow'],
      setup_requires=['pytest-runner'],
      tests_require=['coverage', 'pytest', 'pytest-cov'],
      test_suite='tests')
