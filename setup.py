from setuptools import find_packages, setup


setup(name='scikit-keras',
      packages=find_packages(),
      version='0.1.9',
      description='Scikit-learn-compatible Keras models',
      author='David Diaz Vico',
      author_email='david.diaz.vico@outlook.com',
      url='https://github.com/daviddiazvico/scikit-keras',
      download_url='https://github.com/daviddiazvico/scikit-keras/archive/v0.1.9.tar.gz',
      keywords=['scikit-learn', 'keras'],
      classifiers=['Intended Audience :: Science/Research',
                   'Intended Audience :: Developers',
                   'Topic :: Software Development',
                   'Topic :: Scientific/Engineering',
                   'Programming Language :: Python',
                   'Programming Language :: Python :: 3',
                   'Programming Language :: Python :: 3.5',
                   'Programming Language :: Python :: 3.6'],
      install_requires=['scikit-learn', 'keras'])
