import subprocess
try:
    import Cython
except ImportError:
    subprocess.run("pip install Cython", shell=True, check=True)

try:
    import numpy as np
except ImportError:
    subprocess.run("pip install numpy>=1.16.2",
                   shell=True, check=True)

from os.path import realpath, dirname, join
from setuptools import find_packages
from distutils.core import setup
from Cython.Distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
import galaxy_ml


VERSION = galaxy_ml.__version__
PROJECT_ROOT = dirname(realpath(__file__))

with open(join(PROJECT_ROOT, 'requirements.txt'), 'r') as f:
    install_reqs = f.read().splitlines()

with open(join(PROJECT_ROOT, 'README.md'), 'r') as fh:
    long_description = fh.read()

genome_module = Extension(
    "galaxy_ml.externals.selene_sdk.sequences._sequence",
    ["galaxy_ml/externals/selene_sdk/sequences/_sequence.pyx"],
    include_dirs=[np.get_include()])

genomic_features_module = Extension(
    "galaxy_ml.externals.selene_sdk.targets._genomic_features",
    ["galaxy_ml/externals/selene_sdk/targets/_genomic_features.pyx"],
    include_dirs=[np.get_include()])

ext_modules = [genome_module, genomic_features_module]
cmdclass = {'build_ext': build_ext}

setup(name='Galaxy-ML',
      version=VERSION,
      description='Galaxy Machine Learning Library',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/goeckslab/Galaxy-ML/',
      packages=find_packages(exclude=['tests*', 'test-data*']),
      package_data={
        '': ['README.md',
            'requirements.txt']
      },
      include_package_data=True,
      install_requires=install_reqs,
      extras_require={'docs': ['mkdocs']},
      platforms='any',
      ext_modules=cythonize(ext_modules),
      cmdclass=cmdclass,
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ]
)

