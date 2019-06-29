import subprocess
try:
    import Cython
except ImportError:
    subprocess.run("pip install Cython", shell=True, check=True)

import numpy as np
from os.path import realpath, dirname, join
from setuptools import find_packages
from distutils.core import setup
from Cython.Distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import galaxy_ml


VERSION = galaxy_ml.__version__
PROJECT_ROOT = dirname(realpath(__file__))

REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

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
      url='https://github.com/goeckslab/Galaxy-ML/',
      packages=find_packages(),
      package_data={
        '': ['README.md',
            'requirements.txt']
      },
      include_package_data=True,
      install_requires=install_reqs,
      extras_require={'docs': ['mkdocs']},
      platforms='any',
      ext_modules=cythonize(ext_modules),
      cmdclass=cmdclass
)

