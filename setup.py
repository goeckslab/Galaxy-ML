import subprocess
from distutils.core import setup
from os.path import dirname, join, realpath

from setuptools import find_packages

try:
    import Cython
except ImportError:
    subprocess.run("pip install Cython", shell=True, check=True)

try:
    import numpy as np
except ImportError:
    subprocess.run("pip install numpy>=1.16.2",
                   shell=True, check=True)
    import numpy as np

from Cython.Build import cythonize
from Cython.Distutils import build_ext
from Cython.Distutils.extension import Extension

import galaxy_ml


VERSION = galaxy_ml.__version__
PROJECT_ROOT = dirname(realpath(__file__))

with open(join(PROJECT_ROOT, 'requirements.txt'), 'r') as f:
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

long_description = """

This library contains APIs for
[Galaxy](https://github.com/galaxyproject/galaxy)
machine learning tools(Galaxy-ML).

Galaxy-ML is a web machine learning end-to-end pipeline building
framework, with special support to biomedical data. Under the
management of unified scikit-learn APIs, cutting-edge machine
learning libraries (scikit-learn, tensorflow, mlxtend, imbalanced-learn,
and more) are combined together to provide thousands
of different pipelines suitable for various needs. In the form
of Galalxy tools, Galaxy-ML provides scalabe, reproducible and
transparent machine learning computations.

This library and tools are hosted at
https://github.com/geockslab/Galaxy-ML.

The documentation can be found at
https://goeckslab.github.io/Galaxy-ML/

"""

setup(
    name='Galaxy-ML',
    version=VERSION,
    description='Galaxy Machine Learning Library',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/goeckslab/Galaxy-ML/',
    packages=find_packages(
        exclude=['docs', 'tests*', 'test-data*', 'tools*']
    ),
    package_data={
        '': ['README.md', 'requirements.txt'],
    },
    include_package_data=True,
    install_requires=install_reqs,
    extras_require={'docs': ['mkdocs']},
    platforms='any',
    ext_modules=cythonize(ext_modules),
    cmdclass=cmdclass,
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
