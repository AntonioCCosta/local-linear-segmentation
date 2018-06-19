from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name="LLSA_calculations",
    ext_modules=cythonize('LLSA_calculations.pyx'),
    include_dirs=[numpy.get_include()]
)
