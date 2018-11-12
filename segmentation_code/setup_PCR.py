from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name="SPCR_calculations",
    ext_modules=cythonize('SPCR_calculations.pyx'),
    include_dirs=[numpy.get_include()]
)
