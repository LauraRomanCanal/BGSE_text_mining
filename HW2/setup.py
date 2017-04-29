from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import sys

include_gsl_dir = sys.exec_prefix+"\\include"
lib_gsl_dir = sys.exec_prefix+"\\lib"
ext = Extension("LDA", ["LDA.pyx"],
include_dirs=[numpy.get_include(),
include_gsl_dir],
library_dirs=[lib_gsl_dir],
libraries=["gsl","gslcblas","m"]
)

setup(name = "samplers",
	ext_modules=[ext],
	cmdclass = {'build_ext': build_ext})
