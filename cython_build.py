from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    Extension("init_np_array", ["init_np_array.pyx"], include_dirs=[numpy.get_include()]),
    Extension("init_np_array_dtype", ["init_np_array_dtype.pyx"], include_dirs=[numpy.get_include()])
]

setup(ext_modules=cythonize(extensions))