from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

extensions = [
    #Cython extensions for formatting the ray array
    Extension("cython_dtype", ["cython_dtype.pyx"], include_dirs=[numpy.get_include()]),
    Extension("cython_indexing", ["cython_indexing.pyx"], include_dirs=[numpy.get_include()]),
    Extension("cython_views", ["cython_views.pyx"], include_dirs=[numpy.get_include()]),
    Extension("cython_raw", ["cython_raw.pyx"], include_dirs=[numpy.get_include()]),
    Extension("cython_dict", ["cython_dict.pyx"], include_dirs=[numpy.get_include()]),
    Extension("cython_cloop", ["cython_cloop.pyx"], include_dirs=[numpy.get_include()]),

    #Cython extensions for calculating the interfunction
    Extension("cython_inter", ["cython_inter.pyx"], include_dirs=[numpy.get_include()])
]

setup(ext_modules=cythonize(extensions))