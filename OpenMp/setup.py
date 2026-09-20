from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "k2means_omp",
        ["k2means_omp.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs = [numpy.get_include()]),
]

setup(
    name='k2means_omp',
    ext_modules=cythonize(ext_modules),
)
