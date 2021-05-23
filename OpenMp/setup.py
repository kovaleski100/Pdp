from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "k2means_omp",
        ["2means_omp.px"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name='2means_omp',
    ext_modules=cythonize(ext_modules),
)
