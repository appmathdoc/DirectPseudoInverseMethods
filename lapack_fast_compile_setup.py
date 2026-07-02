## Requires the Cython library
## $python lapack_fast_compile_setup.py build_ext --inplace

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        "lapack_fast",
        ["lapack_fast.pyx"],
        include_dirs=[np.get_include()] # Crucial for cnp.ndarray
    )
]

setup(
    name="Fast LAPACK Bridge",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"})
)