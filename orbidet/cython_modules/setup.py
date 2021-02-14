# run with python setup.py build_ext --inplace



from setuptools import setup
from Cython.Build import cythonize
import numpy

import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

# setup(
#     ext_modules=cythonize("diff_eqs.pyx",compiler_directives={'language_level' : "3"}),
#     include_dirs=[numpy.get_include()]
# )

# setup(
#     ext_modules=cythonize("filter_eqs.pyx",compiler_directives={'language_level' : "3"},annotate=True),
#     include_dirs=[numpy.get_include()]
# )

# setup(
#     ext_modules=cythonize("semi_analytical.pyx",compiler_directives={'language_level' : "3"},annotate=True),
#     include_dirs=[numpy.get_include()]
# )
