import re
import numpy as np

from distutils.core import setup
from distutils.extension import Extension

from Cython.Distutils import build_ext

"""
Run:
$ python setup.py build_ext --inplace
"""

file_to_compile = 'mf_bpr_cython_epoch.pyx'

extension_name = re.sub("\.pyx", "", file_to_compile)

ext_modules = Extension(extension_name,
                        [file_to_compile],
                        extra_compile_args=['-O3'],
                        include_dirs=[np.get_include(), ],
                        )

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[ext_modules]
)
