from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
 
setup(ext_modules=[Extension("template_matching",
                             ["template_matching.pyx"],
                             language="c++",),
                   Extension("fastremap",
                             ["fastremap.pyx"],
                             extra_compile_args=['-fopenmp'],
                             extra_link_args=['-fopenmp'],
                             language="c",)],
      cmdclass = {'build_ext': build_ext})
