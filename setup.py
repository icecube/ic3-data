#!/usr/bin/env python

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools

# get version number
exec(compile(open('ic3_data/__init__.py', "rb").read(),
             'ic3_data/__init__.py',
             'exec'))


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


class get_numpy_include(object):
    """Helper class to determine the pybind11 include path
    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked. """

    def __str__(self):
        import numpy
        return numpy.get_include()


def get_icecube_includes():
    """Helper function to get include paths for IceCube headers.
    """
    import os
    import glob

    # find all icecube packages in the source directory
    include_pattern = os.path.join(os.environ['I3_SRC'], '*/public')
    include_dirs = glob.glob(include_pattern)
    include_dirs.append(os.path.join(os.environ['I3_SRC'],
                                     'cmake/tool-patches/common/'))

    # For parasitic metaprojects, the I3_SRC directory will only contain
    # packages that were added on top of the host metaproject.
    # In this case we need to scan the source directory of the host as well.
    # We can obtain the host metaproject by checking the symlinks of the
    # resources directories in the I3_BUILD directory.

    # Gather source directories
    resource_pattern = os.path.join(os.environ['I3_BUILD'], '*/resources')
    resource_dirs = glob.glob(resource_pattern)
    source_dirs = set([os.readlink(d).replace('resources', 'public')
                       for d in resource_dirs if os.path.islink(d)])
    include_dirs.extend(source_dirs)

    if 'SROOT' in os.environ:
        include_dirs.append(os.path.join(os.environ['SROOT'], 'include/'))

    # get include paths for GitHub icetray builds in cvmfs
    # get directory parts before and after software system
    dir_parts = os.environ['I3_SRC'].split('/metaprojects/')
    if len(dir_parts) > 1:
        # remove the specified software system
        prefix = os.path.dirname(dir_parts[0])
        cvmfs_src_dir = os.path.join(prefix, 'metaprojects', dir_parts[1])
        include_pattern = os.path.join(cvmfs_src_dir, '*/public')
        include_dirs.extend(glob.glob(include_pattern))
        include_dirs.append(os.path.join(
            cvmfs_src_dir, 'cmake/tool-patches/common/'))

    return include_dirs


def get_boost_include_list():
    """Helper function to get combine paths needed for the boost python
    extension.
    """
    include_dirs = get_icecube_includes()
    include_dirs.append(get_numpy_include())
    return include_dirs


def get_i3_lib_list():
    """Helper function to get directories for libraries
    """
    import os
    i3_lib_list = []
    if 'SROOT' in os.environ:
        i3_lib_list.append(os.path.join(os.environ['SROOT'], 'lib/'))
    i3_lib_list.append(os.path.join(os.environ['I3_BUILD'], 'lib/'))
    return i3_lib_list


def get_boost_libraries():
    """Helper function to get a list of libraries to link against
    """
    import sys
    if sys.version_info.major >= 3:
        # python 3 libraries
        suffix = '{}{}'.format(sys.version_info.major, sys.version_info.minor)
        libraries = [
            'boost_python'+suffix, 'boost_numpy'+suffix, 'phys-services',
        ]
    else:
        # python 2 libraries
        libraries = ['boost_python', 'phys-services']
    return libraries


ext_modules = [
    Extension(
        'ic3_data.ext_pybind11',
        ['ic3_data_ext/ext_pybind11.cpp'],
        include_dirs=[
            # Path to pybind11 headers
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++'
    ),
    Extension(
        'ic3_data.ext_boost',
        ['ic3_data_ext/ext_boost.cpp'],
        libraries=get_boost_libraries(),
        library_dirs=get_i3_lib_list(),
        include_dirs=get_boost_include_list(),
        language='c++'
    ),
]


# As of Python 3.6, CCompiler has a `has_flag` method.
# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14] compiler flag.
    The c++14 is prefered over c++11 (when it is available).
    """
    if has_flag(compiler, '-std=c++14'):
        return '-std=c++14'
    elif has_flag(compiler, '-std=c++11'):
        return '-std=c++11'
    else:
        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }

    if sys.platform == 'darwin':
        c_opts['unix'] += ['-stdlib=libc++', '-mmacosx-version-min=10.7']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        if ct == 'unix':
            opts.append(
                '-DVERSION_INFO="%s"' % self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append(
                '/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
        build_ext.build_extensions(self)


setup(name='ic3_data',
      version=__version__,
      description='Creates DNN input data for IceCube',
      long_description='',
      author='Mirco Huennefeld',
      author_email='mirco.huennefeld@tu-dortmund.de',
      url='https://github.com/mhuen/ic3-data',
      packages=setuptools.find_packages(),
      install_requires=['pybind11>=2.2', 'numpy', 'click', 'pyyaml',
                        ],
      ext_modules=ext_modules,
      cmdclass={'build_ext': BuildExt},
      zip_safe=False,
      )
