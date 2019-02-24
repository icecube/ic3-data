#!/usr/bin/env python

from distutils.core import setup
exec(compile(open('version.py', "rb").read(),
             'version.py',
             'exec'))

setup(name='ic3_data',
      version=__version__,
      description='Creates DNN input data for IceCube',
      author='Mirco Huennefeld',
      author_email='mirco.huennefeld@tu-dortmund.de',
      url='https://github.com/mhuen/ic3-data',
      packages=['ic3_data'],
      install_requires=['numpy', 'click', 'pyyaml',
                        ],
      )
