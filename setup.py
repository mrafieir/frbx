# Usage: python3 setup.py install --prefix=$HOME

from setuptools import setup

setup(name='frbx',
      packages=['frbx'],
      install_requires=['numpy',
                        'scipy',
                        'astropy',
                        'matplotlib',
                        'camb',
                        'pyfftw',
                        'healpy',
                        'h5py',
                        'multiprocess',
                        'pathos',
                        'handout',
                        'corner',
                        'requests',
                        'chime_frb_api',
                        'pymangle',
                        'pytz'])
