from setuptools import setup, find_packages

setup(name = 'frbx',
      version = '1.1.0',
      description = 'Tools for simulating, forecasting and analyzing statistical cross-correlations between fast radio bursts and other cosmological sources.',
      url = 'https://github.com/mrafieir/frbx',
      author = 'Masoud Rafiei-Ravandi',
      author_email = 'mrafiei.ravandi@gmail.com',
      packages = find_packages(),
      install_requires = ['numpy',
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
                          'pytz'],
      python_requires = '>3.7')
