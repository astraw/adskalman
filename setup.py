from setuptools import setup
import os, sys

import adskalman.version
version = adskalman.version.__version__

setup(name='adskalman',
      version=version,
      description='Kalman filtering routine',
      author_email='strawman@astraw.com',
      license='BSD',
      zip_safe=True,
      packages = ['adskalman'],
      package_data={'adskalman':['table*.csv',
                                 '*.mat'
                                 ]},
      )
