from setuptools import setup
import os, sys

from motmot.utils.utils import get_svnversion_persistent
version_str = '0.2.dev%(svnversion)s'
version = get_svnversion_persistent(
    os.path.join('adskalman','version.py'),
    version_str)

setup(name='adskalman',
      version=version,
      description='Kalman filtering routine',
      author_email='strawman@astraw.com',
      zip_safe=True,
      packages = ['adskalman'],
      package_data={'adskalman':['table*.csv',
                                 '*.mat'
                                 ]},
      )
