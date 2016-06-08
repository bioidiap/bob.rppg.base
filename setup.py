#!/usr/bin/env python
# encoding: utf-8
# Guillaume HEUSCH <guillaume.heusch@idiap.ch>
# Tue 31 May 12:00:17 CEST 2016

from setuptools import setup, find_packages

# Define package version
version = open("version.txt").read().rstrip()

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

  name='bob.rppg.base',
  version=version,
  description="Algorithms for Remote PPG",
  url='https://gitlab.idiap.ch/biometric/bob.rppg.base',
  license='GPLv3',
  author='Guillaume Heusch',
  author_email='guillaume.heusch@idiap.ch',
  long_description=open('README.rst').read(),

  # This line is required for any distutils based packaging.
  packages=find_packages(),
  include_package_data=True,
  zip_safe=True,

  install_requires=[
      "setuptools",
      "bob.io.base",
      "bob.ip.facedetect",
      "bob.ip.skincolorfilter",
      "bob.db.hci_tagging",
      "bob.db.cohface",
      "docopt",
      "numpy",
      "scipy",
      "matplotlib",
      ],

  entry_points={
    'console_scripts': [
        'cvpr14_extract_signals.py = bob.rppg.cvpr14.script.extract_signals:main',
        'cvpr14_illumination.py = bob.rppg.cvpr14.script.illumination_rectification:main',
        'cvpr14_motion.py = bob.rppg.cvpr14.script.motion_elimination:main',
        'cvpr14_filter.py = bob.rppg.cvpr14.script.filter:main',
        'rppg_get_heart_rate.py = bob.rppg.base.script.frequency_analysis:main',
        'rppg_compute_performance.py = bob.rppg.base.script.compute_performance:main'
        ],
    },

)