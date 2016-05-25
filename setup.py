#!/usr/bin/env python
# encoding: utf-8
# Andre Anjos <andre.anjos@idiap.ch>
# Mon 12 Oct 2015 12:34:22 CEST

from setuptools import setup, find_packages

# Define package version
version = open("version.txt").read().rstrip()

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

  name='bob.rppg.cvpr14',
  version=version,
  description="UOULU's CVPR'14 Remote PPG Technique",
  url='https://gitlab.idiap.ch/biometric/bob.rppg.cvpr14',
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
        'cvpr14_hr.py = bob.rppg.cvpr14.script.frequency_analysis:main',
        'cvpr14_perf.py = bob.rppg.cvpr14.script.generate_results:main',
        'cvpr14_debug.py = bob.rppg.cvpr14.script.debug:main',
        ],
    },

)
