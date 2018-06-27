#!/usr/bin/env python
# encoding: utf-8
# Guillaume HEUSCH <guillaume.heusch@idiap.ch>
# Tue 31 May 12:00:17 CEST 2016

from setuptools import setup, dist
dist.Distribution(dict(setup_requires=['bob.extension']))

from bob.extension.utils import load_requirements, find_packages
install_requires = load_requirements()

# The only thing we do in this file is to call the setup() function with all
# parameters that define our package.
setup(

  name='bob.rppg.base',
  version=open("version.txt").read().rstrip(),
  description="Baseline Algorithms for Remote Photoplethysmography (rPPG)",
  url='https://gitlab.idiap.ch/bob/bob.rppg.base',
  license='GPLv3',
  author='Guillaume Heusch',
  author_email='guillaume.heusch@idiap.ch',
  long_description=open('README.rst').read(),
  keywords = "bob, physiological signals, remote photoplethysmography, ppg",

  # This line is required for any distutils based packaging.
  packages=find_packages(),
  include_package_data=True,
  zip_safe=False,

  install_requires=install_requires,

  entry_points={
    'console_scripts': [
      'bob_rppg_cvpr14_extract_face_and_bg_signals.py = bob.rppg.cvpr14.script.extract_face_and_bg_signals:main',
      'bob_rppg_cvpr14_video2skin.py = bob.rppg.cvpr14.script.video2skin:main',
      'bob_rppg_cvpr14_illumination.py = bob.rppg.cvpr14.script.illumination_rectification:main',
      'bob_rppg_cvpr14_motion.py = bob.rppg.cvpr14.script.motion_elimination:main',
      'bob_rppg_cvpr14_filter.py = bob.rppg.cvpr14.script.filter:main',
      'bob_rppg_chrom_pulse.py = bob.rppg.chrom.script.extract_pulse:main',
      'bob_rppg_chrom_pulse_from_mask.py = bob.rppg.chrom.script.extract_pulse_from_mask:main',
      'bob_rppg_ssr_pulse.py = bob.rppg.ssr.script.spatial_subspace_rotation:main',
      'bob_rppg_ssr_pulse_from_mask.py = bob.rppg.ssr.script.ssr_from_mask:main',
      'bob_rppg_base_get_heart_rate.py = bob.rppg.base.script.frequency_analysis:main',
      'bob_rppg_base_compute_performance.py = bob.rppg.base.script.compute_performance:main',
      ],
    },

  classifiers = [
    'Framework :: Bob',
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Natural Language :: English',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Software Development :: Libraries :: Python Modules',
    ],

  )
