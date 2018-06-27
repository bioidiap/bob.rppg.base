.. py:currentmodule:: bob.rppg.base


Li's CVPR14 user's guide
========================

This package contains a free-software implementation of University of OULU's
CVPR'14 paper "Remote Heart Rate Measurement From Face Videos under Realistic
Conditions", [li-cvpr-2014]_::

  @inproceedings{Li:2014,
    title = {Remote Heart Rate Measurement From Face Videos Under Realistic Situations},
    author = {X. Li AND J. Chen AND G. Zhao AND M. Pietik{\"{a}}inen},
    booktitle = {2014 IEEE Conference on Computer Vision and Pattern Recognition},
    year = {2014},
  }

The algorithm to retrieve the pulse signal can be divided into several steps:

  1. Extracting relevant signals from the video sequence (i.e. skin color and background color)
  2. Correcting for global illumination.
  3. Eliminating non-rigid motion.
  4. Filtering

All the scripts rely on the usage of a configuration file, which specify the
database interface, the path where the raw data (i.e. video sequences) are stored
and various parameters.

Below you can find a (minmal) example of a configuration file.

.. code-block:: python

  import os, sys
  import bob.db.hci_tagging
  import bob.db.hci_tagging.driver

  if os.path.isdir(bob.db.hci_tagging.driver.DATABASE_LOCATION):
    dbdir = bob.db.hci_tagging.driver.DATABASE_LOCATION

  if dbdir == '':
    print("You should provide a directory where the DB is located")
    sys.exit()

  database = bob.db.hci_tagging.Database()
  protocol = 'cvpr14'

As you can see, here you should **at least** have the `database` and 
the `dbdir` parameters set.


Step 1: Extract signals from video sequences
--------------------------------------------

This scripts first load previously detected facial keypoints to build the mask 
covering the bottom region of the face (note that the keypoints are not
provided, but could be obtained using `bob.ip.dlib
<https://gitlab.idiap.ch/bob/bob.ip.dlib>`_ for instance. Once the
mask has been built, it is tracked across the whole sequence using the
methodology described in [li-cvpr-2014]_. The face is 
detected using :py:func:`bob.ip.facedetect.detect_single_face`, and the
tracking makes usage of OpenCV. 

To extract the mean green colors the face region and of
the background across the video sequences of the defined database 
in the configuration file, do the following::

  $ ./bin/bob_rppg_cvpr14_extract_face_and_bg_signals.py config.py -vv

To see the full options, including parameters and protocols, type:: 

  $ ./bin/bob_rppg_cvpr14_extract_face_and_bg_signals.py --help 

Note that you can either pass parameters through command-line, or 
by specififing them in the configuration file. Be aware that
the command-line overrides the configuration file though.

.. note::

   The execution of this script is very slow - mainly due to the face detection. 
   You can speed it up using the gridtk_ toolbox (especially, if you're at Idiap). 
   For example::

     $ ./bin/jman sub -t 3490 -- ./bin/bob_rppg_cvpr14_extract_face_and_bg_signals. config.py

   The number of jobs (i.e. 3490) is given by typing::
     
     $ ./bin/bob_rppg_cvpr14_extract_face_and_bg_signals.py config.py --gridcount


Step 2: Illumination Rectification
----------------------------------

The second step is to remove global illumination from the color signal
extracted from the face area. The background signal is filtered using
Normalized Linear Mean Square and is then removed from the face signal. To get
the rectified green signal of the face area, you should execute the following
script::

  $ ./bin/bob_rppg_cvpr14_illumination.py config.py -v

Again, parameters can be passed either through the configuration file or
the command-line


Step 3: Non rigid Motion Elimination
------------------------------------

Since motion can contaminate the pulse signal, a heuristic to supress large
variation in the signal has been implemented. It simply divides the signal
into segments, and then removes the segments for which the standard deviation is
too high. This script first computes the standard deviation in the green
channel on all the segment of all sequences. By default, the threshold is set such that 95%
of all the segments will be retained. To get the signals where large motion has
been eliminated, execute the following commands::

  $ ./bin/bob_rppg_cvpr14_motion.py config.py --save-threshold threshold.txt -vv
  $ ./bin/bob_rppg_cvpr14_motion.py config.py --load-threshold threshold.txt -vv


Step 4: Filtering
-----------------

In this step, a detrend filter is applied to the color signal. It
will remove global trends in the signal (i.e. the signal will be more
or less flat after this procedure). 
The next step is to remove (high frequency) noise to the detrended signal.
This is done using a moving-average filter, with a relatively small
window. Finally, a bandpass filter is applied to restrict the
frequencies to the range corresponding to a plausible heart-rate. To filter the
signal, you should execute the following command::

  $ ./bin/bob_rppg_cvpr14_filter.py config.py -vv

A Full Configuration File Example
---------------------------------

.. note::

   This configuration file can (and probably should) be used with all the 
   scripts mentioned above

.. code-block:: python

  import os, sys

  import bob.db.hci_tagging
  import bob.db.hci_tagging.driver

  # DATABASE
  if os.path.isdir(bob.db.hci_tagging.driver.DATABASE_LOCATION):
    dbdir = bob.db.hci_tagging.driver.DATABASE_LOCATION
  if dbdir == '':
    print("You should provide a directory where the DB is located")
    sys.exit()
  database = bob.db.hci_tagging.Database()
  protocol = 'cvpr14'

  basedir = 'li-hci-cvpr14/'

  # EXTRACT FACE AND BACKGROUND
  facedir = basedir + 'face'
  bgdir = basedir + 'bg'
  npoints = 200
  indent = 10 
  quality = 0.01
  distance = 10
  verbose = 2

  # ILLUMINATION CORRECTION
  illumdir = basedir + 'illumination'
  start = 306
  end = 2136
  step = 0.05
  length = 3

  # MOTION ELIMINATION
  motiondir = basedir + 'motion'
  seglength = 61
  cutoff = 0.05

  # FILTERING
  pulsedir = basedir + 'pulse'
  Lambda = 300
  window = 21
  framerate = 61
  order = 128

  # FREQUENCY ANALYSIS
  hrdir = basedir + 'hr'
  nsegments = 16
  nfft = 8192

  # RESULTS
  resultdir = basedir + 'results'


.. _gridtk: https://pypi.python.org/pypi/gridtk

