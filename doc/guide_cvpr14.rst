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


Step 1: Extract signals from video sequences
--------------------------------------------

This scripts first load previously detected facial keypoints to build the mask 
covering the bottom region of the face (note that the keypoints are not
provided, but could be obtained using bob.ip.flandmark for instance). Once the
mask has been built, it is tracked across the whole sequence using the
methodology described in [li-cvpr-2014]_. The face is 
detected using :py:func:`bob.ip.facedetect.detect_single_face`, and the
tracking makes usage of openCV (version 2). 

To extract the mean green colors the face region and of
the background across the video sequences of the COHFACE 
database, do the following::

  $ ./bin/cvpr14_extract_signals.py cohface -vv

To see the full options, including parameters and protocols, type:: 

  $ ./bin/cvpr14_extract_signals.py --help 

The output of this step normally goes into directories (you can override on
the options for this application) named ``face`` and ``background``.

.. note::

   The execution of this script is very slow - mainly due to the face detection. 
   You can speed it up using the gridtk_ (especially, if you're at Idiap). For example::

     $ ./bin/jman sub -t 3490 -- ./bin/cvpr14_extract_signals.py cohface

   The number of jobs (i.e. 3490) is given by typing::
     
     $ ./bin/cvpr14_extract_signals.py cohface --gridcount


Step 2: Illumination Rectification
----------------------------------

The second step is to remove global illumination from the color signal
extracted from the face area. The background signal is filtered using
Normalized Linear Mean Square and is then removed from the face signal. To get
the rectified green signal of the face area, you should execute the following
script::

  $ ./bin/cvpr14_illumination.py cohface -v

This script takes as input the result directories (normally named) ``face`` and
``background`` and outputs data to a directory named ``illumination``.


Step 3: Non rigid Motion Elimination
------------------------------------

Since motion can contaminate the pulse signal, a heuristic to supress large
variation in the signal has been implemented. It simply divides the signal
into segments, and then removes the segments for which the standard deviation is
too high. This script first computes the standard deviation in the green
channel on all the segment of all sequences. By default, the threshold is set such that 95%
of all the segments will be retained. To get the signals where large motion has
been eliminated, execute the following commands::

  $ ./bin/cvpr14_motion.py cohface --save-threshold threshold.txt -vv
  $ ./bin/cvpr14_motion.py cohface --load-threshold threshold.txt -vv

This script takes as input the result directory (normally named)
``illumination`` and outputs data to a directory called
``motion``. The first call computes the threshold while the second
one actually discards segments and builds the corrected signals.


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

  $ ./bin/cvpr14_filter.py cohface -vv

This script normally takes data from a directory called ``motion-eliminated``
and outputs data to a directory called ``filtered``.

.. _gridtk: https://pypi.python.org/pypi/gridtk

