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

Also, we provide scripts to infer the heart-rate from the pulse signal, and also
to evaluate global performance on two different databases: Manhob HCI-Tagging 
(http://mahnob-db.eu/hci-tagging/) and COHFACE (www)


Step 1: Extract signals from video sequences
--------------------------------------------

This scripts first load previously detected facial keypoints to build the mask 
covering the bottom region of the face (note that the keypoints are not
provided, but could be obtained using bob.ip.flandmark for instance). Once the
mask has been built, it is tracked across the whole sequence using the
methodology described in [li-cvpr-2014]_. As a consequence, the face is 
detected using :py:func:`bob.ip.facedetect.detect_single_face`, and the
tracking also makes usage of openCV (version 2). 

To extract the signals of the mean color the face region and of
the background across the video sequences of the Manhob HCI-Tagging 
database, do the following::

  $ ./bin/cvpr14_extract_signals.py hci -vv

To see the full options, including parameters and protocols, type:: 

  $ ./bin/cvpr14_extract_signals.py --help 

The output of this step normally goes into directories (you can override on
the options for this application) named ``face`` and ``background``.

.. note::

   The execution of this script is very slow - mainly due to the face detection. 
   You can speed it up using the gridtk_ (especially, if you're at Idiap). For example::

     $ ./bin/jman sub -t 3490 -- ./bin/cvpr14_extract_signals.py hci

   The number of jobs (i.e. 3490) is given by typing::
     
     $ ./bin/cvpr14_extract_signals.py hci --gridcount


Step 2: Illumination Rectification
----------------------------------

The second step is to remove global illumination from the color signals
extracted from the face area. The background signal is filtered using
Normalized Linear Mean Square and is then removed from the face signal. To get
the rectified color signals of the face area, you should execute the following
script::

  $ ./bin/cvpr14_illumination_rectification.py hci -v

This script takes as input the result directories (normally named) ``face`` and
``background`` and outputs data to a directory named ``illumination-corrected``.


Step 3: Non rigid Motion Elimination
------------------------------------

Since motion can contaminate the pulse signal, a heuristic to supress large
variation in the signals has been implemented. It simply divides the signal
into segments, and then removes the signal for which the standard deviation is
too high. This script first computes the standard deviation in the green
channel on all the segment of all sequences. The threshold is set such that 95%
of all the segments will be retained. To get the signals where large motion has
been eliminated, execute the following command::

  $ ./bin/cvpr14_motion_elimination.py hci -vv

This script takes as input the result directory (normally named)
``illumination-corrected`` and outputs data to a directory called
``motion-eliminated``.


Step 4: Filtering
-----------------

In this step, a detrend filter are applied to the color signals. It
will remove global trends in the signal (i.e. the signal will be more
or less flat after this procedure). 
The next step is to remove (high frequency) noise to the detrended signal.
This is done using a moving-average filter, with a relatively small
window. Finally, a bandpass filter is applied to restrict the
frequencies to the range corresponding to a plausible heart-rate. To filter the
corrected signals, you should execute the following command::

  $ ./bin/cvpr14_filter.py hci -vv

This script normally takes data from a directory called ``motion-eliminated``
and outputs data to a directory called ``filtered``.


Step 5: Frequency analysis and Heart-rate computation
-----------------------------------------------------

Finally, the heart-rate is computed by doing an analysis of the filtered green
signal. The Welch's algorithm is applied to find the power spectrum of the
signal, and the heart rate is found using peak detection in the frequency range
of interest.  To obtain the heart-rate, you should do the following::

  $ ./bin/cvpr14_frequency_analysis.py hci -vv

This script normally takes data from a directory called ``filtered``
and outputs data to a directory called ``heart-rate``. This output represents
the end of the processing chain and contains the estimated heart-rate for every
video sequence in the dataset.


Step 6: Generating performance measures
---------------------------------------

In order to get some insights on how good the computed heart-rate match the
ground truth, you should execute the following script::

  $ ./bin/cvpr14_generate_results.py hci -v -P

This will output and save various statistics (Root Mean Square Error, 
Pearson correlation) as well as figures (error distribution, scatter plot)


Licensing
=========

This work is licensed under the GPLv3_.

.. _GPLv3: http://www.gnu.org/licenses/gpl-3.0.en.html
.. _gridtk: https://pypi.python.org/pypi/gridtk
.. _bob: http://idiap.github.io/bob/
