.. py:currentmodule:: bob.rppg.base


CHROM user's guide
==================

This package contains a free-software implementation of the Technical University of Eindhoven's
IEEE Transactions on Biomedical Engineering article "Robust Pulse Rate from Chrominance based rPPG", 
[dehaan-tbe-2013]_::

  @article{deHaan:2013,
    author={G. de Haan and V. Jeanne},
    journal={IEEE Transactions on Biomedical Engineering},
    title={Robust Pulse Rate From Chrominance-Based rPPG},
    year={2013},
    volume={60},
    number={10},
    pages={2878-2886}, 
  }

The algorithm to retrieve the pulse signal can be divided into several steps:
  1. Extracting skin color signals from the video sequence.
  2. Projecting the mean skin color in the defined chrominance space.
  3. Bandpass filtering in the chrominance space.
  4. Building the pulse signal.

Also, we provide scripts to infer the heart-rate from the pulse signal, and also
to evaluate global performance on two different databases: Manhob HCI-Tagging 
(http://mahnob-db.eu/hci-tagging/) and COHFACE (www)


Step 1: Extract the pulse signal from video sequence 
----------------------------------------------------

This scripts first load bounding boxes containing the faces in each frame
of each video sequence (if available) 
Otherwise, it will run a face detector (using :py:func:`bob.ip.facedetect.detect_single_face`).
Then, a skin color filter (:py:mod:`bob.ip.skincolorfilter`)
is applied to retrieve a mask containing skin pixels.

The mean skin color value is then computed, and projected onto the XY chrominance
colorspace. The signals in this colorspace are filtered using a bandpass filter
before the final pulse signal is built.


To extract the pulse signal from video sequences, do the following::

  $ ./bin/chrom_pulse.py hci -vv

To see the full options, including parameters and protocols, type:: 

  $ ./bin/chrom_pulse.py --help 

The output of this step normally goes into a directory (you can override on
the options for this application) named ``pulse``.

.. note::

   The execution of this script is very slow - mainly due to the face detection. 
   You can speed it up using the gridtk_ (especially, if you're at Idiap). For example::

     $ ./bin/jman sub -t 3490 -- ./bin/chrom_pulse.py hci

   The number of jobs (i.e. 3490) is given by typing::
     
     $ ./bin/chrom_pulse.py hci --gridcount


Step 2: Frequency analysis and Heart-rate computation
-----------------------------------------------------

Finally, the heart-rate is computed by doing an analysis of the pulse 
signal. The Welch's algorithm is applied to find the power spectrum of the
signal, and the heart rate is found using peak detection in the frequency range
of interest.  To obtain the heart-rate, you should do the following::

  $ ./bin/chrom_hr.py hci -vv

This script normally takes data from a directory called ``pulse``
and outputs data to a directory called ``heart-rate``. This output represents
the end of the processing chain and contains the estimated heart-rate for every
video sequence in the dataset.


Step 3: Generating performance measures
---------------------------------------

In order to get some insights on how good the computed heart-rate match the
ground truth, you should execute the following script::

  $ ./bin/chrom_perf.py hci -v -P

This will output and save various statistics (Root Mean Square Error, 
Pearson correlation) as well as figures (error distribution, scatter plot)


Licensing
=========

This work is licensed under the GPLv3_.

.. _GPLv3: http://www.gnu.org/licenses/gpl-3.0.en.html
.. _gridtk: https://pypi.python.org/pypi/gridtk
.. _bob: http://idiap.github.io/bob/
