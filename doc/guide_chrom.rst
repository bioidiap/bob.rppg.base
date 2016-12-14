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


Extract the pulse signal from video sequence 
--------------------------------------------

This scripts first load bounding boxes containing the faces in each frame
of each video sequence (if available) 
Otherwise, it will run a face detector (using :py:func:`bob.ip.facedetect.detect_single_face`).
Then, a skin color filter (:py:mod:`bob.ip.skincolorfilter`)
is applied to retrieve a mask containing skin pixels.

The mean skin color value is then computed, and projected onto the XY chrominance
colorspace. The signals in this colorspace are filtered using a bandpass filter
before the final pulse signal is built.


To extract the pulse signal from video sequences, do the following::

  $ ./bin/chrom_pulse.py cohface -vv

To see the full options, including parameters and protocols, type:: 

  $ ./bin/chrom_pulse.py --help 

The output of this step normally goes into a directory (you can override on
the options for this application) named ``pulse``.

.. note::

   The execution of this script is very slow - mainly due to the face detection. 
   You can speed it up using the gridtk_ (especially, if you're at Idiap). For example::

     $ ./bin/jman sub -t 3490 -- ./bin/chrom_pulse.py cohface

   The number of jobs (i.e. 3490) is given by typing::
     
     $ ./bin/chrom_pulse.py cohface --gridcount


.. _gridtk: https://pypi.python.org/pypi/gridtk
