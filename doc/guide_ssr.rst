.. py:currentmodule:: bob.rppg.base


2SR user's guide
==================

This package contains a free-software implementation of IEEE Trans. on Biomedical 
engineering paper "A Novel Algorithm for Remote Photoplesthymograpy: 
Spatial Subspace Rotation", Wang et al [wang-tbe-2015]_::

  @Article{wang-tbe-2015,
    Author         = {Wang, W., Stuijk, S. and de Haan, G},
    Title          = {A Novel Algorithm for Remote Photoplesthymograpy: 
                     Spatial Subspace Rotation}
    Journal        = {IEEE Trans. On Biomedical Engineering},
    Volume         = {PP},
    Number         = {99},
    Pages          = {},
    year           = 2015
  }

The algorithm to retrieve the pulse signal can be divided into several steps:

  1. Extracting skin pixels from each frame 
  2. Compute eigenvalues and eigenvectors of the skin color correlation matrix
  3. Update the pulse signal at each frame using eigenvalues and eigenvectors

Extract the pulse signal from video sequence 
--------------------------------------------

To extract the pulse signal, we first need the set of skin-colored pixels 
from each frame image. Hence, a skin color filter (:py:mod:`bob.ip.skincolorfilter`)
is applied to retrieve a mask containing skin pixels.

After having applied the skin color filter, the full algorithm is applied,
as described in Algorithm 1 in the paper. To get the pulse signal, do
the following::

  $ ./bin/ssr_pulse.py cohface

The result of this script will be the pulse signal. 
The output of this step normally goes into a directory named ``pulse``.

.. note::

   The execution of this script is very slow - mainly due to the face detection. 
   You can speed it up using the gridtk_ (especially, if you're at Idiap). For example::

     $ ./bin/jman sub -t 3490 -- ./bin/ssr_pulse.py cohface

   The number of jobs (i.e. 3490) is given by typing::
     
     $ ./bin/ssr_pulse.py cohface --gridcount


.. _gridtk: https://pypi.python.org/pypi/gridtk
