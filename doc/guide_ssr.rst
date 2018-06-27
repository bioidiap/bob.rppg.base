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
as described in Algorithm 1 in the paper. To get the pulse signals for
all video in a database, do the following::

  $ ./bin/bob_rppg_ssr_pulse.py config.py -v

To see the full options, including parameters and protocols, type:: 

  $ ./bin/bob_rppg_ssr_pulse.py --help 

As you can see, the script takes a configuration file as argument. This
configuration file is required to at least specify the database, but can also
be used to provide various parameters. A full example of configuration is
given below.

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

  basedir = 'ssr-hci-cvpr14/'

  # EXTRACT PULSE 
  pulsedir = basedir + 'pulse'
  start = 306
  end = 2136
  threshold = 0.1
  skininit = True
  stride = 30

  # FREQUENCY ANALYSIS
  hrdir = basedir + 'hr'
  nsegments = 8
  nfft = 4096

  # RESULTS
  resultdir = basedir + 'results'

.. note::

   The execution of this script is very slow - mainly due to the face detection. 
   You can speed it up using the gridtk_ (especially, if you're at Idiap). For example::

     $ ./bin/jman sub -t 3490 -- ./bin/bob_rppg_ssr_pulse.py config.py

   The number of jobs (i.e. 3490) is given by typing::
     
     $ ./bin/bob_rppg_ssr_pulse.py config.py --gridcount


.. _gridtk: https://pypi.python.org/pypi/gridtk
