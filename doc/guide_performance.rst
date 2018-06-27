.. py:currentmodule:: bob.rppg.base


Retrieve the heart-rate and compute performances
================================================


Frequency analysis and Heart-rate computation
---------------------------------------------

The heart-rate is computed by doing an analysis of the pulse 
signal. The Welch's algorithm is applied to find the power spectrum of the
signal, and the heart rate is found using peak detection in the frequency range
of interest.  To obtain the heart-rate, you should do the following::

  $ ./bin/bob_rppg_base_get_heart_rate.py config.py -v


Generating performance measures
---------------------------------------

In order to get some insights on how good the computed heart-rates match the
ground truth, you should execute the following script::

  $ ./bin/bob_rppg_base_compute_performance.py config.py -v 

This will output and save various statistics (Root Mean Square Error, 
Pearson correlation) as well as figures (error distribution, scatter plot).


Again, these scripts rely on the use of configuration 
files. An minimal example is given below:

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

  # FREQUENCY ANALYSIS
  hrdir = basedir + 'hr'
  nsegments = 16
  nfft = 8192

