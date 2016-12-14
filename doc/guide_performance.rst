.. py:currentmodule:: bob.rppg.base


Retrieve the heart-rate and compute performances
================================================


Frequency analysis and Heart-rate computation
---------------------------------------------

The heart-rate is computed by doing an analysis of the pulse 
signal. The Welch's algorithm is applied to find the power spectrum of the
signal, and the heart rate is found using peak detection in the frequency range
of interest.  To obtain the heart-rate, you should do the following::

  $ ./bin/rppg_frequency_analysis.py hci -vv

This script normally takes data from a directory called ``pulse``
and outputs data to a directory called ``heart-rate``. This output represents
the end of the processing chain and contains the estimated heart-rate for every
video sequence in the dataset.


Generating performance measures
---------------------------------------

In order to get some insights on how good the computed heart-rates match the
ground truth, you should execute the following script::

  $ ./bin/rppg_compute_performance.py hci --indir heart-rate -v -P 

This will output and save various statistics (Root Mean Square Error, 
Pearson correlation) as well as figures (error distribution, scatter plot)
