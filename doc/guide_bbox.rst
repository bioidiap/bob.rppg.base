.. py:currentmodule:: bob.rppg.base


Computing bounding boxes
=========================

If you would like to run several experiments on the same database, you may want
to first generate bounding boxes around faces for each frame of each sequence. 
Note that this step is not required, but saves time in the long run.

Note also that this process is computationnaly demanding, and as result
it could last for quite a long time.

To detect and save the bounding boxes for all the sequences of the 
COHFACE database, you should do the following::

  $ ./bin/rppg_bbox.py cohface --dbdir path/to/the/data --outdir bounding-boxes-cohface

To detect and save the bounding boxes for all the sequences of the 
Manhob HCI Tagging database, you should do the following::

  $ ./bin/rppg_bbox.py hci --dbdir path/to/the/data --outdir bounding-boxes-hci
