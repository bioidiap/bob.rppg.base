.. Guillaume HEUSCH <guillaume.heusch@idiap.ch>
.. Fri 15 Apr 15:09:35 CEST 2016

========================================
Remote Heart Rate Measurement Algorithms
========================================

This package implements several algorithms for remote photoplesthymography (rPPG). The following algorithms are available:

  - "Remote heart rate measurement from face videos under realistic situations", Li X, Chen J, Zhao G & Pietikäinen M, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014 (`pdf <http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_Remote_Heart_Rate_2014_CVPR_paper.pdf>`_) 
  - "Robust Pulse Rate From Chrominance-Based rPPG", de Haan & Jeanne, IEEE Transactions on Biomedical Engineering, 60, 10, 2013
  - "A Novel Algorithm for Remote Photoplesthymograpy: Spatial Subspace Rotation", IEEE Transactions on Biomedical Engineering, 2015.

Note that we are not providing the raw data files of the databases used by this package, but you can download them here:
  
  * Manhob HCI-Tagging (http://mahnob-db.eu/hci-tagging/) 
  * COHFACE (http://www.idiap.ch/dataset/cohface).


Installation
------------

This package heavily depends on bob_, so if you use this package and/or its results, please cite the following publication::

    @inproceedings{Anjos_ACMMM_2012,
        author = {A. Anjos AND L. El Shafey AND R. Wallace AND M. G\"unther AND C. McCool AND S. Marcel},
        title = {Bob: a free signal processing and machine learning toolbox for researchers},
        year = {2012},
        month = oct,
        booktitle = {20th ACM Conference on Multimedia Systems (ACMMM), Nara, Japan},
        publisher = {ACM Press},
    }

Note that currently, bob and this package are supported for Linux distributions only

    1. Install bob using conda as per instructions on the `bob website <https://www.idiap.ch/software/bob/install>`_
    2. Activate your conda environment containing bob
    3. Install the package by downloading the zip archive, opening a terminal and running::
       
       $ cd bob.rppg.base
       $ python bootstrap-buildout.py
       $ ./bin/buildout
    
    4. Download the metadata for the Manhob HCI Tagging database::
       
       $ ./bin/bob_dbmanage.py hci_tagging download --force

    5. Generate the documentation of the package, by running::

       $ ./bin/sphinx-build doc sphinx
       
       Point your browser to ``sphinx/index.html`` for further documentation and run instructions.


Reproducing article results
---------------------------

After having downloaded the database(s), you will be able to reproduce complete experiments
presented in the accompanying paper. To do so, you will first have to provide a database 
directory in the root folder of the package. 

We advise you to first use the COHFACE database since it is smaller and that the bounding boxes are provided.
Now create a symlink to the database directory containing the raw data::

    $ ln -s /path/to/the/cohface/database cohface

For instance, to reproduce results of the CHROM algorithm reported in Table 5::

    $ ./bin/python scripts-article/chrom-cohface-clean.py

Have a look at the scripts-article folder and the README within for more examples.

