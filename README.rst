.. Guillaume HEUSCH <guillaume.heusch@idiap.ch>
.. Fri 15 Apr 15:09:35 CEST 2016

.. image:: https://gitlab.idiap.ch/biometric/bob.rppg.base/badges/master/build.svg?

========================================
Remote Heart Rate Measurement Algorithms
========================================

This package implements several algorithms for remote photoplesthymography (rPPG). The following algorithms are available:

  - "Remote heart rate measurement from face videos under realistic situations", Li X, Chen J, Zhao G & Pietikäinen M, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014 (`pdf <http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_Remote_Heart_Rate_2014_CVPR_paper.pdf>`_) 
  - "Robust Pulse Rate From Chrominance-Based rPPG", de Haan & Jeanne, IEEE Transactions on Biomedical Engineering, 60, 10, 2013
  - "A Novel Algorithm for Remote Photoplesthymograpy: Spatial Subspace Rotation", IEEE Transactions on Biomedical Engineering, 2015 

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

To install the package, git-clone it or download the zip archive, open a terminal and run:

  .. code:: bash 
     
     $ cd bob.rppg.base
     $ python bootstrap-buildout.py
     $ ./bin/buildout


For bob_ (and hence this package) to be able to work properly, some dependent packages are required to be installed.
Please make sure that you have read the `Dependencies <https://github.com/idiap/bob/wiki/Dependencies>`_ for your operating system.
In particular, this package requires OpenCV (version 2.4.10) to be installed. If you have a different version (i.e 
version 3), the code might need some editing, see full documentation for details.

Documentation and Further Information
-------------------------------------

You can generate the documentation locally:

  .. code:: bash 
     
     $ ./bin/sphinx-build doc sphinx

The documentation, including a user's guide is available at ./sphinx/index.html.
For information on other packages of bob_, on tutorials, asking questions, submitting issues and starting discussions, please visit its website.

.. _bob: https://www.idiap.ch/software/bob

