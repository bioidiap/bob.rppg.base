.. Guillaume HEUSCH <guillaume.heusch@idiap.ch>
.. Fri 15 Apr 15:09:35 CEST 2016

.. image:: http://img.shields.io/badge/docs-stable-yellow.png
   :target: http://pythonhosted.org/bob.rppg.cvpr14/index.html
.. image:: http://img.shields.io/badge/docs-latest-orange.png
   :target: https://www.idiap.ch/software/bob/docs/latest/bioidiap/bob.rppg.cvpr14/master/index.html
.. image:: https://travis-ci.org/bioidiap/bob.rppg.cvpr14.svg?branch=master
   :target: https://travis-ci.org/bioidiap/bob.rppg.cvpr14?branch=master
.. image:: https://coveralls.io/repos/bioidiap/bob.rppg.cvpr14/badge.svg?branch=master
   :target: https://coveralls.io/r/bioidiap/bob.rppg.cvpr14?branch=master
.. image:: https://img.shields.io/badge/github-master-0000c0.png
   :target: https://github.com/bioidiap/bob.rppg.cvpr14/tree/master
.. image:: http://img.shields.io/pypi/v/bob.rppg.cvpr14.png
   :target: https://pypi.python.org/pypi/bob.rppg.cvpr14
.. image:: http://img.shields.io/pypi/dm/bob.rppg.cvpr14.png
   :target: https://pypi.python.org/pypi/bob.rppg.cvpr14


============================================
Li's Remote Heart Rate Measurement Algorithm
============================================

This package implements the remote heart rate measurement algorithm described in "Remote heart rate measurement from face videos under realistic situations", Li X, Chen J, Zhao G & Pietikäinen M, IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014 (`pdf <http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_Remote_Heart_Rate_2014_CVPR_paper.pdf>`_) 

Installation
------------

This package heavily depends on `bob <http://idiap.github.io/bob/>`_, so if you use this package and/or its results, please cite the following publication::

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
     
     $ cd bob.rppg.cvpr14
     $ python bootstrap-buildout.py
     $ ./bin/buildout


For Bob_ (and hence this package) to be able to work properly, some dependent packages are required to be installed.
Please make sure that you have read the `Dependencies <https://github.com/idiap/bob/wiki/Dependencies>`_ for your operating system.
In particular, this package requires OpenCV (version 2.4.10) to be installed. If you have a different version (i.e 
version 3), the code might need some editing, see full documentation for details

Documentation and Further Information
-------------------------------------

You can generate the documentation locally:

  .. code:: bash 
     
     $ ./bin/sphinx-build doc sphinx

The documentation, including a user's guide is available at ./sphinx/index.html.
For information on other packages of Bob_, on tutorials, asking questions, submitting issues and starting discussions, please visit its website.

.. _bob: https://www.idiap.ch/software/bob

