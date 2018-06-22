.. Bob rppg base documentation master file, created by
   sphinx-quickstart on Mon Apr  4 14:15:40 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. _bob.ip.facedetect:
.. _bob.ip.skincolorfilter:


==================================================
 Bob's implementation of different rPPG algorithms
==================================================

This module contains the implementation of different remote photoplesthymography (rPPG) algorithms:
  
  * Li's CVPR14 [li-cvpr-2014]_.
  * CHROM [dehaan-tbe-2013]_.
  * 2SR [wang-tbe-2015]_. 

Also, we provide scripts to infer the heart-rate from the pulse signals, and
to evaluate global performance on two different databases: Manhob HCI-Tagging 
(http://mahnob-db.eu/hci-tagging/) and COHFACE (http://www.idiap.ch/dataset/cohface).

.. warning:: 
  
   You should download the databases before trying to run anything below!

Documentation
-------------

.. toctree::
   :maxdepth: 3
   
   guide_cvpr14
   py_api_cvpr14

   guide_chrom
   py_api_chrom

   guide_ssr
   py_api_ssr

   guide_performance


References
----------

.. [li-cvpr-2014]  *Li X, Chen J, Zhao G & Pietikäinen M*. **Remote heart rate measurement from face videos under realistic situations** IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2014. `pdf <http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Li_Remote_Heart_Rate_2014_CVPR_paper.pdf>`__

.. [dehaan-tbe-2013]  *de Haan, G. & Jeanne, V*. **Robust Pulse Rate from Chrominance based rPPG** IEEE Transactions on Biomedical Engineering, 2013. `pdf <http://www.es.ele.tue.nl/~dehaan/pdf/169_ChrominanceBasedPPG.pdf>`__

.. [wang-tbe-2015] *Wang, W., Stuijk, S. and de Haan, G*. **A Novel Algorithm for Remote Photoplesthymograpy: Spatial Subspace Rotation** IEEE Trans. On Biomedical Engineering, 2015

Licensing
---------

This work is licensed under the GPLv3_.

.. _GPLv3: http://www.gnu.org/licenses/gpl-3.0.en.html

.. _gridtk: https://pypi.python.org/pypi/gridtk
.. _bob: http://idiap.github.io/bob/

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
