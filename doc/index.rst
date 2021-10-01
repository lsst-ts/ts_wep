.. py:currentmodule:: lsst.ts.wep

.. _lsst.ts.wep:

##############
lsst.ts.wep
##############

Wavefront estimation pipeline (WEP) calculates the wavefront error in annular Zernike polynomials up to 22 terms based on the intra- and extra-focal donut images. The wavefront error is determined by solving the transport of intensity equation (TIE) that approximates the change of intensity mainly comes from the wavefront error.

.. _lsst.ts.wep-using:

Using lsst.ts.wep
====================

.. toctree::
   :maxdepth: 1

Important classes:

* `WfEstimator` calculates the wavefront error in annular Zernike polynomials up to 22 terms based on the defocal donut images.

Important Pipeline Tasks:

* `EstimateZernikesFamTask` calculates the Zernike polynomials for LSSTFam or ComCam sensors when provided input exposures and source catalog.
* `EstimateZernikesCwfsTask` calculates the Zernike polynomials on corner wavefront sensors when provided input exposures and source catalog.

Important enums:

* `FilterType` defines the type of active filter.
* `CamType` defines the type of camera.
* `BscDbType` defines the type of bright star catalog database.

.. _lsst.ts.wep-pyapi:

Python API reference
====================

.. automodapi:: lsst.ts.wep
    :no-inheritance-diagram:

.. _lsst.ts.wep-content:

Content
====================

.. toctree::

   content

.. _lsst.ts.wep-contributing:

Contributing
============

``lsst.ts.wep`` is developed at https://github.com/lsst-ts/ts_wep.


.. _lsst.ts.wep-see-also:

See Also
========

.. toctree::
    :maxdepth: 2
    :titlesonly:
    :glob:

    phosimDonutTemplates.rst

.. _lsst.ts.wep-version:

Version
====================

.. toctree::

   versionHistory
