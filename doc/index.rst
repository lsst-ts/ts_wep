.. |CSC_developer| replace:: *Bryce Kalmbach <brycek@uw.edu>* and *Chris Suberlak <suberlak@uw.edu>*
.. |CSC_product_owner| replace:: *Sandrine Thomas <sthomas@lsst.org>*

.. Note that the ts_ prefix is omitted from the title

###
WEP
###

.. image:: https://img.shields.io/badge/GitHub-ts__wep-green.svg
    :target: https://github.com/lsst-ts/ts_wep
.. image:: https://img.shields.io/badge/Jenkins-ts__wep-green.svg
    :target: https://tssw-ci.lsst.org/job/LSST_Telescope-and-Site/job/ts_wep

.. _Overview:

Overview
========

The wavefront estimation pipeline (WEP) calculates the wavefront error in annular Zernike polynomials up to 22 terms based
on the intra- and extra-focal donut images. The wavefront error is determined by solving the transport of intensity equation
(TIE) that approximates the change of intensity mainly comes from the wavefront error.

In automatic operation, WEP will be run as part of the Main Telescope Active Optics System (`MTAOS <https://ts-mtaos.lsst.io/index.html>`_).

The badges above navigate to the GitHub repository for the WEP code and Jenkins CI jobs.

.. _User_Documentation:

User Documentation
==================

Observatory operators and other interested parties should consult the user guide for insights into WEP operations.

.. toctree::
    user-guide/user-guide
    :maxdepth: 1

.. _Configuration:

Configuring the WEP
=====================

WEP's configuration is described at the following link.

.. toctree::
    configuration/configuration
    :maxdepth: 1

.. _Development_Documentation:

Development Documentation
=========================

Classes and their methods, and how to get involved in the WEP development is described in this section.

.. toctree::
    developer-guide/developer-guide
    :maxdepth: 1

.. _Version_History:

Version History
===============

The version history is at the following link.

.. toctree::
    versionHistory
    :maxdepth: 1

The released version is `here <https://github.com/lsst-ts/ts_wep/releases>`_.

.. _Contact_Personnel:

Contact Personnel
=================

For questions not covered in the documentation, emails should be addressed to the developers: |CSC_developer|.
The product owner is |CSC_product_owner|.

This page was last modified |today|.