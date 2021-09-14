.. py:currentmodule:: lsst.ts.wep

.. _lsst.ts.wep-phosimDonutTemplates:

#########################################
Using Phosim Donut Templates
#########################################

This document describes how to create and use the phosim donut templates
required by :py:class:`DonutTemplatePhosim`.

Creating the templates
======================

The code to generate phosim donut templates is located in the `ts_phosim` package.
`ts_phosim` has documentation on how to create the phosim donut templates using `runCreatePhosimDonutTemplates.py` in the bin.src directory of `ts_phosim`.

Location of files
=================

After creating the phosim donut templates the code in :py:class:`DonutTemplatePhosim`
will look for the templates in the folder `policy/cwfs/donutTemplateData/phosimTemplates`.

