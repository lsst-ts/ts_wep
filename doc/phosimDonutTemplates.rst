.. py:currentmodule:: lsst.ts.wep

.. _lsst.ts.wep-phosimDonutTemplates:

########################################
Creating Phosim Donut Templates
########################################

This document describes the code to generate the phosim donut templates
that are used by :py:class:`DonutTemplatePhosim`.

Running the code
================

1) Make sure your environment is set up to use ts_wep and that you have phosim
   installed with the environment variable $PHOSIMPATH set to the phosim directory.
2) Use python to run `runCreatePhosimDonutTemplates.py` in the bin.src directory.
   Use `--help` to see all options.
3) Templates by default will appear in
   `ts_wep/policy/cwfs/templateDonutData/phosimTemplates`.

Input files
===========

The following input files found in `policy/cwfs/donutTemplateData` are used
to create the donut templates.

* **starExtra.inst**: Phosim instance catalog to create one extra-focal donut per CCD
                      of LSST camera using PhosimMapper.
* **starIntra.inst**: Phosim instance catalog to create one intra-focal donut per CCD
                      of LSST camera using PhosimMapper.
* **star.cmd**: Phosim command file.
