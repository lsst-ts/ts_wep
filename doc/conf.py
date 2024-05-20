"""Sphinx configuration file for an LSST stack package.

This configuration only affects single-package Sphinx documentation builds.
"""

import os.path

import lsst.ts.wep
from documenteer.conf.pipelinespkg import *

project = "ts_wep"
html_theme_options["logotext"] = project
html_title = project
html_short_title = project
doxylink = {}


# Support the sphinx extension of mermaid
extensions = [
    "sphinxcontrib.mermaid",
    "sphinx_automodapi.automodapi",
]
