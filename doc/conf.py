"""Sphinx configuration file for an LSST stack package.

This configuration only affects single-package Sphinx documentation builds.
"""

import lsst.ts.wep
from documenteer.conf.pipelinespkg import *  # type: ignore # noqa

project = "ts_wep"
html_theme_options["logotext"] = project  # type: ignore # noqa
html_title = project
html_short_title = project
doxylink = {}  # type: ignore


# Support the sphinx extension of mermaid
extensions = [
    "sphinxcontrib.mermaid",  # type: ignore # noqa
    "sphinx_automodapi.automodapi",  # type: ignore # noqa
]
