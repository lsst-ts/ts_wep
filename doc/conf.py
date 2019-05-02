"""Sphinx configuration file for an LSST stack package.

This configuration only affects single-package Sphinx documentation builds.
"""

from documenteer.sphinxconfig.stackconf import build_package_configs
import lsst.ts.wep


_g = globals()
_g.update(build_package_configs(
    project_name='ts_wep',
    version=lsst.ts.wep.version.__version__))

# Support the sphinx extension of plantuml
extensions.append('sphinxcontrib.plantuml')

# Put the path to plantuml.jar
plantuml = 'java -jar /home/lsst/plantuml.jar'
