# -*- coding: utf-8 -*-
from .Utility import FilterType, CamType, BscDbType

# This class needs the scons to build the cython code. In the Jenkins test,
# this will be a problem to import.
try:
    from .WfEstimator import WfEstimator
except ImportError:
    pass

# The version file is gotten by the scons. However, the scons does not support
# the build without unit tests. This is a needed function for the Jenkins to
# use.
try:
    from .version import *
except ImportError:
    __version__ = "?"
