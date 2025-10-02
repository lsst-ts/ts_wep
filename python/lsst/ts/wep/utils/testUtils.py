# This file is part of ts_wep.
#
# Developed for the LSST Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["enforce_single_threading"]

import os

import threadpoolctl


def enforce_single_threading() -> None:
    """
    Set OpenBLAS, MKL, and OMP to use only one thread.
    We implemented this to avoid issues with test hanging
    on Jenkins. Primarily used in tests that use
    danish which is where we saw the most issues.
    """
    # Set environment variables for single-threading
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # Additional environment variables
    os.environ["NUMEXPR_NUM_THREADS"] = "1"  # For NumExpr
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # For Apple Accelerate
    os.environ["BLIS_NUM_THREADS"] = "1"  # For BLIS

    # Limit threadpool to 1 thread for BLAS and OpenMP
    threadpoolctl.threadpool_limits(limits=1, user_api="blas")
    threadpoolctl.threadpool_limits(limits=1, user_api="openmp")
