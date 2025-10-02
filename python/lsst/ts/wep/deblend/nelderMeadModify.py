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

"""
Pure Python/Numpy implementation of the Nelder-Mead algorithm.
Reference: https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method

Download from https://github.com/fchollet/nelder-mead.
Modify this script to fit the interger step for need.
"""

__all__ = ["feval", "nelderMeadModify"]

import copy
from typing import Callable

import numpy as np


def feval(func: Callable, vars: tuple = ()) -> float:
    """Evaluate the function.

    Parameters
    ----------
    func : callable
        Function to evaluate.
    vars : tuple, optional
        Inputs of function (the default is ().)

    Returns
    -------
    float
        Output of evaluated function.
    """

    return eval("func")(*vars)


def nelderMeadModify(
    func: Callable,
    x_start: np.ndarray,
    args: tuple = (),
    step: float = 0.1,
    no_improve_thr: float = 10e-6,
    no_improv_break: int = 10,
    max_iter: int = 0,
    alpha: float = 1.0,
    gamma: float = 2.0,
    rho: float = -0.5,
    sigma: float = 0.5,
) -> list:
    """Optimization of the Nelder-Mead algorithm.

    Parameters
    ----------
    func : callable
        Function to optimize, must return a scalar score and operate over a
        numpy array of the same dimensions as x_start.
    x_start : numpy.ndarray
        Initial position.
    args : tuple, optional
        Additional arguments required by func. (the default is ().)
    step : float, optional
        Look-around radius in initial step (the default is 0.1.)
    no_improve_thr : float, optional
        Break after no_improv_break iterations with an improvement lower than
        no_improv_thr (the default is 10e-6.)
    no_improv_break : int, optional
        Break after no_improv_break iterations with an improvement lower than
        no_improv_thr (the default is 10.)
    max_iter : int, optional
        Always break after this number of iterations. Set it to 0 to loop
        indefinitely (the default is 0.)
    alpha : float, optional
        Reflection parameter of the algorithm. (the default is 1.)
    gamma : float, optional
        Expansion parameter of the algorithm. (the default is 2.)
    rho : float, optional
        Contraction parameter of the algorithm. (the default is -0.5.)
    sigma : float, optional
        Reduction parameter of the algorithm (the default is 0.5.)

    Returns
    -------
    tuple
        Best parameter array and best score for the evaluated function.
    """

    # init
    dim = len(x_start)

    vars = (x_start,) + args
    prev_best = feval(func, vars)

    no_improv = 0
    res = [[x_start, prev_best]]

    for i in range(dim):
        x = copy.copy(x_start)
        x[i] = x[i] + step

        vars = (x,) + args
        score = feval(func, vars)

        res.append([x, score])

    # simplex iter
    iters = 0
    while 1:
        # order
        res.sort(key=lambda x: x[1])
        best = res[0][1]

        # break after max_iter
        if max_iter and iters >= max_iter:
            return res[0]
        iters += 1

        # break after no_improv_break iterations with no improvement
        # print '...best so far:', best

        if best < prev_best - no_improve_thr:
            no_improv = 0
            prev_best = best
        else:
            no_improv += 1

        if no_improv >= no_improv_break:
            return res[0]

        # centroid
        x0 = [0.0] * dim
        for tup in res[:-1]:
            for i, c in enumerate(tup[0]):
                x0[i] += c / (len(res) - 1)

        # reflection
        xr = x0 + alpha * (x0 - res[-1][0])

        vars = (xr,) + args
        rscore = feval(func, vars)

        if res[0][1] <= rscore < res[-2][1]:
            del res[-1]
            res.append([xr, rscore])
            continue

        # expansion
        if rscore < res[0][1]:
            xe = x0 + gamma * (x0 - res[-1][0])

            vars = (xe,) + args
            escore = feval(func, vars)

            if escore < rscore:
                del res[-1]
                res.append([xe, escore])
                continue
            else:
                del res[-1]
                res.append([xr, rscore])
                continue

        # contraction
        xc = x0 + rho * (x0 - res[-1][0])

        vars = (xc,) + args
        cscore = feval(func, vars)

        if cscore < res[-1][1]:
            del res[-1]
            res.append([xc, cscore])
            continue

        # reduction
        x1 = res[0][0]
        nres = []
        for tup in res:
            redx = x1 + sigma * (tup[0] - x1)

            vars = (redx,) + args
            score = feval(func, vars)

            nres.append([redx, score])
        res = nres
