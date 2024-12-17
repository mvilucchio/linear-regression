import numpy as np
from numba import njit


@njit(error_model="numpy", fastmath=True)
def find_first_greather_than_zero(vec, reversed):
    if reversed:
        vec = np.flip(vec)
    for i, elem in enumerate(vec):
        if elem > 0:
            if reversed:
                return len(vec) - i - 1
            return i
    return -1


@njit(error_model="numpy", fastmath=True)
def brent_root_finder(
    fun: callable, xa: float, xb: float, xtol: float, rtol: float, max_iter: int, args: tuple
):
    xpre, xcur = xa, xb
    xblk = 0.0
    fblk = 0.0
    spre = 0.0
    scur = 0.0

    # /* the tolerance is 2*delta */

    fpre = fun(xpre, *args)
    fcur = fun(xcur, *args)

    if fpre * fcur > 0:
        raise ValueError("BRENT The endpoints should have different signs.")

    if fpre == 0:
        return xpre
    if fcur == 0:
        return xcur

    for i in range(max_iter):
        if fpre != 0 and fcur != 0 and (np.sign(fpre) != np.sign(fcur)):
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre

        if np.abs(fblk) < np.abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre

        delta = (xtol + rtol * np.abs(xcur)) / 2
        sbis = (xblk - xcur) / 2

        if fcur == 0 or np.abs(sbis) < delta:
            return xcur

        if np.abs(spre) > delta and np.abs(fcur) < np.abs(fpre):
            if xpre == xblk:
                # /* interpolate */
                stry = -fcur * (xcur - xpre) / (fcur - fpre)
            else:
                # /* extrapolate */
                dpre = (fpre - fcur) / (xpre - xcur)
                dblk = (fblk - fcur) / (xblk - xcur)
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))

            if 2 * np.abs(stry) < np.minimum(np.abs(spre), 3 * np.abs(sbis) - delta):
                # /* good short step */
                spre = scur
                scur = stry
            else:
                # /* bisect */
                spre = sbis
                scur = sbis
        else:
            # /* bisect */
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur
        if np.abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0 else -delta

        fcur = fun(xcur, *args)

    if i == max_iter - 1:
        raise RuntimeError("BRENT Maximum number of iterations reached.")

    return xcur


@njit(error_model="numpy", fastmath=True)
def all_brents(
    fun: callable,
    D_fun: callable,
    fun_args: tuple,
    left_bnd: float,
    right_bnd: float,
    min_x: float,
    max_x: float,
    n_points: int,
    xtol: float,
    rtol: float,
    max_iter: int,
) -> float:
    """
    Find all zeros of D_fun in [min_x, max_x] and return the one that minimizes fun.
    """
    if left_bnd >= right_bnd:
        raise ValueError("Left bound must be smaller than right bound.")
    if min_x >= max_x:
        raise ValueError("Min x must be smaller than max x.")
    if left_bnd > min_x or right_bnd < max_x:
        raise ValueError("Bounds must contain the interval [min_x, max_x].")

    # Create extended grid
    x = np.empty(n_points + 2)
    x[0] = left_bnd
    x[1 : n_points + 1] = np.linspace(min_x, max_x, n_points)
    x[n_points + 1] = right_bnd

    y = np.array([D_fun(xi, *fun_args) for xi in x])

    # Count exact zeros and sign changes
    n_changes = 0
    for i in range(1, len(y) - 1):  # Skip extreme points
        if y[i] == 0.0:  # Exact zero
            n_changes += 1
        elif y[i] * y[i + 1] < 0:  # Sign change between non-zero values
            n_changes += 1

    # Pre-allocate arrays
    zeros = np.zeros(n_changes)
    idx = 0

    # Process exact zeros and sign changes
    for i in range(1, len(y) - 1):  # Skip extreme points
        if y[i] == 0.0:  # Exact zero found
            zeros[idx] = x[i]
            idx += 1
        elif y[i] * y[i + 1] < 0:  # Sign change between non-zero values
            # Use Brent's method to find the zero
            xa, xb = x[i], x[i + 1]
            zeros[idx] = brent_root_finder(D_fun, xa, xb, xtol, rtol, max_iter, fun_args)
            idx += 1

    # If no zeros found, check endpoints
    if n_changes == 0:
        f_min = fun(min_x, *fun_args)
        f_max = fun(max_x, *fun_args)
        return min_x if f_min < f_max else max_x

    # Evaluate function at all zeros and endpoints
    candidates = np.zeros(n_changes + 2)
    candidates[0:n_changes] = zeros
    candidates[n_changes] = min_x
    candidates[n_changes + 1] = max_x

    f_values = np.zeros(len(candidates))
    for i in range(len(candidates)):
        f_values[i] = fun(candidates[i], *fun_args)

    # Return the point with lowest function value
    min_idx = 0
    min_value = f_values[0]
    for i in range(1, len(f_values)):
        if f_values[i] < min_value:
            min_value = f_values[i]
            min_idx = i

    return candidates[min_idx]
