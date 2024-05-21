import numpy as np
from numba import njit


@njit(error_model="numpy", fastmath=False)
def bracket(
    func: callable,
    xa: float = 0.0,
    xb: float = 1.0,
    args=(),
    grow_limit: float = 110.0,
    maxiter: float = 1000,
) -> tuple:

    _gold = 1.618034  # golden ratio: (1.0+sqrt(5.0))/2.0
    _verysmall_num = 1e-21

    fa = func(xa, *args)
    fb = func(xb, *args)

    if fa < fb:  # Switch so fa > fb
        xa, xb = xb, xa
        fa, fb = fb, fa

    xc = xb + _gold * (xb - xa)
    fc = func(xc, *args)

    funcalls = 3
    iter = 0

    while fc < fb:
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        val = tmp2 - tmp1
        if abs(val) < _verysmall_num:
            denom = 2.0 * _verysmall_num
        else:
            denom = 2.0 * val
        w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        wlim = xb + grow_limit * (xc - xb)
        if iter > maxiter:
            raise RuntimeError(
                "No valid bracket was found before the iteration limit was reached. Consider trying different initial points or increasing `maxiter`."
            )
        iter += 1
        if (w - xc) * (xb - w) > 0.0:
            fw = func(w, *args)
            funcalls += 1
            if fw < fc:
                xa = xb
                xb = w
                fa = fb
                fb = fw
                break
            elif fw > fb:
                xc = w
                fc = fw
                break
            w = xc + _gold * (xc - xb)
            fw = func(w, *args)
            funcalls += 1
        elif (w - wlim) * (wlim - xc) >= 0.0:
            w = wlim
            fw = func(w, *args)
            funcalls += 1
        elif (w - wlim) * (xc - w) > 0.0:
            fw = func(w, *args)
            funcalls += 1
            if fw < fc:
                xb = xc
                xc = w
                w = xc + _gold * (xc - xb)
                fb = fc
                fc = fw
                fw = func(w, *args)
                funcalls += 1
        else:
            w = xc + _gold * (xc - xb)
            fw = func(w, *args)
            funcalls += 1
        xa = xb
        xb = xc
        xc = w
        fa = fb
        fb = fc
        fc = fw

    # three conditions for a valid bracket
    cond1 = (fb < fc and fb <= fa) or (fb < fa and fb <= fc)
    cond2 = xa < xb < xc or xc < xb < xa
    cond3 = np.isfinite(xa) and np.isfinite(xb) and np.isfinite(xc)

    if not (cond1 and cond2 and cond3):
        raise RuntimeError(
            "The algorithm terminated without finding a valid bracket. Consider trying different initial points."
        )

    return xa, xb, xc, fa, fb, fc, funcalls


@njit(error_model="numpy", fastmath=False)
def get_bracket_info(fun, args, brack):
    if brack is None:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(fun, args=args)
    elif len(brack) == 2:
        xa, xb, xc, fa, fb, fc, funcalls = bracket(fun, xa=brack[0], xb=brack[1], args=args)
    elif len(brack) == 3:
        xa, xb, xc = brack
        if xa > xc:
            xc, xa = xa, xc
        if not ((xa < xb) and (xb < xc)):
            raise ValueError(
                "Bracketing values (xa, xb, xc) do not fulfill this requirement: (xa < xb) and (xb < xc)"
            )
        fa = fun(xa, *args)
        fb = fun(xb, *args)
        fc = fun(xc, *args)
        if not ((fb < fa) and (fb < fc)):
            raise ValueError(
                "Bracketing values (xa, xb, xc) do not fulfill"
                " this requirement: (f(xb) < f(xa)) and (f(xb) < f(xc))"
            )

        funcalls = 3
    else:
        raise ValueError("Bracketing interval must be length 2 or 3 sequence.")

    return xa, xb, xc, fa, fb, fc, funcalls


@njit(error_model="numpy", fastmath=True)
def brent_minimize_scalar(
    fun: callable, xa: float, xb: float, rtol: float, max_iter: int, args: tuple
) -> tuple:
    # probably it can be not called each time
    xa, xb, xc, fa, fb, fc, funalls = get_bracket_info(fun, args, None)
    _mintol = 1.0e-11
    _cg = 0.3819660  # not really sure what this is
    # rtol = 1e-2 * rtol

    x = w = v = xb
    fw = fv = fx = fb
    if xa < xc:
        a = xa
        b = xc
    else:
        a = xc
        b = xa
    deltax = 0.0
    iter = 0

    while iter < max_iter:
        tol1 = rtol * abs(x) + _mintol
        tol2 = 2.0 * tol1
        xmid = 0.5 * (a + b)

        # check for convergence
        if abs(x - xmid) < (tol2 - 0.5 * (b - a)):
            break

        # XXX In the first iteration, rat is only bound in the true case
        # of this conditional. This used to cause an UnboundLocalError
        # (gh-4140). It should be set before the if (but to what?).

        if abs(deltax) <= tol1:
            if x >= xmid:
                deltax = a - x  # do a golden section step
            else:
                deltax = b - x
            rat = _cg * deltax
        else:  # do a parabolic step
            tmp1 = (x - w) * (fx - fv)
            tmp2 = (x - v) * (fx - fw)
            p = (x - v) * tmp2 - (x - w) * tmp1
            tmp2 = 2.0 * (tmp2 - tmp1)
            if tmp2 > 0.0:
                p = -p
            tmp2 = abs(tmp2)
            dx_temp = deltax
            deltax = rat
            # check parabolic fit
            if (
                (p > tmp2 * (a - x))
                and (p < tmp2 * (b - x))
                and (abs(p) < abs(0.5 * tmp2 * dx_temp))
            ):
                rat = p * 1.0 / tmp2  # if parabolic step is useful.
                u = x + rat
                if (u - a) < tol2 or (b - u) < tol2:
                    if xmid - x >= 0:
                        rat = tol1
                    else:
                        rat = -tol1
            else:
                if x >= xmid:
                    deltax = a - x  # if it's not do a golden section step
                else:
                    deltax = b - x
                rat = _cg * deltax

        if abs(rat) < tol1:  # update by at least tol1
            if rat >= 0:
                u = x + tol1
            else:
                u = x - tol1
        else:
            u = x + rat
        fu = fun(u, *args)  # calculate new output value
        funalls += 1

        if fu > fx:  # if it's bigger than current
            if u < x:
                a = u
            else:
                b = u
            if (fu <= fw) or (w == x):
                v = w
                w = u
                fv = fw
                fw = fu
            elif (fu <= fv) or (v == x) or (v == w):
                v = u
                fv = fu
        else:
            if u >= x:
                a = x
            else:
                b = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu

        iter += 1

    return x, fx
