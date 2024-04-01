from .aux_functions import sigmoid, D_sigmoid, log1pexp
from math import sqrt
from numba import njit, vectorize
from numpy import where, abs as np_abs, exp as np_exp, asarray, float64


# ----
# Regression losses
# ----


@njit
def l2_loss(xs, ys, proj_matrix, w, non_lin_f):
    sq_d = sqrt(xs.shape[-1])
    return 0.5 * (ys - non_lin_f(xs @ proj_matrix.T / sq_d) @ w) ** 2


def Dw_l2_loss(xs, ys, proj_matrix, w, non_lin_f, D_non_lin_f):
    raise NotImplementedError


@njit
def Dx_l2_loss(xs, ys, proj_matrix, w, non_lin_f, D_non_lin_f):
    if xs.ndim == 1:
        xs_r = xs.reshape(1, -1)
    else:
        xs_r = xs

    ys_r = asarray(ys, dtype=float64)

    sq_d = sqrt(xs_r.shape[-1])
    return (
        (ys - non_lin_f(xs @ proj_matrix.T / sq_d) @ w)[:, None]
        * (D_non_lin_f(xs_r @ proj_matrix.T / sq_d) @ (w[:, None] * proj_matrix))
    ) / sq_d


def l1_loss(xs, ys, proj_matrix, w, non_lin_f):
    sq_d = sqrt(xs.shape[-1])
    return np_abs(ys - non_lin_f(xs @ proj_matrix.T / sq_d) @ w)


def huber_loss(xs, ys, proj_matrix, w, non_lin_f, a=1.0):
    sq_d = sqrt(xs.shape[-1])
    diff = ys - non_lin_f(xs @ proj_matrix.T / sq_d) @ w
    return where(np_abs(diff) <= a, 0.5 * diff**2, a * (np_abs(diff) - 0.5 * a))


# ----
# Classification losses
# ----


@njit
def hinge_loss(xs, ys, proj_matrix, w, non_lin_f):
    sq_d = sqrt(xs.shape[-1])
    return where(
        ys * (non_lin_f(xs @ proj_matrix.T / sq_d) @ w) <= 1,
        1 - ys * (non_lin_f(xs @ proj_matrix.T / sq_d) @ w),
        0.0,
    )


def Dw_hinge_loss(xs, ys, proj_matrix, w, non_lin_f, D_non_lin_f):
    raise NotImplementedError


def Dx_hinge_loss(xs, ys, proj_matrix, w, non_lin_f, D_non_lin_f):
    raise NotImplementedError


@njit
def exponential_loss(xs, ys, proj_matrix, w, non_lin_f):
    sq_d = sqrt(xs.shape[-1])
    return np_exp(-ys * (non_lin_f(xs @ proj_matrix.T / sq_d) @ w))


def Dw_exponential_loss(xs, ys, proj_matrix, w, non_lin_f, D_non_lin_f):
    raise NotImplementedError


def Dx_exponential_loss(xs, ys, proj_matrix, w, non_lin_f, D_non_lin_f):
    raise NotImplementedError


@njit
def logistic_loss(xs, ys, proj_matrix, w, non_lin_f):
    d = xs.shape[-1]
    return log1pexp(-ys * (non_lin_f(xs @ proj_matrix.T / sqrt(d)) @ w))


def Dw_logistic_loss(xs, ys, proj_matrix, w, non_lin_f):
    d = xs.shape[-1]
    raise NotImplementedError


@njit
def Dx_logistic_loss(xs, ys, proj_matrix, w, non_lin_f, D_non_lin_f):
    if xs.ndim == 1:
        xs_r = xs.reshape(1, -1)
    else:
        xs_r = xs

    ys_r = asarray(ys, dtype=float64)

    sq_d = sqrt(xs_r.shape[-1])
    return (
        (-ys_r * sigmoid(-ys_r * (non_lin_f(xs_r @ proj_matrix.T / sq_d) @ w)))[:, None]
        * (D_non_lin_f(xs_r @ proj_matrix.T / sq_d) @ (w[:, None] * proj_matrix))
    ) / sq_d
