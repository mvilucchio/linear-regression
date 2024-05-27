from numba import vectorize, njit
from .loss_functions import logistic_loss, exponential_loss, DDz_logistic_loss, DDz_exponential_loss
from ..utils.minimizers import brent_minimize_scalar
from . import MAX_ITER_BRENT_MINIMIZE, TOL_BRENT_MINIMIZE


# ---------------------------------------------------------------------------- #
#                            Loss functions proximal                           #
# ---------------------------------------------------------------------------- #


# -------------------------------- hinge loss -------------------------------- #
@njit(error_model="numpy", fastmath=False)
def proximal_Hinge_loss(y: float, omega: float, V: float) -> float:
    if y * omega <= 1 - V:
        return omega + V * y
    elif 1 - V < y * omega <= 1:
        return y
    else:
        return omega


@njit(error_model="numpy", fastmath=False)
def Dproximal_Hinge_loss(y: float, omega: float, V: float) -> float:
    if y * omega < 1 - V:
        return 1.0
    elif y * omega < 1:
        return 0.0
    else:
        return 1.0


# ------------------------------- logistic loss ------------------------------ #
@njit(error_model="numpy", fastmath=False)
def moreau_loss_Logistic(x: float, y: float, omega: float, V: float) -> float:
    return (x - omega) ** 2 / (2 * V) + logistic_loss(y, x)


@njit(error_model="numpy", fastmath=False)
def proximal_Logistic_loss(y: float, omega: float, V: float) -> float:
    return brent_minimize_scalar(
        moreau_loss_Logistic,
        -5000,
        5000,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (y, omega, V),
    )[0]


@njit(error_model="numpy", fastmath=False)
def Dproximal_Logistic_loss(y: float, omega: float, V: float) -> float:
    proximal = brent_minimize_scalar(
        moreau_loss_Logistic,
        -5000,
        5000,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (y, omega, V),
    )[0]
    return 1 / (1 + V * DDz_logistic_loss(y, proximal))


# ------------------------- adversarial logistic loss ------------------------ #
@njit(error_model="numpy", fastmath=False)
def moreau_loss_Logistic_adversarial(
    x: float, y: float, omega: float, V: float, P: float, eps_t: float
) -> float:
    return (x - omega) ** 2 / (2 * V) + logistic_loss(y, x - y * P * eps_t)


@njit(error_model="numpy", fastmath=False)
def proximal_Logistic_adversarial(
    y: float, omega: float, V: float, P: float, eps_t: float
) -> float:
    return brent_minimize_scalar(
        moreau_loss_Logistic_adversarial,
        -5000,
        5000,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (y, omega, V, P, eps_t),
    )[0]


@njit(error_model="numpy", fastmath=False)
def Dproximal_Logistic_adversarial(
    y: float, omega: float, V: float, P: float, eps_t: float
) -> float:
    proximal = brent_minimize_scalar(
        moreau_loss_Logistic_adversarial,
        -5000,
        5000,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (y, omega, V, P, eps_t),
    )[0]
    return 1 / (1 + V * DDz_logistic_loss(y, proximal - y * eps_t * P))


# ----------------------------- exponential loss ----------------------------- #
# @vectorize("float64(float64, float64, float64, float64)")
@njit(error_model="numpy", fastmath=False)
def moreau_loss_Exponential(x: float, y: float, omega: float, V: float) -> float:
    return (x - omega) ** 2 / (2 * V) + exponential_loss(y, x)


# @vectorize("float64(float64, float64, float64)")
@njit(error_model="numpy", fastmath=False)
def proximal_Exponential_loss(y: float, omega: float, V: float) -> float:
    return brent_minimize_scalar(
        moreau_loss_Exponential,
        -5000,
        5000,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (y, omega, V),
    )[0]


# ---------------------------------------------------------------------------- #
#                           Regularisation proximals                           #
# ---------------------------------------------------------------------------- #
# @vectorize("float64(float64, float64, float64, float64, float64, float64, float64)")
@njit(error_model="numpy", fastmath=False)
def moreau_loss_sum_absolute_values(
    x: float, gamma: float, Λ: float, lambda_p: float, p: float, lambda_q: float, q: float
) -> float:
    return 0.5 * Λ * (x - gamma / Λ) ** 2 + lambda_p * abs(x) ** p + lambda_q * abs(x) ** q


@njit(error_model="numpy", fastmath=False)
def proximal_sum_absolute_values(
    gamma: float, Λ: float, lambda_p: float, p: float, lambda_q: float, q: float
) -> float:
    return brent_minimize_scalar(
        moreau_loss_sum_absolute_values,
        -50000,
        50000,
        TOL_BRENT_MINIMIZE,
        1000,
        (gamma, Λ, lambda_p, p, lambda_q, q),
    )[0]


def proximal_L1(gamma: float, Λ: float, lambda_p: float) -> float:
    raise NotImplementedError


def proximal_L2(gamma: float, Λ: float, lambda_p: float) -> float:
    raise NotImplementedError
