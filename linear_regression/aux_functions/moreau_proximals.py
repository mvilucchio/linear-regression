from numba import njit
from .loss_functions import logistic_loss, exponential_loss, DDz_logistic_loss, DDz_exponential_loss
from .regularisation_functions import (
    power_regularisation,
    Dx_power_regularisation,
    DDx_power_regularisation,
    Dreg_param_power_regularisation,
    DxDreg_param_power_regularisation,
)
from ..utils.minimizers import brent_minimize_scalar
from . import MAX_ITER_BRENT_MINIMIZE, TOL_BRENT_MINIMIZE


BIG_NUMBER = 500_000_000
SMALL_NUMBER = 1e-15

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
def Dω_proximal_Hinge_loss(y: float, omega: float, V: float) -> float:
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
        -BIG_NUMBER,
        BIG_NUMBER,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (y, omega, V),
    )[0]


@njit(error_model="numpy", fastmath=False)
def Dω_proximal_Logistic_loss(y: float, omega: float, V: float) -> float:
    proximal = brent_minimize_scalar(
        moreau_loss_Logistic,
        -BIG_NUMBER,
        BIG_NUMBER,
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
        -BIG_NUMBER,
        BIG_NUMBER,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (y, omega, V, P, eps_t),
    )[0]


@njit(error_model="numpy", fastmath=False)
def Dω_proximal_Logistic_adversarial(
    y: float, omega: float, V: float, P: float, eps_t: float
) -> float:
    proximal = brent_minimize_scalar(
        moreau_loss_Logistic_adversarial,
        -BIG_NUMBER,
        BIG_NUMBER,
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
        -BIG_NUMBER,
        BIG_NUMBER,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (y, omega, V),
    )[0]


# ---------------------------------------------------------------------------- #
#                           Regularisation proximals                           #
# ---------------------------------------------------------------------------- #
@njit(error_model="numpy", fastmath=False)
def moreau_loss_sum_absolute(
    x: float, Ɣ: float, Λ: float, lambda_p: float, p: float, lambda_q: float, q: float
) -> float:
    return (
        0.5 * Λ * x**2
        - Ɣ * x
        + power_regularisation(x, p, lambda_p)
        + power_regularisation(x, q, lambda_q)
    )


@njit(error_model="numpy", fastmath=False)
def proximal_sum_absolute(
    Ɣ: float, Λ: float, lambda_p: float, p: float, lambda_q: float, q: float
) -> float:
    return brent_minimize_scalar(
        moreau_loss_sum_absolute,
        -BIG_NUMBER,
        BIG_NUMBER,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (Ɣ, Λ, lambda_p, p, lambda_q, q),
    )[0]


@njit(error_model="numpy", fastmath=False)
def DƔ_proximal_sum_absolute(
    Ɣ: float, Λ: float, lambda_p: float, p: float, lambda_q: float, q: float
) -> float:
    proximal = brent_minimize_scalar(
        moreau_loss_sum_absolute,
        -BIG_NUMBER,
        BIG_NUMBER,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (Ɣ, Λ, lambda_p, p, lambda_q, q),
    )[0]
    if abs(proximal) < SMALL_NUMBER:
        return 0.0
    return 1 / (
        Λ
        + DDx_power_regularisation(proximal, p, lambda_p)
        + DDx_power_regularisation(proximal, q, lambda_q)
    )


@njit(error_model="numpy", fastmath=False)
def Dlambdaq_proximal_sum_absolute(
    Ɣ: float, Λ: float, lambda_p: float, p: float, lambda_q: float, q: float
) -> float:
    proximal = brent_minimize_scalar(
        moreau_loss_sum_absolute,
        -BIG_NUMBER,
        BIG_NUMBER,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (Ɣ, Λ, lambda_p, p, lambda_q, q),
    )[0]
    return -(DxDreg_param_power_regularisation(proximal, q, lambda_q)) / (
        Λ
        + DDx_power_regularisation(proximal, p, lambda_p)
        + DDx_power_regularisation(proximal, q, lambda_q)
    )


@njit(error_model="numpy", fastmath=False)
def Dlambdaq_moreau_loss_sum_absolute(
    Ɣ: float, Λ: float, lambda_p: float, p: float, lambda_q: float, q: float
) -> float:
    dprox = Dlambdaq_proximal_sum_absolute(Ɣ, Λ, lambda_p, p, lambda_q, q)
    prox = brent_minimize_scalar(
        moreau_loss_sum_absolute,
        -BIG_NUMBER,
        BIG_NUMBER,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (Ɣ, Λ, lambda_p, p, lambda_q, q),
    )[0]
    if abs(prox) < SMALL_NUMBER:
        first_term = 0.0
    else:
        first_term = Dx_power_regularisation(prox, q, lambda_q) + Dx_power_regularisation(
            prox, p, lambda_p
        )
    return (first_term + Λ * prox - Ɣ) * dprox + Dreg_param_power_regularisation(prox, q, lambda_q)


@njit(error_model="numpy", fastmath=False)
def proximal_L1(Ɣ: float, Λ: float, reg_param: float) -> float:
    if Ɣ > reg_param:
        return (Ɣ - reg_param) / Λ
    elif Ɣ < -reg_param:
        return (Ɣ + reg_param) / Λ
    else:
        return 0.0


@njit(error_model="numpy", fastmath=False)
def DƔ_proximal_L1(Ɣ: float, Λ: float, reg_param: float) -> float:
    if Ɣ > reg_param:
        return 1.0 / Λ
    elif Ɣ < -reg_param:
        return 1.0 / Λ
    else:
        return 0.0


@njit(error_model="numpy", fastmath=False)
def Dphat_proximal_L1(Ɣ: float, Λ: float, reg_param: float, Phat: float) -> float:
    if Phat + reg_param < Ɣ:
        return (-Phat + Ɣ - reg_param) / Λ
    elif Ɣ < Phat + reg_param:
        return (-Phat - Ɣ - reg_param) / Λ
    else:
        return 0.0


@njit(error_model="numpy", fastmath=False)
def proximal_L2(Ɣ: float, Λ: float, reg_param: float) -> float:
    return Ɣ / (reg_param + Λ)


@njit(error_model="numpy", fastmath=False)
def DƔ_proximal_L2(Ɣ: float, Λ: float, reg_param: float) -> float:
    return 1 / (reg_param + Λ)


@njit(error_model="numpy", fastmath=False)
def proximal_Elastic_net(Ɣ: float, Λ: float, lambda_1: float, lambda_2: float) -> float:
    """
    Proximal operator of the elastic net regularisation defined as:
    λ_1 |β| + (λ_2 / 2) |β|^2
    """
    return
