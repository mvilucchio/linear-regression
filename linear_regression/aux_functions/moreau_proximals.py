from numba import njit
from .loss_functions import (
    logistic_loss,
    DDz_logistic_loss,
    exponential_loss,
    DDz_exponential_loss,
    tukey_loss,
    DDz_tukey_loss,
    mod_tukey_loss_cubic,
    Dz_mod_tukey_loss_cubic,
    DDz_mod_tukey_loss_cubic,
    mod_tukey_loss_quad,
    Dz_mod_tukey_loss_quad,
    DDz_mod_tukey_loss_quad,
    cauchy_loss,
    Dz_cauchy_loss,
    DDz_cauchy_loss,
)
from .regularisation_functions import (
    power_regularisation,
    Dx_power_regularisation,
    DDx_power_regularisation,
    Dreg_param_power_regularisation,
    DxDreg_param_power_regularisation,
)
from ..utils.minimizers import brent_minimize_scalar
from ..utils.root_finding import all_brents
from . import MAX_ITER_BRENT_MINIMIZE, TOL_BRENT_MINIMIZE


BIG_NUMBER = 500_000_000
SMALL_NUMBER = 1e-15

# ---------------------------------------------------------------------------- #
#                            Loss functions proximal                           #
# ---------------------------------------------------------------------------- #


# -------------------------------- tukey loss -------------------------------- #
@njit(error_model="numpy", fastmath=False)
def moreau_loss_Tukey(x: float, y: float, omega: float, V: float, τ: float) -> float:
    return (x - omega) ** 2 / (2 * V) + tukey_loss(y, x, τ)


# maybe it is better some initialisation of the brent_minimize_scalar
@njit(error_model="numpy", fastmath=False)
def proximal_Tukey_loss(y: float, omega: float, V: float, τ: float) -> float:
    return brent_minimize_scalar(
        moreau_loss_Tukey,
        -BIG_NUMBER,
        BIG_NUMBER,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (y, omega, V, τ),
    )[0]


@njit(error_model="numpy", fastmath=False)
def Dω_proximal_Tukey_loss(y: float, omega: float, V: float, τ: float) -> float:
    proximal = brent_minimize_scalar(
        moreau_loss_Tukey,
        -BIG_NUMBER,
        BIG_NUMBER,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (y, omega, V, τ),
    )[0]
    return 1 / (1 + V * DDz_tukey_loss(y, proximal, τ))


# Matéo begins 

@njit(error_model="numpy", fastmath=False)
def moreau_loss_Tukey_TI(
    r: float, delta: float, V: float, τ: float
) -> float:
    """
    Moreau loss for the Tukey loss using the translation invariance property.
    """
    return (delta-r) ** 2 / 2 + V*tukey_loss(r, 0, τ)

@njit(error_model="numpy", fastmath=False)
def proximal_Tukey_loss_TI(delta :float, V : float, τ : float) -> float:
    """
    Proximal operator fo the Tukey loss using the translation invariance property.
    """

    return brent_minimize_scalar(
        moreau_loss_Tukey_TI,
        -BIG_NUMBER,
        BIG_NUMBER,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (delta, V, τ)
    )[0]

@njit(error_model="numpy", fastmath=False)
def Ddelta_proximal_Tukey_loss_TI(delta: float, V: float, τ: float) -> float:
    """
    Derivative of the proximal operator of the Tukey loss using the translation invariance property.
    """
    proximal = proximal_Tukey_loss_TI(delta, V, τ)

    return 1 / (1 + V * DDz_tukey_loss(proximal, 0, τ))

# Matéo ends

# ---------------------------- modified tukey loss --------------------------- #
@njit(error_model="numpy", fastmath=False)
def moreau_loss_Tukey_modified_cubic(
    x: float, y: float, omega: float, V: float, τ: float, c: float
) -> float:
    return (x - omega) ** 2 / (2 * V) + mod_tukey_loss_cubic(y, x, τ, c)


@njit(error_model="numpy", fastmath=False)
def proximal_loss_Tukey_modified_cubic(
    x: float, y: float, omega: float, V: float, τ: float, c: float
) -> float:
    return (x - omega) / V + Dz_mod_tukey_loss_cubic(y, x, τ, c)


@njit(error_model="numpy", fastmath=False)
def proximal_Tukey_modified_cubic(y: float, omega: float, V: float, τ: float, c: float) -> float:
    return all_brents(
        moreau_loss_Tukey_modified_cubic,
        proximal_loss_Tukey_modified_cubic,
        (y, omega, V, τ, c),
        -BIG_NUMBER,
        BIG_NUMBER,
        y - 3 * τ,
        y + 3 * τ,
        20,
        TOL_BRENT_MINIMIZE,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
    )


@njit(error_model="numpy", fastmath=False)
def Dω_proximal_Tukey_modified_cubic(y: float, omega: float, V: float, τ: float, c: float) -> float:
    proximal = all_brents(
        moreau_loss_Tukey_modified_cubic,
        proximal_loss_Tukey_modified_cubic,
        (y, omega, V, τ, c),
        -BIG_NUMBER,
        BIG_NUMBER,
        y - 3 * τ,
        y + 3 * τ,
        20,
        TOL_BRENT_MINIMIZE,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
    )
    return 1 / (1 + V * DDz_mod_tukey_loss_cubic(y, proximal, τ, c))


# ------------------------------ mod tukey loss ------------------------------ #
@njit(error_model="numpy", fastmath=False)
def moreau_loss_Tukey_modified_quad(
    x: float, y: float, omega: float, V: float, τ: float, c: float
) -> float:
    return (x - omega) ** 2 / (2 * V) + mod_tukey_loss_quad(y, x, τ, c)


@njit(error_model="numpy", fastmath=False)
def proximal_loss_Tukey_modified_quad(
    x: float, y: float, omega: float, V: float, τ: float, c: float
) -> float:
    return (x - omega) / V + Dz_mod_tukey_loss_quad(y, x, τ, c)


@njit(error_model="numpy", fastmath=False)
def proximal_Tukey_modified_quad(y: float, omega: float, V: float, τ: float, c: float) -> float:
    return all_brents(
        moreau_loss_Tukey_modified_quad,
        proximal_loss_Tukey_modified_quad,
        (y, omega, V, τ, c),
        -BIG_NUMBER,
        BIG_NUMBER,
        y - 1.5 * τ,
        y + 1.5 * τ,
        20,
        TOL_BRENT_MINIMIZE,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
    )


@njit(error_model="numpy", fastmath=False)
def Dω_proximal_Tukey_modified_quad(y: float, omega: float, V: float, τ: float, c: float) -> float:
    proximal = all_brents(
        moreau_loss_Tukey_modified_quad,
        proximal_loss_Tukey_modified_quad,
        (y, omega, V, τ, c),
        -BIG_NUMBER,
        BIG_NUMBER,
        y - 1.5 * τ,
        y + 1.5 * τ,
        20,
        TOL_BRENT_MINIMIZE,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
    )
    return 1 / (1 + V * DDz_mod_tukey_loss_quad(y, proximal, τ, c))


# -------------------------------- cauchy loss ------------------------------- #
@njit(error_model="numpy", fastmath=False)
def moreau_loss_Cauchy(x: float, y: float, omega: float, V: float, τ: float) -> float:
    return (x - omega) ** 2 / (2 * V) + cauchy_loss(y, x, τ)


@njit(error_model="numpy", fastmath=False)
def proximal_loss_Cauchy(x: float, y: float, omega: float, V: float, τ: float) -> float:
    return (x - omega) / V + Dz_cauchy_loss(y, x, τ)


@njit(error_model="numpy", fastmath=False)
def proximal_Cauchy(y: float, omega: float, V: float, τ: float) -> float:
    return all_brents(
        moreau_loss_Cauchy,
        proximal_loss_Cauchy,
        (y, omega, V, τ),
        -BIG_NUMBER,
        BIG_NUMBER,
        y - 3 * τ,
        y + 3 * τ,
        20,
        TOL_BRENT_MINIMIZE,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
    )


@njit(error_model="numpy", fastmath=False)
def Dω_proximal_Cauchy(y: float, omega: float, V: float, τ: float) -> float:
    proximal = all_brents(
        moreau_loss_Cauchy,
        proximal_loss_Cauchy,
        (y, omega, V, τ),
        -BIG_NUMBER,
        BIG_NUMBER,
        y - 3 * τ,
        y + 3 * τ,
        20,
        TOL_BRENT_MINIMIZE,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
    )
    return 1 / (1 + V * DDz_cauchy_loss(y, proximal, τ))


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


# ------------------------ adversarial logisic generic ----------------------- #
@njit(error_model="numpy", fastmath=False)
def moreau_loss_Logistic_adversarial(
    x: float, y: float, omega: float, V: float, P: float, ε: float, pstar: float
) -> float:
    return (x - omega) ** 2 / (2 * V) + logistic_loss(y, x - y * ε * P ** (1 / pstar))


@njit(error_model="numpy", fastmath=False)
def proximal_Logistic_adversarial(
    y: float, omega: float, V: float, P: float, ε: float, pstar: float
) -> float:
    return brent_minimize_scalar(
        moreau_loss_Logistic_adversarial,
        -BIG_NUMBER,
        BIG_NUMBER,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (y, omega, V, P, ε, pstar),
    )[0]


@njit(error_model="numpy", fastmath=False)
def Dω_proximal_Logistic_adversarial(
    y: float, omega: float, V: float, P: float, ε: float, pstar: float
) -> float:
    proximal = brent_minimize_scalar(
        moreau_loss_Logistic_adversarial,
        -BIG_NUMBER,
        BIG_NUMBER,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (y, omega, V, P, ε, pstar),
    )[0]
    return 1 / (1 + V * DDz_logistic_loss(y, proximal - y * ε * P ** (1 / pstar)))


# ------------------------- adversarial logistic loss ------------------------ #
@njit(error_model="numpy", fastmath=False)
def moreau_loss_Logistic_adversarial(
    x: float, y: float, omega: float, V: float, P: float, ε: float
) -> float:
    return (x - omega) ** 2 / (2 * V) + logistic_loss(y, x - y * P * ε)


@njit(error_model="numpy", fastmath=False)
def proximal_Logistic_adversarial(y: float, omega: float, V: float, P: float, ε: float) -> float:
    return brent_minimize_scalar(
        moreau_loss_Logistic_adversarial,
        -BIG_NUMBER,
        BIG_NUMBER,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (y, omega, V, P, ε),
    )[0]


@njit(error_model="numpy", fastmath=False)
def Dω_proximal_Logistic_adversarial(y: float, omega: float, V: float, P: float, ε: float) -> float:
    proximal = brent_minimize_scalar(
        moreau_loss_Logistic_adversarial,
        -BIG_NUMBER,
        BIG_NUMBER,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (y, omega, V, P, ε),
    )[0]
    return 1 / (1 + V * DDz_logistic_loss(y, proximal - y * ε * P))


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
    """
    Proximal operator of the L1 regularisation defined as:
    reg_param |β|
    """
    if Ɣ > reg_param:
        return (Ɣ - reg_param) / Λ
    elif Ɣ < -reg_param:
        return (Ɣ + reg_param) / Λ
    else:
        return 0.0


@njit(error_model="numpy", fastmath=False)
def DƔ_proximal_L1(Ɣ: float, Λ: float, reg_param: float) -> float:
    """
    Derivative of the proximal operator of the L1 regularisation defined as:
    reg_param |β|
    """
    if Ɣ > reg_param:
        return 1.0 / Λ
    elif Ɣ < -reg_param:
        return 1.0 / Λ
    else:
        return 0.0


@njit(error_model="numpy", fastmath=False)
def Dphat_proximal_L1(Ɣ: float, Λ: float, reg_param: float, Phat: float) -> float:
    """
    Derivative of the proximal operator of the L1 regularisation defined as:
    reg_param |β|
    """
    if Phat + reg_param < Ɣ:
        return (-Phat + Ɣ - reg_param) / Λ
    elif Ɣ < Phat + reg_param:
        return (-Phat - Ɣ - reg_param) / Λ
    else:
        return 0.0


@njit(error_model="numpy", fastmath=False)
def proximal_L2(Ɣ: float, Λ: float, reg_param: float) -> float:
    """
    Proximal operator of the L2 regularisation defined as:
    reg_param / 2 |β|^2
    """
    return Ɣ / (reg_param + Λ)


@njit(error_model="numpy", fastmath=False)
def DƔ_proximal_L2(Ɣ: float, Λ: float, reg_param: float) -> float:
    """
    Derivative of the proximal operator of the L2 regularisation defined as:
    reg_param / 2 |β|^2
    """
    return 1 / (reg_param + Λ)


@njit(error_model="numpy", fastmath=False)
def proximal_Elastic_net(Ɣ: float, Λ: float, lambda1: float, lambda2: float) -> float:
    """
    Proximal operator of the elastic net regularisation defined as:
    λ_1 |β| + (λ_2 / 2) |β|^2
    """
    if Ɣ > lambda1:
        return (Ɣ - lambda1) / ((lambda2 / Λ + 1) * Λ)
    elif Ɣ < -lambda1:
        return (Ɣ + lambda1) / ((lambda2 / Λ + 1) * Λ)
    else:
        return 0


@njit(error_model="numpy", fastmath=False)
def DƔ_proximal_Elastic_net(Ɣ: float, Λ: float, lambda1: float, lambda2: float) -> float:
    """
    Derivative of the proximal operator of the elastic net regularisation defined as:
    λ_1 |β| + (λ_2 / 2) |β|^2
    """
    if Ɣ > lambda1:
        return 1 / ((lambda2 / Λ + 1) * Λ)
    elif Ɣ < -lambda1:
        return 1 / ((lambda2 / Λ + 1) * Λ)
    else:
        return 0


@njit(error_model="numpy", fastmath=False)
def Dlambda1_proximal_Elastic_net(Ɣ: float, Λ: float, lambda1: float, lambda2: float) -> float:
    """
    Derivative of the proximal operator of the elastic net regularisation defined as:
    λ_1 |β| + (λ_2 / 2) |β|^2
    """
    if Ɣ > lambda1:
        return (-Ɣ + lambda1) / ((lambda2 / Λ + 1) * Λ)
    elif Ɣ < -lambda1:
        return (-Ɣ - lambda1) / ((lambda2 / Λ + 1) * Λ)
    else:
        return 0
