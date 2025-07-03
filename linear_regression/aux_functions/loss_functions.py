from numba import vectorize, njit
from math import exp, log, tanh, cosh, sqrt, pow, pi
from numpy import log1p
from ..utils.minimizers import brent_minimize_scalar
from . import MAX_ITER_BRENT_MINIMIZE, TOL_BRENT_MINIMIZE
from .likelihood_channel_functions import log_Z_out_Bayes_decorrelated_noise


BIG_NUMBER = 50_000_000


# ---------------------------------------------------------------------------- #
#                               Regression Losses                              #
# ---------------------------------------------------------------------------- #


@vectorize("float64(float64, float64)")
def l2_loss(y: float, z: float):
    return 0.5 * (y - z) ** 2


# ----
@vectorize("float64(float64, float64)")
def l1_loss(y: float, z: float) -> float:
    return abs(y - z)


# ----
@vectorize("float64(float64, float64, float64)")
def huber_loss(y: float, z: float, a: float) -> float:
    if abs(y - z) < a:
        return 0.5 * (y - z) ** 2
    else:
        return a * abs(y - z) - 0.5 * a**2


# ------------------------------ real tukey loss ----------------------------- #
@vectorize("float64(float64, float64, float64)")
def tukey_loss(y: float, z: float, τ: float) -> float:
    if abs(y - z) <= τ:
        return τ**2 / 6 * (1 - (1 - ((y - z) / τ) ** 2) ** 3)
    else:
        return τ**2 / 6


@vectorize("float64(float64, float64, float64)")
def Dz_tukey_loss(y: float, z: float, τ: float) -> float:
    if abs(y - z) <= τ:
        return -(y - z) * (1 - (y - z) ** 2 / τ**2) ** 2
    else:
        return 0.0


@vectorize("float64(float64, float64, float64)")
def DDz_tukey_loss(y: float, z: float, τ: float) -> float:
    if abs(y - z) <= τ:
        return 1 + (5 * (y - z) ** 4) / τ**4 - (6 * (y - z) ** 2) / τ**2
    else:
        return 0.0

# ---------------------- regularised tukey loss (cubic) ---------------------- #
@vectorize("float64(float64, float64, float64, float64)")
def mod_tukey_loss_cubic(y: float, z: float, τ: float, c: float) -> float:
    if abs(y - z) <= τ:
        return τ**2 / 6 * (1 - (1 - ((y - z) / τ) ** 2) ** 3)
    elif y - z > τ:
        return c * (y - z - τ) ** 3 + τ**2 / 6.0
    else:
        return τ**2 / 6.0 - c * (y - z + τ) ** 3


@vectorize("float64(float64, float64, float64, float64)")
def Dz_mod_tukey_loss_cubic(y: float, z: float, τ: float, c: float) -> float:
    if abs(y - z) <= τ:
        return -(y - z) * (1 - (y - z) ** 2 / τ**2) ** 2
    elif y - z > τ:
        return -3 * c * (-y + z + τ) ** 2
    else:
        return 3 * c * (y - z + τ) ** 2


@vectorize("float64(float64, float64, float64, float64)")
def DDz_mod_tukey_loss_cubic(y: float, z: float, τ: float, c: float) -> float:
    if abs(y - z) <= τ:
        return 1 + (5 * (y - z) ** 4) / τ**4 - (6 * (y - z) ** 2) / τ**2
    elif y - z > τ:
        return 6 * c * (y - z - τ)
    else:
        return -6 * c * (y - z + τ)


# -------------------- regularised tukey loss (quadratic) -------------------- #
@vectorize("float64(float64, float64, float64, float64)")
def mod_tukey_loss_quad(y: float, z: float, τ: float, c: float) -> float:
    if abs(y - z) <= τ:
        return τ**2 / 6 * (1 - (1 - ((y - z) / τ) ** 2) ** 3)
    elif y - z > τ:
        return c * (y - z - τ) ** 2 + τ**2 / 6.0
    else:
        return τ**2 / 6.0 + c * (y - z + τ) ** 2


@vectorize("float64(float64, float64, float64, float64)")
def Dz_mod_tukey_loss_quad(y: float, z: float, τ: float, c: float) -> float:
    if abs(y - z) <= τ:
        return -(y - z) * (1 - (y - z) ** 2 / τ**2) ** 2
    elif y - z > τ:
        return -2 * c * (y - z - τ)
    else:
        return -2 * c * (y - z + τ)


@vectorize("float64(float64, float64, float64, float64)")
def DDz_mod_tukey_loss_quad(y: float, z: float, τ: float, c: float) -> float:
    if abs(y - z) <= τ:
        return 1 + (5 * (y - z) ** 4) / τ**4 - (6 * (y - z) ** 2) / τ**2
    else:
        return 2 * c


# -------------------------------- cauchy loss ------------------------------- #
@vectorize("float64(float64, float64, float64)")
def cauchy_loss(y: float, z: float, τ: float) -> float:
    return 0.5 * τ**2 * log(1 + (y - z) ** 2 / τ**2)


@vectorize("float64(float64, float64, float64)")
def Dz_cauchy_loss(y: float, z: float, τ: float) -> float:
    return ((-y + z) * τ**2) / ((y - z) ** 2 + τ**2)


@vectorize("float64(float64, float64, float64)")
def DDz_cauchy_loss(y: float, z: float, τ: float) -> float:
    return (-((y - z) ** 2 * τ**2) + τ**4) / ((y - z) ** 2 + τ**2) ** 2


# --------------------------------- tahn loss -------------------------------- #

# Translation-invariant losses, with variable r
# Matéo begins here
@njit(error_model="numpy",fastmath=False)
def Dr_tukey_loss(r :float, τ: float) -> float:
    """
    Compute the derivative of the Tukey loss with respect to r.
    """
    if abs(r) <= τ:
        return r * (1 - r ** 2 / τ**2) ** 2
    else:
        return 0.0

@njit(error_model="numpy",fastmath=False)
def DDr_tukey_loss(r: float, τ: float) -> float:
    """
    Compute the second derivative of the Tukey loss with respect to r.
    """
    if abs(r) <= τ:
        return 1 + (5 * r ** 4) / τ**4 - (6 * r ** 2) / τ**2
    else:
        return 0.0

# Matéo ends here


# ---------------------------------------------------------------------------- #
#                             Classification Losses                            #
# ---------------------------------------------------------------------------- #


@vectorize("float64(float64, float64)")
def hinge_loss(y: float, z: float) -> float:
    return max(0.0, 1.0 - y * z)


# ----
# Compute log(1 + exp(x)) componentwise.
# inspired from sklearn and https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
# and http://fa.bianp.net/blog/2019/evaluate_logistic/
@vectorize("float64(float64)")
def log1pexp(x: float) -> float:
    if x <= -37:
        return exp(x)
    elif -37 < x <= -2:
        return log1p(exp(x))
    elif -2 < x <= 18:
        return log(1.0 + exp(x))
    elif 18 < x <= 33.3:
        return exp(-x) + x
    else:
        return x


@vectorize("float64(float64, float64)")
def logistic_loss(y: float, z: float) -> float:
    return log1pexp(-y * z)


@vectorize("float64(float64, float64)")
def Dz_logistic_loss(y: float, z: float) -> float:
    return -y / (1 + exp(y * z))


@vectorize("float64(float64, float64)")
def DDz_logistic_loss(y: float, z: float) -> float:
    return 0.5 * y**2 / (1 + cosh(y * z))


# ----
@vectorize("float64(float64, float64)")
def exponential_loss(y: float, z: float) -> float:
    return exp(-y * z)


@vectorize("float64(float64, float64)")
def Dz_exponential_loss(y: float, z: float) -> float:
    return -y * exp(-y * z)


@vectorize("float64(float64, float64)")
def DDz_exponential_loss(y: float, z: float) -> float:
    return y**2 * exp(-y * z)


# ----
@njit
def min_problem(
    ω: float,
    y: float,
    z: float,
    param: float,
    delta_in: float,
    delta_out: float,
    eps: float,
    beta: float,
) -> float:
    return 0.5 * (z - ω) ** 2 / param + log_Z_out_Bayes_decorrelated_noise(
        y, ω, param, delta_in, delta_out, eps, beta
    )


@njit
def optimal_loss_double_noise(
    y: float,
    z: float,
    param: float,
    delta_in: float,
    delta_out: float,
    eps: float,
    beta: float,
) -> float:
    return -brent_minimize_scalar(
        min_problem,
        -BIG_NUMBER,
        BIG_NUMBER,
        TOL_BRENT_MINIMIZE,
        MAX_ITER_BRENT_MINIMIZE,
        (y, z, param, delta_in, delta_out, eps, beta),
    )[0]
