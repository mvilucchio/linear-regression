from numba import vectorize, njit
from math import exp, log, tanh, cosh, sqrt, pow, pi
from numpy import log1p


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


# ----


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
    return 0.5 * y**2 * 0.5 * (1 + tanh(0.5 * y * z)) ** 2


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
