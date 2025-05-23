from numba import vectorize, njit
from numpy import sqrt, sign
from math import erf, exp, pi

# from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad, dblquad, tplquad
from ..aux_functions.moreau_proximals import (
    moreau_loss_Exponential,
    moreau_loss_Logistic,
)
from ..aux_functions.loss_functions import DDz_logistic_loss, DDz_exponential_loss
from ..aux_functions.misc import gaussian
from ..utils.integration_utils import (
    stability_integration_domains,
    stability_integration_domains_triple,
)


BIG_NUMBER = 6

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Stability functions for Regression                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# those are wrong, the integral needs to consider the correaltion between the ω and z


@vectorize
def stability_L2_decorrelated_regress(
    m: float,
    q: float,
    V: float,
    alpha: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
) -> float:
    return 1 - alpha * (V / (V + 1)) ** 2


@vectorize
def stability_L1_decorrelated_regress(
    m: float,
    q: float,
    V: float,
    alpha: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
) -> float:
    # return 1 - alpha * (
    #     (1 - percentage) * erf(V / sqrt(2 * (q + 1 + delta_in)))
    #     + percentage * erf(V / sqrt(2 * (q + beta**2 + delta_out)))
    # )
    return 1 - alpha * (
        (1 - percentage) * erf(V / sqrt(2 * (q - 2 * m + delta_in + 1)))
        + percentage * erf(V / sqrt(2 * (q - 2 * m * beta + delta_out + beta**2)))
    )


@vectorize
def stability_Huber_decorrelated_regress(
    m: float,
    q: float,
    V: float,
    alpha: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
    a: float,
) -> float:
    # return 1 - alpha * (V / (V + 1)) ** 2 * (
    #     (1 - percentage) * erf(a * (V + 1) / sqrt(2 * (q + 1 + delta_in)))
    #     + percentage * erf(a * (V + 1) / sqrt(2 * (q + beta**2 + delta_out)))
    # )
    return 1 - alpha * (V / (V + 1)) ** 2 * (
        (1 - percentage) * erf(a * (V + 1) / sqrt(2 * (q - 2 * m + delta_in + 1)))
        + percentage * erf(a * (V + 1) / sqrt(2 * (q - 2 * m * beta + delta_out + beta**2)))
    )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Stability functions for Classification                  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # Probit model                            # # # # #


# @njit(error_model="numpy", fastmath=False)
def positive_integrand_stability_Hinge_probit_classif(
    w: float, z: float, m: float, q: float, V: float, Δ: float
) -> float:
    # print(w, z, m, q, V, Δ)
    denom = sqrt(2 * (q - m**2))
    return (
        0.5
        * gaussian(w, 0, Δ)
        * gaussian(z, 0, 1)
        * (erf((-1 + m * z + V) / denom) - erf((-1 + m * z) / denom))
    )


# @njit(error_model="numpy", fastmath=False)
def negative_integrand_stability_Hinge_probit_classif(
    w: float, z: float, m: float, q: float, V: float, Δ: float
) -> float:
    # print(w, z, m, q, V, Δ)
    denom = sqrt(2 * (q - m**2))
    return (
        0.5
        * gaussian(w, 0, Δ)
        * gaussian(z, 0, 1)
        * (erf((1 + m * z) / denom) - erf((1 + m * z - V) / denom))
    )


def stability_Hinge_probit_classif(m: float, q: float, V: float, alpha: float, Δ: float) -> float:
    domains = [
        [
            [-BIG_NUMBER * sqrt(Δ), BIG_NUMBER * sqrt(Δ)],
            [lambda w: -BIG_NUMBER, lambda w: -w],
        ],
        [
            [-BIG_NUMBER * sqrt(Δ), BIG_NUMBER * sqrt(Δ)],
            [lambda w: -w, lambda w: BIG_NUMBER],
        ],
    ]
    integral_value = 0.0
    for domain_w, domain_z in domains:
        integral_value += dblquad(
            negative_integrand_stability_Hinge_probit_classif,
            domain_w[0],
            domain_w[1],
            domain_z[0],
            domain_z[1],
            args=(m, q, V, Δ),
        )[0]
    return 1 - alpha * integral_value


# -----
def integrand_stability_Logistic_probit_classif(z, ω, w, m, q, V, delta):
    proximal = minimize_scalar(moreau_loss_Logistic, args=(sign(z + w), ω, V))["x"]
    Dproximal = 1 / (1 + V * DDz_logistic_loss(sign(z + w), proximal))
    return (
        exp(-0.5 * (q * z**2 - 2 * m * z * ω + ω**2) / (q - m**2))
        / (2 * pi * sqrt(q - m**2))
        * gaussian(w, 0, delta)
        * (Dproximal - 1) ** 2
    )


def stability_Logistic_probit_classif(
    m: float, q: float, V: float, alpha: float, delta: float
) -> float:
    domains_z, domains_ω, domains_w = stability_integration_domains_triple()
    integral_value = 0.0
    for domain_z, domain_ω, domain_w in zip(domains_z, domains_ω, domains_w):
        integral_value += tplquad(
            integrand_stability_Logistic_probit_classif,
            domain_z[0],
            domain_z[1],
            domain_ω[0],
            domain_ω[1],
            domain_w[0],
            domain_w[1],
            args=(m, q, V, delta),
            epsabs=1e-3,
            epsrel=1e-2,
        )[0]
    return 1 - alpha * integral_value


# -----
def integrand_stability_Exponential_probit_classif(z, ω, w, m, q, V, delta):
    proximal = minimize_scalar(moreau_loss_Exponential, args=(sign(z + w), ω, V))["x"]
    Dproximal = 1 / (1 + V * DDz_exponential_loss(sign(z + w), proximal))
    return (
        exp(-0.5 * (q * z**2 - 2 * m * z * ω + ω**2) / (q - m**2))
        / (2 * pi * sqrt(q - m**2))
        * gaussian(w, 0, delta)
        * (Dproximal - 1) ** 2
    )


def stability_Exponential_probit_classif(
    m: float, q: float, V: float, alpha: float, delta: float
) -> float:
    domains_z, domains_ω, domains_w = stability_integration_domains_triple()
    integral_value = 0.0
    for domain_z, domain_ω, domain_w in zip(domains_z, domains_ω, domains_w):
        integral_value += tplquad(
            integrand_stability_Exponential_probit_classif,
            domain_z[0],
            domain_z[1],
            domain_ω[0],
            domain_ω[1],
            domain_w[0],
            domain_w[1],
            args=(m, q, V, delta),
            epsabs=1e-3,
            epsrel=1e-2,
        )[0]
    return 1 - alpha * integral_value


# # # # # No Noise model                          # # # # #


@njit(error_model="numpy", fastmath=False)
def positive_integrand_stability_Hinge_no_noise_classif(z, m, q, V):
    denom = sqrt(2 * (q - m**2))
    return 0.5 * gaussian(z, 0, 1) * (erf((-1 + m * z + V) / denom) - erf((-1 + m * z) / denom))


@njit(error_model="numpy", fastmath=False)
def negative_integrand_stability_Hinge_no_noise_classif(z, m, q, V):
    denom = sqrt(2 * (q - m**2))
    return 0.5 * gaussian(z, 0, 1) * (erf((1 + m * z) / denom) - erf((1 + m * z - V) / denom))


def stability_Hinge_no_noise_classif(m: float, q: float, V: float, alpha: float) -> float:
    integral_value = quad(
        negative_integrand_stability_Hinge_no_noise_classif,
        -BIG_NUMBER,
        0,
        args=(m, q, V),
    )[0]
    integral_value += quad(
        positive_integrand_stability_Hinge_no_noise_classif,
        0,
        BIG_NUMBER,
        args=(m, q, V),
    )[0]
    return 1 - alpha * integral_value


# -----
def integrand_stability_Logistic_no_noise_classif(z, ω, m, q, V):
    proximal = minimize_scalar(moreau_loss_Logistic, args=(sign(z), ω, V))["x"]
    Dproximal = 1 / (1 + V * DDz_logistic_loss(sign(z), proximal))
    return (
        exp(-0.5 * (q * z**2 - 2 * m * z * ω + ω**2) / (q - m**2))
        / (2 * pi * sqrt(q - m**2))
        * (Dproximal - 1) ** 2
    )


def stability_Logistic_no_noise_classif(m: float, q: float, V: float, alpha: float) -> float:
    domains_z, domains_ω = stability_integration_domains()

    integral_value = 0.0
    for domain_z, domain_ω in zip(domains_z, domains_ω):
        integral_value += dblquad(
            integrand_stability_Logistic_no_noise_classif,
            domain_z[0],
            domain_z[1],
            domain_ω[0],
            domain_ω[1],
            args=(m, q, V),
        )[0]
    return 1 - alpha * integral_value


# -----
def integrand_stability_Exponential_no_noise_classif(z, ω, m, q, V):
    proximal = minimize_scalar(moreau_loss_Exponential, args=(sign(z), ω, V))["x"]
    Dproximal = 1 / (1 + V * DDz_exponential_loss(sign(z), proximal))
    return (
        exp(-0.5 * (q * z**2 - 2 * m * z * ω + ω**2) / (q - m**2))
        / (2 * pi * sqrt(q - m**2))
        * (Dproximal - 1) ** 2
    )


def stability_Exponential_no_noise_classif(m: float, q: float, V: float, alpha: float) -> float:
    domains_z, domains_ω = stability_integration_domains()
    integral_value = 0.0
    for domain_z, domain_ω in zip(domains_z, domains_ω):
        integral_value += dblquad(
            integrand_stability_Exponential_no_noise_classif,
            domain_z[0],
            domain_z[1],
            domain_ω[0],
            domain_ω[1],
            args=(m, q, V),
        )[0]

    return 1 - alpha * integral_value


# # # # # Noise model                             # # # # #


# @vectorize
def stability_Hinge_noise_classif(
    m: float, q: float, V: float, alpha: float, delta: float
) -> float:
    raise NotImplementedError
