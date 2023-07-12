from numba import vectorize
from numpy import sqrt, sign
from scipy.special import erf
from scipy.optimize import minimize_scalar
from scipy.integrate import quad, dblquad, tplquad
from ..aux_functions.moreau_proximal_losses import moreau_loss_Exponential, moreau_loss_Logistic
from ..aux_functions.loss_functions import DDz_logistic_loss, DDz_exponential_loss
from ..aux_functions.misc import gaussian
from ..utils.integration_utils import stability_integration_domains, stability_integration_domains_triple


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Stability functions for Regression                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# @vectorize
def stability_L2_decorrelated_regress(
    m: float,
    q: float,
    Σ: float,
    alpha: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
) -> float:
    return 1 - alpha * (Σ / (Σ + 1)) ** 2


# @vectorize
def stability_L1_decorrelated_regress(
    m: float,
    q: float,
    Σ: float,
    alpha: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
) -> float:
    return 1 - alpha * (
        (1 - percentage) * erf(Σ / sqrt(2 * (q + 1 + delta_in)))
        + percentage * erf(Σ / sqrt(2 * (q + beta**2 + delta_out)))
    )


# @vectorize
def stability_Huber_decorrelated_regress(
    m: float,
    q: float,
    Σ: float,
    alpha: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
    a: float,
) -> float:
    return 1 - alpha * (Σ / (Σ + 1)) ** 2 * (
        (1 - percentage) * erf(a * (Σ + 1) / sqrt(2 * (q + 1 + delta_in)))
        + percentage * erf(a * (Σ + 1) / sqrt(2 * (q + beta**2 + delta_out)))
    )


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Stability functions for Classification                  #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# # # # # Probit model                            # # # # #

# to check this function
# @vectorize
def stability_Hinge_probit_classif(m: float, q: float, Σ: float, alpha: float, delta: float) -> float:
    return 1 - alpha * (erf(1 / sqrt(2 * q)) + erf((Σ - 1) / sqrt(2 * q)))


def integrand_stability_Logistic_probit_classif(z, ω, w, m, q, Σ, delta):
    proximal = minimize_scalar(moreau_loss_Logistic, args=(sign(z + w), ω, Σ))["x"]
    Dproximal = 1 / (1 + Σ * DDz_logistic_loss(sign(z + w), proximal))
    return gaussian(z, 0, 1) * gaussian(ω, 0, q) * gaussian(w, 0, delta) * (Dproximal - 1) ** 2


def stability_Logistic_probit_classif(m: float, q: float, Σ: float, alpha: float, delta: float) -> float:
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
            args=(m, q, Σ, delta),
        )[0]
    return 1 - alpha * integral_value


def integrand_stability_Exponential_probit_classif(z, ω, m, q, Σ, delta):
    raise NotImplementedError
    proximal = minimize_scalar(moreau_loss_Exponential, args=(sign(z), ω, Σ))["x"]
    Dproximal = 1 / (1 + Σ * DDz_exponential_loss(sign(z), proximal))
    return gaussian(z, 0, 1) * gaussian(ω, 0, q) * (Dproximal - 1) ** 2


def stability_Exponential_probit_classif(m: float, q: float, Σ: float, alpha: float, delta: float) -> float:
    domains_z, domains_ω = stability_integration_domains()
    integral_value = 0.0
    for domain_z, domain_ω in zip(domains_z, domains_ω):
        integral_value += dblquad(
            integrand_stability_Exponential_probit_classif,
            domain_z[0],
            domain_z[1],
            domain_ω[0],
            domain_ω[1],
            args=(m, q, Σ, delta),
        )[0]

    return 1 - alpha * integral_value


# # # # # No Noise model                          # # # # #

# @vectorize
def stability_Hinge_no_noise_classif(m: float, q: float, Σ: float, alpha: float) -> float:
    return 1 - 0.5 * alpha * (erf(1 / sqrt(2 * q)) + erf((Σ - 1) / sqrt(2 * q)))


def integrand_stability_Logistic_no_noise_classif(z, ω, m, q, Σ):
    proximal = minimize_scalar(moreau_loss_Logistic, args=(sign(z), ω, Σ))["x"]
    Dproximal = 1 / (1 + Σ * DDz_logistic_loss(sign(z), proximal))
    return gaussian(z, 0, 1) * gaussian(ω, 0, q) * (Dproximal - 1) ** 2


def stability_Logistic_no_noise_classif(m: float, q: float, Σ: float, alpha: float) -> float:
    domains_z, domains_ω = stability_integration_domains()
    integral_value = 0.0
    for domain_z, domain_ω in zip(domains_z, domains_ω):
        integral_value += dblquad(
            integrand_stability_Logistic_no_noise_classif,
            domain_z[0],
            domain_z[1],
            domain_ω[0],
            domain_ω[1],
            args=(m, q, Σ),
        )[0]
    return 1 - alpha * integral_value


def integrand_stability_Exponential_no_noise_classif(z, ω, m, q, Σ):
    proximal = minimize_scalar(moreau_loss_Exponential, args=(sign(z), ω, Σ))["x"]
    Dproximal = 1 / (1 + Σ * DDz_exponential_loss(sign(z), proximal))
    return gaussian(z, 0, 1) * gaussian(ω, 0, q) * (Dproximal - 1) ** 2


def stability_Exponential_no_noise_classif(m: float, q: float, Σ: float, alpha: float) -> float:
    domains_z, domains_ω = stability_integration_domains()
    integral_value = 0.0
    for domain_z, domain_ω in zip(domains_z, domains_ω):
        integral_value += dblquad(
            integrand_stability_Exponential_no_noise_classif,
            domain_z[0],
            domain_z[1],
            domain_ω[0],
            domain_ω[1],
            args=(m, q, Σ),
        )[0]

    return 1 - alpha * integral_value


# # # # # Noise model                             # # # # #


# @vectorize
def stability_Hinge_noise_classif(m: float, q: float, Σ: float, alpha: float, delta: float) -> float:
    raise NotImplementedError