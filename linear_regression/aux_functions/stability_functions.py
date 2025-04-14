from numba import vectorize, njit
import numpy as np
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

from .likelihood_channel_functions import Z_out_Bayes_decorrelated_noise
from .moreau_proximals import proximal_Tukey_modified_quad
from .loss_functions import DDz_mod_tukey_loss_quad

BIG_NUMBER = 6
DEFAULT_INTEGRATION_BOUND = 15.0

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
    #     (1 - percentage) * erf(V / np.sqrt(2 * (q + 1 + delta_in)))
    #     + percentage * erf(V / np.sqrt(2 * (q + beta**2 + delta_out)))
    # )
    return 1 - alpha * (
        (1 - percentage) * erf(V / np.sqrt(2 * (q - 2 * m + delta_in + 1)))
        + percentage * erf(V / np.sqrt(2 * (q - 2 * m * beta + delta_out + beta**2)))
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
    #     (1 - percentage) * erf(a * (V + 1) / np.sqrt(2 * (q + 1 + delta_in)))
    #     + percentage * erf(a * (V + 1) / np.sqrt(2 * (q + beta**2 + delta_out)))
    # )
    return 1 - alpha * (V / (V + 1)) ** 2 * (
        (1 - percentage) * erf(a * (V + 1) / np.sqrt(2 * (q - 2 * m + delta_in + 1)))
        + percentage * erf(a * (V + 1) / np.sqrt(2 * (q - 2 * m * beta + delta_out + beta**2)))
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
    denom = np.sqrt(2 * (q - m**2))
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
    denom = np.sqrt(2 * (q - m**2))
    return (
        0.5
        * gaussian(w, 0, Δ)
        * gaussian(z, 0, 1)
        * (erf((1 + m * z) / denom) - erf((1 + m * z - V) / denom))
    )


def stability_Hinge_probit_classif(m: float, q: float, V: float, alpha: float, Δ: float) -> float:
    domains = [
        [
            [-BIG_NUMBER * np.sqrt(Δ), BIG_NUMBER * np.sqrt(Δ)],
            [lambda w: -BIG_NUMBER, lambda w: -w],
        ],
        [
            [-BIG_NUMBER * np.sqrt(Δ), BIG_NUMBER * np.sqrt(Δ)],
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
    proximal = minimize_scalar(moreau_loss_Logistic, args=(np.sign(z + w), ω, V))["x"]
    Dproximal = 1 / (1 + V * DDz_logistic_loss(np.sign(z + w), proximal))
    return (
        exp(-0.5 * (q * z**2 - 2 * m * z * ω + ω**2) / (q - m**2))
        / (2 * pi * np.sqrt(q - m**2))
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
    proximal = minimize_scalar(moreau_loss_Exponential, args=(np.sign(z + w), ω, V))["x"]
    Dproximal = 1 / (1 + V * DDz_exponential_loss(np.sign(z + w), proximal))
    return (
        exp(-0.5 * (q * z**2 - 2 * m * z * ω + ω**2) / (q - m**2))
        / (2 * pi * np.sqrt(q - m**2))
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
    denom = np.sqrt(2 * (q - m**2))
    return 0.5 * gaussian(z, 0, 1) * (erf((-1 + m * z + V) / denom) - erf((-1 + m * z) / denom))


@njit(error_model="numpy", fastmath=False)
def negative_integrand_stability_Hinge_no_noise_classif(z, m, q, V):
    denom = np.sqrt(2 * (q - m**2))
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
    proximal = minimize_scalar(moreau_loss_Logistic, args=(np.sign(z), ω, V))["x"]
    Dproximal = 1 / (1 + V * DDz_logistic_loss(np.sign(z), proximal))
    return (
        exp(-0.5 * (q * z**2 - 2 * m * z * ω + ω**2) / (q - m**2))
        / (2 * pi * np.sqrt(q - m**2))
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
    proximal = minimize_scalar(moreau_loss_Exponential, args=(np.sign(z), ω, V))["x"]
    Dproximal = 1 / (1 + V * DDz_exponential_loss(np.sign(z), proximal))
    return (
        exp(-0.5 * (q * z**2 - 2 * m * z * ω + ω**2) / (q - m**2))
        / (2 * pi * np.sqrt(q - m**2))
        * (Dproximal - 1) ** 2
    )


def stability_Exponential_no_noise_classif(m: float, q: float, V: float, alpha: float) -> float:
    print(" m = {:.3f} q = {:.3f} q - m^2 = {:.3f}".format(m, q, q - m**2))
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

#########################################################
# Replica symetry condition
###########################################################

@njit(error_model="numpy", fastmath=False)
def RS_int_mod_Tukey_decorrelated_noise(
    # Dblquad passe d'abord la variable d'intégration interne (y ici), puis externe (xi)
    y: float,
    xi: float,
    # Ensuite les args dans l'ordre
    q: float,
    m: float,
    V: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
    tau: float,
    c: float,
):
    """
    Intégrande pour la condition RS avec la loss Tukey modifiée (quadratique).
    """
    # Vérifications pour la stabilité numérique
    if q <= m**2 or q < 1e-12: return 0.0
    eta = m**2 / q
    if 1 - eta < 1e-12: return 0.0 # Évite division par zéro ou sqrt(neg) dans Z_out

    proximal = proximal_Tukey_modified_quad(y, np.sqrt(q) * xi, V, tau, c)

    ddz_loss = DDz_mod_tukey_loss_quad(y, proximal, tau, c)

    if 1 + V * ddz_loss < 1e-12: return 0.0 # Éviter division par zéro
    Dproximal = 1.0 / (1.0 + V * ddz_loss)

    # Calcul de Z_out
    z_out_val = Z_out_Bayes_decorrelated_noise(
        y, np.sqrt(eta) * xi, 1.0 - eta, delta_in, delta_out, percentage, beta
    )

    return (
         gaussian(xi, 0, 1) 
         * z_out_val
         * (Dproximal - 1.0)**2
         / V**2
     )

def RS_E2_mod_Tukey_decorrelated_noise(
    m: float,
    q: float,
    V: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
    tau: float,
    c: float,
    integration_bound: float = DEFAULT_INTEGRATION_BOUND,
    integration_epsabs: float = 1e-12,
    integration_epsrel: float = 1e-10,
) -> float:
    """
    Calcule l'intégrale double pour la condition RS avec Tukey modifiée (quad).
    """
    xi_dom = [-integration_bound, integration_bound]
    y_dom = [-integration_bound, integration_bound]

    # Vérifications initiales
    if q <= m**2 or q < 1e-12 or V < 1e-12:
        return np.nan

    args_integrand = (
        q, m, V, delta_in, delta_out, percentage, beta, tau, c
    )

    try:
        int_value_RS, abserr_RS = dblquad(
            RS_int_mod_Tukey_decorrelated_noise,
            xi_dom[0], xi_dom[1],
            #y_dom[0], y_dom[1],
            lambda xi: -integration_bound,
            lambda xi: integration_bound,
            args=args_integrand,
            epsabs=integration_epsabs,
            epsrel=integration_epsrel
        )
    except Exception as e:
        print(f"\nErreur dans dblquad (RS): {e} pour m={m:.3f}, q={q:.3f}, V={V:.3f}")
        int_value_RS = np.nan

    # Gérer les NaN ou infinis potentiels retournés par dblquad
    if not np.isfinite(int_value_RS):
        print(f"\nAttention : Échec d'intégration (résultat non fini) pour m={m:.3f}, q={q:.3f}, V={V:.3f}. Retourne NaN.")
        return np.nan

    # Retourne seulement la valeur de l'intégrale
    return int_value_RS
