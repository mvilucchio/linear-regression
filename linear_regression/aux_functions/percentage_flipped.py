from math import pi, atan, sqrt, erf
from scipy.special import owens_t
from math import gamma


def percentage_flipped_direct_space_true_label(
    m: float, q: float, rho: float, epsilon: float
) -> float:
    η = m**2 / (q * rho)
    return (
        atan(sqrt(η) / sqrt(1 - η)) / pi
        + 0.5 * erf(epsilon * sqrt(0.5 * (1 - η)))
        + 2 * owens_t(epsilon * sqrt((1 - η)), sqrt(η / (1 - η)))
    )


def percentage_flipped_direct_space_true_min(
    m: float, q: float, rho: float, epsilon: float, p
) -> float:
    if p == "inf":
        Cpstar = 1 / sqrt(pi)
        return erf(epsilon * sqrt(1 - m**2 / (q * rho)) * Cpstar)
    else:
        pstar = p / (p - 1)
        Cpstar = (gamma((pstar + 1) / 2) / sqrt(pi)) ** (1 / pstar)
        return erf(epsilon * sqrt(1 - m**2 / (q * rho)) * Cpstar)


def percentage_flipped_direct_space_FGM(m: float, q: float, rho: float, epsilon: float, p) -> float:
    if p == "inf":
        return erf((epsilon * sqrt(1 - m**2 / (q * rho))) / sqrt(2.0))
    else:
        Cp = 2 ** (p / 2) * gamma((p + 1) / 2) / sqrt(pi)
        return erf(((epsilon / Cp ** (1 / p)) * sqrt(1 - m**2 / (q * rho))) / sqrt(2.0))


def percentage_flipped_linear_features_space_true_min(
    m: float, q: float, rho: float, epsilon: float, p, gamma_val: float
) -> float:
    if p == "inf":
        Cpstar = 1 / sqrt(pi) * gamma_val ** (1 / 2)
        return erf(epsilon * sqrt(1 - m**2 / (q * rho)) * Cpstar)
    else:
        pstar = p / (p - 1)
        Cpstar = (gamma((pstar + 1) / 2) / sqrt(pi)) ** (1 / pstar) * gamma_val ** (1 / 2 - 1 / p)
        return erf(epsilon * sqrt(1 - m**2 / (q * rho)) * Cpstar)
