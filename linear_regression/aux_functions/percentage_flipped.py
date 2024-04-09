from math import pi, atan, sqrt, erf
from scipy.special import owens_t


def percentage_flipped_direct_space_true_label(
    m: float, q: float, rho: float, epsilon: float
) -> float:
    η = m**2 / (q * rho)
    return (
        atan(sqrt(η) / sqrt(1 - η)) / pi
        + 0.5 * erf(epsilon * sqrt(0.5 * (1 - η)))
        + 2 * owens_t(epsilon * sqrt((1 - η)), sqrt(η / (1 - η)))
    )


def percentage_flipped_direct_space(
    m: float, q: float, rho: float, epsilon: float
) -> float:
    return erf((epsilon * sqrt(2 - (2 * m**2) / (q * rho))) / 2.0)
