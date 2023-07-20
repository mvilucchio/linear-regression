from numba import vectorize
from .loss_functions import logistic_loss, exponential_loss


@vectorize("float64(float64, float64, float64)")
def proximal_Hinge_loss(y: float, omega: float, V: float) -> float:
    if y * omega <= 1 - V:
        return omega + V * y
    elif y * omega <= 1 and y * omega > 1 - V:
        return y
    else:
        return omega


@vectorize("float64(float64, float64, float64)")
def Domega_proximal_Hinge_loss(y: float, omega: float, V: float) -> float:
    if y * omega < 1 - V:
        return 1.0
    elif y * omega < 1:
        return 0.0
    else:
        return 1.0


@vectorize("float64(float64, float64, float64, float64)")
def moreau_loss_Logistic(x: float, y: float, omega: float, V: float) -> float:
    return (x - omega) ** 2 / (2 * V) + logistic_loss(y, x)


@vectorize("float64(float64, float64, float64, float64)")
def moreau_loss_Exponential(x: float, y: float, omega: float, V: float) -> float:
    return (x - omega) ** 2 / (2 * V) + exponential_loss(y, x)
