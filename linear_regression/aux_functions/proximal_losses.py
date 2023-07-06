from numba import vectorize


# @vectorize("float64(float64, float64)")
def proximal_hinge_loss(y: float, omega: float, V: float) -> float:
    if y * omega < 1 - V:
        return omega + V
    elif y * omega < 1:
        return y
    else:
        return omega
