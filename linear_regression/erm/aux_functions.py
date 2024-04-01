from numba import vectorize
from math import exp, log, log1p, tanh, cosh


@vectorize("float64(float64)")
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


@vectorize("float64(float64)")
def D_sigmoid(x: float) -> float:
    return exp(x) / (1 + exp(x)) ** 2


@vectorize("float64(float64)")
def hyperbolic_tangent(x: float) -> float:
    return tanh(x)


@vectorize("float64(float64)")
def D_hyperbolic_tangent(x: float) -> float:
    return 1 / (cosh(x) ** 2)


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
