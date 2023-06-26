from numba import vectorize
# from math import erf, sqrt
from numpy import sqrt
from scipy.special import erf


# @vectorize
def stability_ridge(
    m: float,
    q: float,
    sigma: float,
    alpha: float,
    reg_param: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
):
    return 1 - alpha * (sigma / (sigma + 1)) ** 2


# @vectorize
def stability_l1_l2(
    m: float,
    q: float,
    sigma: float,
    alpha: float,
    reg_param: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
):
    return 1 - alpha * (
        (1 - percentage) * erf(sigma / sqrt(2 * (q + 1 + delta_in)))
        + percentage * erf(sigma / sqrt(2 * (q + beta**2 + delta_out)))
    )


# @vectorize
def stability_huber(
    m: float,
    q: float,
    sigma: float,
    alpha: float,
    reg_param: float,
    delta_in: float,
    delta_out: float,
    percentage: float,
    beta: float,
    a: float,
):
    return 1 - alpha * (sigma / (sigma + 1))**2 * (
        (1- percentage) * erf(a * (sigma + 1) / sqrt(2 * (q + 1 + delta_in))) + 
        percentage * erf(a * (sigma + 1) / sqrt(2 * (q + beta**2 + delta_out)))
    )
