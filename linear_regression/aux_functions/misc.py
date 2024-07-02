from math import exp, sqrt, acos, log1p, log, cosh, tanh, erf, erfc, inf
from numpy import pi, arccos, dot, ndarray, array
from numpy import max as np_max
from numpy import abs as np_abs
import numpy as np
from numpy.linalg import norm, det, inv
from numpy.random import normal
from scipy.integrate import quad
from numba import vectorize, njit


def sample_vector_informed(ground_truth_theta, m, q):
    """
    Generate an informed random vector for a linear regression problem.

    The random vector is generated by taking a normally distributed vector and projecting it onto the ground truth
    parameter vector, then scaling the projection and the orthogonal component to have the desired magnitude.

    Parameters
    ----------
    ground_truth_theta : numpy.ndarray
        The ground truth parameter vector for the linear regression problem.
    m : float
        The magnitude of the projection of the random vector onto the ground truth parameter vector.
    q : float
        The desired magnitude of the random vector.

    Returns
    -------
    numpy.ndarray
        An informed random vector for the linear regression problem.
    """
    random_vector = normal(size=ground_truth_theta.shape)

    projection = (
        dot(random_vector, ground_truth_theta)
        / dot(ground_truth_theta, ground_truth_theta)
        * ground_truth_theta
    )

    orthogonal = random_vector - projection
    scaled_projection = m * ground_truth_theta
    scaled_orthogonal = sqrt(q - m**2) / norm(orthogonal) * orthogonal

    init_w = scaled_projection + scaled_orthogonal

    return init_w


def sample_vector_random(n_features, squared_radius):
    """
    Generate a random vector of length `n_features` with Euclidean norm `sqrt(squared_radius)`.

    Parameters
    ----------
    n_features : int
        The length of the random vector.
    squared_radius : float
        The squared Euclidean norm of the random vector.

    Returns
    -------
    numpy.ndarray
        A random vector of length `n_features` with Euclidean norm `sqrt(squared_radius)`.
    """
    random_vector = normal(size=n_features)
    random_vector = sqrt(squared_radius) * random_vector / norm(random_vector)

    return random_vector


@vectorize("float64(float64, float64, float64)")
def gaussian(x: float, mean: float, var: float) -> float:
    """
    Compute the value of a Gaussian probability density function at a given point.

    Parameters
    ----------
    x : float
        The point at which to evaluate the Gaussian probability density function.
    mean : float
        The mean of the Gaussian probability density function.
    var : float
        The variance of the Gaussian probability density function.

    Returns
    -------
    float
        The value of the Gaussian probability density function at the given point.
    """
    return exp(-0.5 * pow(x - mean, 2.0) / var) / sqrt(2 * pi * var)


@njit
def multivariate_gaussian(x: ndarray, mean: ndarray, cov: ndarray) -> float:
    """
    Compute the relative Gaussian probability of a given mean vector and covariance matrix.

    Parameters
    ----------
    x : numpy.ndarray
        The input vector for which to compute the probability.
    mean : numpy.ndarray
        The mean vector of the Gaussian distribution.
    cov : numpy.ndarray
        The covariance matrix of the Gaussian distribution.

    Returns
    -------
    float
        The relative Gaussian probability of the input vector.
    """
    exponent = -0.5 * dot(x - mean, dot(inv(cov), x - mean))
    normalization = 1.0 / sqrt((2 * pi) ** len(x) * det(cov))
    return exp(exponent) * normalization


@vectorize("float64(float64, float64, float64)")
def damped_update(new, old, damping):
    """
    Damped update of old value with new value.
    the opertation that is performed is:
    damping * new + (1 - damping) * old
    """
    return damping * new + (1 - damping) * old


@njit(error_model="numpy", fastmath=False)
def max_difference(x, y):
    max = -inf
    for i in range(len(x)):
        diff = abs(x[i] - y[i])
        if diff > max:
            max = diff
    return max


# --------------------------- errors classification -------------------------- #


def classification_adversarial_error(m, q, P, eps, pstar):
    Iminus = quad(
        lambda x: np.exp(-0.5 * x**2 / q) * erfc(m * x / np.sqrt(2 * q * (q - m**2))),
        -eps * P ** (1 / pstar),
        np.inf,
    )[0]
    Iplus = quad(
        lambda x: np.exp(-0.5 * x**2 / q) * (1 + erf(m * x / np.sqrt(2 * q * (q - m**2)))),
        -np.inf,
        eps * P ** (1 / pstar),
    )[0]
    return 0.5 * (Iminus + Iplus) / np.sqrt(2 * pi * q)


# ----------------------------- errors regression ---------------------------- #


# @njit(error_model="numpy", fastmath=True)
def estimation_error(m, q, sigma, **args):
    return 1 + q - 2.0 * m


# @njit
def angle_teacher_student(m, q, sigma, **args):
    return np.arccos(m / np.sqrt(q)) / pi


def margin_probit_classif(m, q, sigma, delta):
    return (4 * m * sqrt(2 * pi**3)) / sqrt(delta + 1)


# errors
def gen_error(m, q, sigma, delta_in, delta_out, percentage, beta):
    return q - 2 * m * (1 + (-1 + beta) * percentage) + 1 + percentage * (-1 + beta**2)


def excess_gen_error(m, q, sigma, delta_in, delta_out, percentage, beta):
    gen_err_BO_alpha_inf = (1 - percentage) * percentage**2 * (1 - beta) ** 2 + percentage * (
        1 - percentage
    ) ** 2 * (beta - 1) ** 2
    return gen_error(m, q, sigma, delta_in, delta_out, percentage, beta) - gen_err_BO_alpha_inf


def excess_gen_error_oracle_rescaling(m, q, sigma, delta_in, delta_out, percentage, beta):
    oracle_norm = 1 - percentage + percentage * beta
    m_prime = oracle_norm * m / sqrt(q)
    q_prime = oracle_norm**2

    return excess_gen_error(m_prime, q_prime, sigma, delta_in, delta_out, percentage, beta)


def estimation_error_rescaled(m, q, sigma, delta_in, delta_out, percentage, beta, norm_const):
    m = m / norm_const
    q = q / (norm_const**2)

    return estimation_error(m, q, sigma)


def estimation_error_oracle_rescaling(m, q, sigma, delta_in, delta_out, percentage, beta):
    oracle_norm = 1.0  # abs(1 - percentage + percentage * beta)
    m_prime = oracle_norm * m / sqrt(q)
    q_prime = oracle_norm**2

    return estimation_error(m_prime, q_prime, sigma)


def gen_error_BO(m, q, sigma, delta_in, delta_out, percentage, beta):
    return (1 + percentage * (-1 + beta**2) - (1 + percentage * (-1 + beta)) ** 2 * q) - (
        (1 - percentage) * percentage**2 * (1 - beta) ** 2
        + percentage * (1 - percentage) ** 2 * (beta - 1) ** 2
    )


def gen_error_BO_old(m, q, sigma, delta_in, delta_out, percentage, beta):
    q = (1 - percentage + percentage * beta) ** 2 * q
    m = (1 - percentage + percentage * beta) * m

    return estimation_error(m, q, sigma, tuple())


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


# overlaps
def m_overlap(m, q, sigma, **args):
    return m


def q_overlap(m, q, sigma, **args):
    return q


def sigma_overlap(m, q, sigma, **args):
    return sigma
