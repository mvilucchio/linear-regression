from numpy import (
    divide,
    ndarray,
    sqrt,
    identity,
    sum,
    dot,
    inf,
    tile,
    finfo,
    float64,
    float32,
    empty,
    sum,
)
import numpy as np
import numpy.linalg as LA
from numpy.random import normal
from numba import njit
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize, dual_annealing, SR1
from cvxpy import (
    Variable,
    Minimize,
    Maximize,
    Problem,
    norm,
    sum_squares,
    logistic,
    multiply,
    norm1,
)
from cvxpy import sum as cp_sum
import jax.numpy as jnp
import jax
from jax import grad, vmap, hessian, jit
from numpy.random import default_rng
from ..erm import GTOL_MINIMIZE, MAX_ITER_MINIMIZE, XTOL_MINIMIZE

rng = default_rng()


# ---------------------------------------------------------------------------- #
#                                  Regression                                  #
# ---------------------------------------------------------------------------- #
@njit(error_model="numpy", fastmath=True)
def _loss_and_gradient_L2(w, xs_norm, ys, reg_param):
    linear_loss = ys - xs_norm @ w
    loss = 0.5 * dot(linear_loss, linear_loss) + 0.5 * reg_param * dot(w, w)
    gradient = -xs_norm.T @ linear_loss + reg_param * w

    return loss, gradient


# @njit(error_model="numpy", fastmath=True)
def find_coefficients_L2(ys, xs, reg_param):
    _, d = xs.shape
    a = divide(xs.T.dot(xs), d) + reg_param * identity(d)
    b = divide(xs.T.dot(ys), sqrt(d))
    return LA.solve(a, b)


def find_coefficients_L1(ys, xs, reg_param):
    _, d = xs.shape
    # w = np.random.normal(loc=0.0, scale=1.0, size=(d,))
    xs_norm = divide(xs, sqrt(d))
    w = Variable(shape=d)
    obj = Minimize(norm(ys - xs_norm @ w, 1) + 0.5 * reg_param * sum_squares(w))
    prob = Problem(obj)
    prob.solve(eps_abs=1e-3)
    return w.value


@jit
def _loss_Huber(w, xs_norm, ys, reg_param, a, d):
    target = xs_norm @ w
    abs_diff = jnp.abs(target - ys)
    return (
        jnp.sum(jnp.where(abs_diff > a, a * (abs_diff - 0.5 * a), 0.5 * abs_diff**2))
        + 0.5 * reg_param * jnp.dot(w, w)
    ) / d


_grad_loss_Huber = jit(grad(_loss_Huber))
_hess_loss_Huber = jit(hessian(_loss_Huber))


# the reason why we don't use the one of sklearn is because it has strange bounds on a
def find_coefficients_Huber(ys, xs, reg_param, a, inital_w=None, scale_init=1.0):
    _, d = xs.shape
    if inital_w is None:
        w = normal(loc=0.0, scale=scale_init, size=(d,))
    else:
        w = inital_w.copy()
    xs_norm = divide(xs, sqrt(d))

    bounds = tile([-inf, inf], (w.shape[0], 1))
    bounds[-1][0] = finfo(float64).eps * 10

    opt_res = minimize(
        _loss_Huber,
        w,
        method="trust-constr",
        jac=_grad_loss_Huber,
        hess=_hess_loss_Huber,
        args=(xs_norm, ys, reg_param, a, d),
        options={"maxiter": MAX_ITER_MINIMIZE, "gtol": GTOL_MINIMIZE, "xtol": XTOL_MINIMIZE},
        bounds=bounds,
    )

    if opt_res.status == 2:
        raise ValueError(
            f"HuberRegressor convergence failed: trust-constr solver terminated with {opt_res.message}"
        )

    return opt_res.x


# import numpy as np
# import numpy.linalg as LA
# from numpy.random import normal
# from numba import njit
# from sklearn.svm import LinearSVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.utils import axis0_safe_slice
# from sklearn.utils.extmath import safe_sparse_dot
# from scipy.optimize import minimize, line_search
# from cvxpy import Variable, Minimize, Problem, norm, sum_squares
# from ..utils.matrix_utils import axis0_pos_neg_mask, safe_sparse_dot
# from ..erm import GTOL_MINIMIZE, MAX_ITER_MINIMIZE


# def _loss_and_gradient_Huber(w, xs_norm, ys, reg_param, a):
#     linear_loss = ys - xs_norm @ w
#     abs_linear_loss = np.abs(linear_loss)
#     outliers_mask = abs_linear_loss > a

#     outliers = abs_linear_loss[outliers_mask]
#     num_outliers = np.count_nonzero(outliers_mask)
#     n_non_outliers = xs_norm.shape[0] - num_outliers

#     loss = a * sum(outliers) - 0.5 * num_outliers * a**2

#     non_outliers = linear_loss[~outliers_mask]
#     loss += 0.5 * np.dot(non_outliers, non_outliers)
#     loss += 0.5 * reg_param * np.dot(w, w)

#     xs_non_outliers = -axis0_safe_slice(xs_norm, ~outliers_mask, n_non_outliers)
#     gradient = safe_sparse_dot(non_outliers, xs_non_outliers)

#     signed_outliers = np.ones_like(outliers)
#     signed_outliers_mask = linear_loss[outliers_mask] < 0
#     signed_outliers[signed_outliers_mask] = -1.0

#     xs_outliers = axis0_safe_slice(xs_norm, outliers_mask, num_outliers)

#     gradient -= a * safe_sparse_dot(signed_outliers, xs_outliers)
#     gradient += reg_param * w

#     return loss, gradient


# # the reason why we don't use the one of sklearn is because it has strange bounds on a
# def find_coefficients_Huber(ys, xs, reg_param, a, scale_init=1.0):
#     _, d = xs.shape
#     w = normal(loc=0.0, scale=scale_init, size=(d,))
#     xs_norm = divide(xs, sqrt(d))

#     bounds = tile([-inf, inf], (w.shape[0], 1))
#     bounds[-1][0] = finfo(float64).eps * 10

#     opt_res = minimize(
#         _loss_and_gradient_Huber,
#         w,
#         method="L-BFGS-B",
#         jac=True,
#         args=(xs_norm, ys, reg_param, a),
#         options={"maxiter": MAX_ITER_MINIMIZE, "gtol": GTOL_MINIMIZE, "iprint": -1},
#         bounds=bounds,
#     )

#     if opt_res.status == 2:
#         raise ValueError(
#             f"HuberRegressor convergence failed: l-BFGS-b solver terminated with {opt_res.message}"
#         )

#     return opt_res.x


@jit
def _loss_mod_Tukey(w, xs_norm, ys, reg_param, tau, c, d, p):
    target = xs_norm @ w
    diff = ys - target

    middle_mask = jnp.abs(diff) <= tau
    upper_mask = diff > tau
    lower_mask = diff < -tau

    tau_squared_div_6 = tau**2 / 6.0

    sign_factor = (-1) ** (p % 2)

    middle_term = tau_squared_div_6 * (1 - (1 - (diff**2) / (tau**2)) ** 3)
    upper_term = tau_squared_div_6 + c * (diff - tau) ** p
    lower_term = tau_squared_div_6 + sign_factor * c * (tau + diff) ** p

    return (
        jnp.sum(middle_mask * middle_term + upper_mask * upper_term + lower_mask * lower_term)
        + 0.5 * reg_param * jnp.dot(w, w)
    ) / d


_grad_loss_Tukey = jit(grad(_loss_mod_Tukey))


def find_coefficients_mod_Tukey(ys, xs, reg_param, tau, c, p=3, initial_w=None, scale_init=1.0):
    _, d = xs.shape
    if initial_w is None:
        w = normal(loc=0.0, scale=scale_init, size=(d,))
    else:
        w = initial_w.copy()

    if not isinstance(p, int):
        raise ValueError("p must be an integer.")
    if p < 2:
        raise ValueError("p must be greater than 1.")

    xs_norm = divide(xs, sqrt(d))

    sr1 = SR1()

    opt_res = minimize(
        _loss_mod_Tukey,
        w,
        args=(xs_norm, ys, reg_param, tau, c, d, p),
        method="trust-constr",
        jac=_grad_loss_Tukey,
        hessp=sr1,
        options={
            "maxiter": MAX_ITER_MINIMIZE,
            "verbose": 0,
            "xtol": XTOL_MINIMIZE,
            "gtol": GTOL_MINIMIZE,
        },
    )

    if opt_res.status == 2:
        print(
            f"TuKeyRegression convergence failed: trust-constr solver terminated with {opt_res.message}"
        )

    return opt_res.x


@jit
def _loss_Cauchy(w, xs_norm, ys, reg_param, tau, d):
    target = xs_norm @ w
    diff = ys - target
    return (
        0.5 * tau**2 * jnp.sum(tau * jnp.log1p((diff / tau) ** 2)) + 0.5 * reg_param * jnp.dot(w, w)
    ) / d


_grad_loss_Cauchy = jit(grad(_loss_Cauchy))


def find_coefficients_Cauchy(ys, xs, reg_param, tau, initial_w=None, scale_init=1.0):
    _, d = xs.shape
    if initial_w is None:
        w = normal(loc=0.0, scale=scale_init, size=(d,))
    else:
        w = initial_w.copy()

    xs_norm = divide(xs, sqrt(d))

    sr1 = SR1()

    opt_res = minimize(
        _loss_Cauchy,
        w,
        args=(xs_norm, ys, reg_param, tau, d),
        method="trust-constr",
        jac=_grad_loss_Cauchy,
        hessp=sr1,
        options={
            "maxiter": MAX_ITER_MINIMIZE,
            "verbose": 0,
            "xtol": XTOL_MINIMIZE,
            "gtol": GTOL_MINIMIZE,
        },
    )

    if opt_res.status == 2:
        print(
            f"CauchyRegression convergence failed: trust-constr solver terminated with {opt_res.message}"
        )

    return opt_res.x


def find_coefficients_vanilla_GD(
    ys: ndarray,
    xs: ndarray,
    reg_param: float,
    loss_grad_function: callable,
    loss_grad_args: tuple,
    lr: float = 1e-3,
    w_init: ndarray = None,
    max_iters: int = 1000,
    save_run: bool = False,
    ground_truth_theta: ndarray = None,
):
    if lr <= 0:
        raise ValueError("lr must be positive.")

    if max_iters <= 0 or not isinstance(max_iters, int):
        raise ValueError("max_iters must be a positive integer.")

    n, d = xs.shape
    xs_norm = xs / np.sqrt(d)

    if w_init is None:
        w = normal(loc=0.0, scale=1.0, size=(d,))
    else:
        w = w_init.copy()

    if save_run:
        if ground_truth_theta is None:
            raise ValueError("ground_truth_theta must be provided when save_run is True.")
        losses = empty(max_iters + 1)
        qs = empty(max_iters + 1)
        estimation_error = empty(max_iters + 1)
        losses[0], _ = loss_grad_function(w, xs_norm, ys, reg_param, *loss_grad_args)
        losses[0] /= n
        estimation_error[0] = sum((ground_truth_theta - w) ** 2) / d
        qs[0] = sum(w**2) / d

    for t in range(1, max_iters + 1):
        loss, gradient = loss_grad_function(w, xs_norm, ys, reg_param, *loss_grad_args)
        w -= lr * gradient
        if save_run:
            losses[t] = loss / n
            estimation_error[t] = sum((ground_truth_theta - w) ** 2) / d
            qs[t] = sum(w**2) / d

    if save_run:
        return w, losses, qs, estimation_error
    return w


# -----------------------------------
@jit
def linear_classif_loss(x, y, w):
    prediction = jnp.dot(x, w)
    return -y * prediction


vec_linear_classif_loss = jit(vmap(linear_classif_loss, in_axes=(0, 0, None)))

grad_linear_classif_loss = jit(grad(linear_classif_loss, argnums=0))

vec_grad_linear_classif_loss = jit(vmap(grad_linear_classif_loss, in_axes=(0, 0, None)))


# ---------------------------------------------------------------------------- #
#                                Classification                                #
# ---------------------------------------------------------------------------- #
def find_coefficients_Hinge(ys, xs, reg_param):
    _, d = xs.shape
    clf = LinearSVC(
        penalty="l2",
        loss="hinge",
        fit_intercept=False,
        C=1 / reg_param,
        max_iter=MAX_ITER_MINIMIZE,
        tol=GTOL_MINIMIZE,
        dual=True,
    )
    clf.fit(xs / sqrt(d), ys)
    return clf.coef_[0]


def find_coefficients_Logistic(ys, xs, reg_param):
    _, d = xs.shape
    clf = LogisticRegression(
        solver="lbfgs",
        C=(1.0 / reg_param),
        dual=False,
        fit_intercept=False,
        max_iter=MAX_ITER_MINIMIZE,
        tol=GTOL_MINIMIZE,
    )
    clf.fit(xs / sqrt(d), ys)
    return clf.coef_[0]


# --------------------------- adversarial training --------------------------- #
@jit
def _loss_Logistic_adv(
    w, xs_norm, ys, reg_param: float, ε: float, reg_order: float, pstar: float, d: int
):
    loss = jnp.sum(
        jnp.log1p(
            jnp.exp(
                -ys * jnp.dot(xs_norm, w)
                + ε * jnp.sum(jnp.abs(w) ** pstar) ** (1 / pstar) / d**pstar
            )
        )
    ) + reg_param * jnp.sum(jnp.abs(w) ** reg_order)
    return loss


_grad_loss_Logistic_adv = jit(grad(_loss_Logistic_adv))
_hess_loss_Logistic_adv = jit(hessian(_loss_Logistic_adv))


def find_coefficients_Logistic_adv(
    ys, xs, reg_param: float, ε: float, reg_order: float, pstar: float, initial_w: ndarray = None
):
    _, d = xs.shape

    if initial_w is None:
        initial_w = rng.standard_normal((d,), float32)

    w = initial_w.copy() + 0.01 * rng.standard_normal((d,), float32)

    xs_norm = divide(xs, sqrt(d))

    opt_res = minimize(
        _loss_Logistic_adv,
        w,
        method="Newton-CG",
        jac=_grad_loss_Logistic_adv,
        hess=_hess_loss_Logistic_adv,
        args=(xs_norm, ys, reg_param, ε, reg_order, pstar, d),
        options={"maxiter": MAX_ITER_MINIMIZE, "xtol": 1e-4},
    )

    if opt_res.status == 2:
        raise ValueError(
            f"LogisticRegressor convergence failed: Newton-CG solver terminated with {opt_res.message}"
        )

    return opt_res.x


# ------------------------------ Vdeltacase ------------------------------ #
@jit
def _loss_Logistic_adv_Sigmadelta(
    w, xs_norm, ys, reg_param: float, ε: float, Sigmadelta, Sigmaw, d: int
):
    loss = jnp.sum(
        jnp.log1p(
            jnp.exp(
                -ys * jnp.dot(xs_norm, w) + ε * jnp.sqrt(jnp.dot(w, Sigmadelta @ w)) / jnp.sqrt(d)
            )
        )
    ) + 0.5 * reg_param * jnp.dot(w, Sigmaw @ w)
    return loss


_grad_loss_Logistic_adv_Sigmadelta = jit(grad(_loss_Logistic_adv_Sigmadelta))
_hess_loss_Logistic_adv_Sigmadelta = jit(hessian(_loss_Logistic_adv_Sigmadelta))


def find_coefficients_Logistic_adv_Sigmadelta(
    ys, xs, reg_param: float, ε: float, w_star: ndarray, Sigmadelta, Sigmaw
):
    _, d = xs.shape
    w = w_star.copy() + 0.01 * rng.standard_normal((d,), float32)
    xs_norm = divide(xs, sqrt(d))

    opt_res = minimize(
        _loss_Logistic_adv_Sigmadelta,
        w,
        method="Newton-CG",
        jac=_grad_loss_Logistic_adv_Sigmadelta,
        hess=_hess_loss_Logistic_adv_Sigmadelta,
        args=(xs_norm, ys, reg_param, ε, Sigmadelta, Sigmaw, d),
        options={"maxiter": MAX_ITER_MINIMIZE},
    )

    if opt_res.status == 2:
        raise ValueError(
            f"LogisticRegressor Sigmadelta convergence failed: Newton-CG solver terminated with {opt_res.message}"
        )

    return opt_res.x


# ------------------------------ approximate L1 ------------------------------ #
@jit
def approximalte_L1_regularsation(x, a):
    return (jnp.log(1 + jnp.exp(a * x)) + jnp.log(1 + jnp.exp(-a * x))) / a


v_approximalte_L1_regularsation = jit(vmap(approximalte_L1_regularsation, in_axes=(0, None)))


@jit
def _loss_Logistic_adv_approx_L1(w, xs_norm, ys, reg_param: float, ε: float, pstar: float, d: int):
    loss = (
        jnp.sum(
            jnp.log1p(
                jnp.exp(
                    -ys * jnp.dot(xs_norm, w)
                    + ε * jnp.sum(jnp.abs(w) ** pstar) ** (1 / pstar) / d**pstar
                )
            )
        )
        + reg_param * v_approximalte_L1_regularsation(w, 32.0).sum()
    )
    return loss


_grad_loss_Logistic_approx_L1 = jit(grad(_loss_Logistic_adv_approx_L1))
_hess_loss_Logistic_approx_L1 = jit(hessian(_loss_Logistic_adv_approx_L1))


def find_coefficients_Logistic_approx_L1(
    ys, xs, reg_param: float, ε: float, pstar: float, wstar: ndarray
):
    _, d = xs.shape
    # w = rng.standard_normal((d,), float32)  # normal(loc=0.0, scale=1.0, size=(d,))
    w = wstar.copy() + rng.standard_normal((d,), float32)
    xs_norm = divide(xs, sqrt(d))

    opt_res = minimize(
        _loss_Logistic_adv_approx_L1,
        w,
        method="Newton-CG",
        jac=_grad_loss_Logistic_approx_L1,
        hess=_hess_loss_Logistic_approx_L1,
        args=(xs_norm, ys, reg_param, ε, pstar, d),
        options={"maxiter": MAX_ITER_MINIMIZE, "xtol": 1e-6},
    )

    if opt_res.status == 2:
        raise ValueError(
            f"LogisticRegressor convergence failed: Newton-CG solver terminated with {opt_res.message}"
        )

    return opt_res.x


# for this it is better to consider the data as parameters and avoid recompilation at each iteration
def find_coefficients_Logistic_adv_Linf_L1(ys: ndarray, xs: ndarray, reg_param: float, ε: float):
    _, d = xs.shape
    xs_norm = divide(xs, sqrt(d))
    w = Variable(d)
    loss_term = cp_sum(logistic(-multiply(ys, xs_norm @ w) + ε * norm1(w) / d))

    l1_reg = norm1(w)

    objective = Minimize(loss_term + reg_param * l1_reg)
    problem = Problem(objective)
    problem.solve(verbose=True)

    return w.value


# for this it is better to consider the data as parameters and avoid recompilation at each iteration
def find_coefficients_Logistic_adv_Linf_L2(ys: ndarray, xs: ndarray, reg_param: float, ε: float):
    _, d = xs.shape
    xs_norm = divide(xs, sqrt(d))
    w = Variable(d)
    loss_term = cp_sum(logistic(-multiply(ys, xs_norm @ w) + ε * norm1(w) / d))

    l2_reg = norm(w) ** 2

    objective = Minimize(loss_term + reg_param * l2_reg)
    problem = Problem(objective)
    # solver_opts = {"abstol": 1e-10, "reltol": 1e-10, "feastol": 1e-10, "max_iters": 1000}
    # problem.solve(solver="ECOS", verbose=False, **solver_opts)
    problem.solve(verbose=False)

    return w.value
