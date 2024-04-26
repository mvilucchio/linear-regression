from numpy import (
    divide,
    ndarray,
    identity,
    sqrt,
    abs,
    count_nonzero,
    sum,
    dot,
    ones_like,
    inf,
    tile,
    finfo,
    float64,
    empty,
    sum,
)
import numpy as np
import numpy.linalg as LA
from numpy.random import normal
from numba import njit
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.utils import axis0_safe_slice
from sklearn.utils.extmath import safe_sparse_dot
from scipy.optimize import minimize, line_search
from cvxpy import Variable, Minimize, Problem, norm, sum_squares
from ..utils.matrix_utils import axis0_pos_neg_mask, safe_sparse_dot
from ..erm import GTOL_MINIMIZE, MAX_ITER_MINIMIZE
import jax.numpy as jnp
import jax
from jax.scipy.optimize import minimize as jax_minimize


@njit(error_model="numpy", fastmath=True)
def _loss_and_gradient_L2(w, xs_norm, ys, reg_param):
    linear_loss = ys - xs_norm @ w
    loss = 0.5 * dot(linear_loss, linear_loss) + 0.5 * reg_param * dot(w, w)
    gradient = -xs_norm.T @ linear_loss + reg_param * w

    return loss, gradient


@njit(error_model="numpy", fastmath=True)
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


# @njit(error_model="numpy")
# def _loss_and_gradient_Huber(w, xs_norm, ys, reg_param, a):
#     linear_loss = ys - xs_norm @ w
#     abs_linear_loss = np.abs(linear_loss)
#     outliers_mask = abs_linear_loss > a

#     outliers = abs_linear_loss[outliers_mask]
#     num_outliers = count_nonzero(outliers_mask)
#     n_non_outliers = xs_norm.shape[0] - num_outliers

#     loss = a * sum(outliers) - 0.5 * num_outliers * a**2

#     non_outliers = linear_loss[~outliers_mask]
#     loss += 0.5 * dot(non_outliers, non_outliers)
#     loss += 0.5 * reg_param * dot(w, w)

#     (xs_outliers, xs_non_outliers) = axis0_pos_neg_mask(xs_norm, outliers_mask, num_outliers)
#     xs_non_outliers *= -1.0

#     gradient = safe_sparse_dot(non_outliers, xs_non_outliers)

#     signed_outliers = ones_like(outliers)
#     signed_outliers_mask = linear_loss[outliers_mask] < 0
#     signed_outliers[signed_outliers_mask] = -1.0

#     gradient -= a * safe_sparse_dot(signed_outliers, xs_outliers)
#     gradient += reg_param * w

#     return loss, gradient


def _loss_and_gradient_Huber(w, xs_norm, ys, reg_param, a):
    linear_loss = ys - xs_norm @ w
    abs_linear_loss = np.abs(linear_loss)
    outliers_mask = abs_linear_loss > a

    outliers = abs_linear_loss[outliers_mask]
    num_outliers = np.count_nonzero(outliers_mask)
    n_non_outliers = xs_norm.shape[0] - num_outliers

    loss = a * sum(outliers) - 0.5 * num_outliers * a**2

    non_outliers = linear_loss[~outliers_mask]
    loss += 0.5 * np.dot(non_outliers, non_outliers)
    loss += 0.5 * reg_param * np.dot(w, w)

    xs_non_outliers = -axis0_safe_slice(xs_norm, ~outliers_mask, n_non_outliers)
    gradient = safe_sparse_dot(non_outliers, xs_non_outliers)

    signed_outliers = np.ones_like(outliers)
    signed_outliers_mask = linear_loss[outliers_mask] < 0
    signed_outliers[signed_outliers_mask] = -1.0

    xs_outliers = axis0_safe_slice(xs_norm, outliers_mask, num_outliers)

    gradient -= a * safe_sparse_dot(signed_outliers, xs_outliers)
    gradient += reg_param * w

    return loss, gradient


# the reason why we don't use the one of sklearn is because it has strange bounds on a
def find_coefficients_Huber(ys, xs, reg_param, a, scale_init=1.0):
    _, d = xs.shape
    w = normal(loc=0.0, scale=scale_init, size=(d,))
    xs_norm = divide(xs, sqrt(d))

    bounds = tile([-inf, inf], (w.shape[0], 1))
    bounds[-1][0] = finfo(float64).eps * 10

    opt_res = minimize(
        _loss_and_gradient_Huber,
        w,
        method="L-BFGS-B",
        jac=True,
        args=(xs_norm, ys, reg_param, a),
        options={"maxiter": MAX_ITER_MINIMIZE, "gtol": GTOL_MINIMIZE, "iprint": -1},
        bounds=bounds,
    )

    if opt_res.status == 2:
        raise ValueError(
            "HuberRegressor convergence failed: l-BFGS-b solver terminated with %s"
            % opt_res.message
        )

    return opt_res.x


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


@jax.jit
def _loss_Logistic_adv_Linf(w, xs_norm, ys, reg_param, eps_t, reg_order):
    n, d = xs_norm.shape
    loss = jnp.sum(
        jnp.log(
            1 + jnp.exp(-ys * jnp.dot(xs_norm, w) + eps_t * jnp.sum(jnp.abs(w)) / d)
        )
    )
    + reg_param * jnp.sum(jnp.abs(w) ** reg_order)
    return loss


def find_coefficients_Logistic_adv_Linf(ys, xs, reg_param, eps_t, reg_order):
    _, d = xs.shape
    w = normal(loc=0.0, scale=1.0, size=(d,))
    xs_norm = divide(xs, sqrt(d))

    opt_res = jax_minimize(
        _loss_Logistic_adv_Linf,
        w,
        method="BFGS",
        args=(xs_norm, ys, reg_param, eps_t, reg_order),
        options={"maxiter": MAX_ITER_MINIMIZE},
    )

    if opt_res.status == 2:
        raise ValueError(
            "LogisticRegressor convergence failed: l-BFGS solver terminated with %s"
            % opt_res.message
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
            raise ValueError(
                "ground_truth_theta must be provided when save_run is True."
            )
        losses = empty(max_iters + 1)
        qs = empty(max_iters + 1)
        estimation_error = empty(max_iters + 1)
        losses[0], _ = loss_grad_function(w, xs_norm, ys, reg_param, *loss_grad_args)
        losses[0] /= n
        estimation_error[0] = sum((ground_truth_theta - w) ** 2) / d
        qs[0] = sum(w**2) / d

    for t in range(1, max_iters + 1):
        # if t % 50 == 0:
        #     print("Iteration ", t)
        loss, gradient = loss_grad_function(w, xs_norm, ys, reg_param, *loss_grad_args)
        w -= lr * gradient
        if save_run:
            losses[t] = loss / n
            estimation_error[t] = sum((ground_truth_theta - w) ** 2) / d
            qs[t] = sum(w**2) / d

    if save_run:
        return w, losses, qs, estimation_error
    return w


# ---------------


# not checked
def find_coefficients_Huber_on_sphere(ys, xs, reg_param, q_fixed, a, gamma=1e-04):
    _, d = xs.shape
    w = normal(loc=0.0, scale=1.0, size=(d,))
    w = w / sqrt(LA.norm(w)) * sqrt(q_fixed)
    xs_norm = divide(xs, sqrt(d))

    loss, grad = _loss_and_gradient_Huber(w, xs_norm, ys, reg_param, a)
    iter = 0
    while iter < MAX_ITER_MINIMIZE and LA.norm(grad) > GTOL_MINIMIZE:
        if iter % 10 == 0:
            print(
                str(iter)
                + "th Iteration  Loss :: "
                + str(loss)
                + " gradient :: "
                + str(LA.norm(grad))
            )

        alpha = 1
        new_w = w - alpha * grad
        new_loss, new_grad = _loss_and_gradient_Huber(w, xs_norm, ys, reg_param, a)

        # backtracking line search
        while new_loss > loss - gamma * alpha * LA.norm(grad):
            alpha = alpha / 2
            new_w = w - alpha * grad
            new_loss, new_grad = _loss_and_gradient_Huber(w, xs_norm, ys, reg_param, a)

        loss = new_loss
        grad = new_grad
        w = new_w

        iter += 1

    return w
