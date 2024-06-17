from numpy import (
    divide,
    ndarray,
    sqrt,
    identity,
    abs,
    count_nonzero,
    sum,
    dot,
    ones_like,
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
from scipy.optimize import minimize, line_search
from cvxpy import (
    Variable,
    Parameter,
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
from ..erm import GTOL_MINIMIZE, MAX_ITER_MINIMIZE
import jax.numpy as jnp
import jax

from numpy.random import default_rng

# from jax.scipy.optimize import minimize as jax_minimize

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


@jax.jit
def _loss_Huber(w, xs_norm, ys, reg_param, a):
    target = xs_norm @ w
    abs_diff = jnp.abs(target - ys)
    return jnp.where(
        abs_diff > a, a * (abs_diff - 0.5 * a), 0.5 * abs_diff**2
    ) + 0.5 * reg_param * jnp.dot(w, w)


_grad_loss_Huber = jax.jit(jax.grad(_loss_Huber))
_hess_loss_Huber = jax.jit(jax.hessian(_loss_Huber))


# the reason why we don't use the one of sklearn is because it has strange bounds on a
def find_coefficients_Huber(ys, xs, reg_param, a, scale_init=1.0):
    _, d = xs.shape
    w = normal(loc=0.0, scale=scale_init, size=(d,))
    xs_norm = divide(xs, sqrt(d))

    bounds = tile([-inf, inf], (w.shape[0], 1))
    bounds[-1][0] = finfo(float64).eps * 10

    opt_res = minimize(
        _loss_Huber,
        w,
        method="Newton-CG",
        jac=_grad_loss_Huber,
        hess=_hess_loss_Huber,
        args=(xs_norm, ys, reg_param, a),
        options={"maxiter": MAX_ITER_MINIMIZE, "gtol": GTOL_MINIMIZE, "iprint": -1},
        bounds=bounds,
    )

    if opt_res.status == 2:
        raise ValueError(
            f"HuberRegressor convergence failed: l-BFGS-b solver terminated with {opt_res.message}"
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


# # not checked
# def find_coefficients_Huber_on_sphere(ys, xs, reg_param, q_fixed, a, gamma=1e-04):
#     _, d = xs.shape
#     w = normal(loc=0.0, scale=1.0, size=(d,))
#     w = w / sqrt(LA.norm(w)) * sqrt(q_fixed)
#     xs_norm = divide(xs, sqrt(d))

#     loss, grad = _loss_and_gradient_Huber(w, xs_norm, ys, reg_param, a)
#     iter = 0
#     while iter < MAX_ITER_MINIMIZE and LA.norm(grad) > GTOL_MINIMIZE:
#         if iter % 10 == 0:
#             print(
#                 str(iter)
#                 + "th Iteration  Loss :: "
#                 + str(loss)
#                 + " gradient :: "
#                 + str(LA.norm(grad))
#             )

#         alpha = 1
#         new_w = w - alpha * grad
#         new_loss, new_grad = _loss_and_gradient_Huber(w, xs_norm, ys, reg_param, a)

#         # backtracking line search
#         while new_loss > loss - gamma * alpha * LA.norm(grad):
#             alpha = alpha / 2
#             new_w = w - alpha * grad
#             new_loss, new_grad = _loss_and_gradient_Huber(w, xs_norm, ys, reg_param, a)

#         loss = new_loss
#         grad = new_grad
#         w = new_w

#         iter += 1

#     return w


# -----------------------------------
@jax.jit
def linear_classif_loss(x, y, w):
    prediction = jnp.dot(x, w)
    return -y * prediction


vec_linear_classif_loss = jax.jit(jax.vmap(linear_classif_loss, in_axes=(0, 0, None)))

grad_linear_classif_loss = jax.jit(jax.grad(linear_classif_loss, argnums=0))

vec_grad_linear_classif_loss = jax.jit(jax.vmap(grad_linear_classif_loss, in_axes=(0, 0, None)))


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
@jax.jit
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


_grad_loss_Logistic_adv = jax.jit(jax.grad(_loss_Logistic_adv))
_hess_loss_Logistic_adv = jax.jit(jax.hessian(_loss_Logistic_adv))


def find_coefficients_Logistic_adv(
    ys, xs, reg_param: float, ε: float, reg_order: float, pstar: float, wstar: ndarray
):
    _, d = xs.shape
    # w = rng.standard_normal((d,), float32)  # normal(loc=0.0, scale=1.0, size=(d,))
    w = wstar.copy() + rng.standard_normal((d,), float32)
    xs_norm = divide(xs, sqrt(d))

    opt_res = minimize(
        _loss_Logistic_adv,
        w,
        method="Newton-CG",
        jac=_grad_loss_Logistic_adv,
        hess=_hess_loss_Logistic_adv,
        args=(xs_norm, ys, reg_param, ε, reg_order, pstar, d),
        options={"maxiter": MAX_ITER_MINIMIZE, "xtol": 1e-3},
    )

    if opt_res.status == 2:
        raise ValueError(
            f"LogisticRegressor convergence failed: Newton-CG solver terminated with {opt_res.message}"
        )

    return opt_res.x


# ------------------------------ sigmadeltacase ------------------------------ #
@jax.jit
def _loss_Logistic_adv_Sigmadelta(w, xs_norm, ys, reg_param: float, ε: float, Sigmadelta, d: int):
    loss = jnp.sum(
        jnp.log1p(
            jnp.exp(
                -ys * jnp.dot(xs_norm, w) + ε / jnp.sqrt(d) * jnp.sqrt(jnp.dot(w, Sigmadelta @ w))
            )
        )
    ) + 0.5 * reg_param * jnp.sum(w**2)
    return loss


_grad_loss_Logistic_adv_Sigmadelta = jax.jit(jax.grad(_loss_Logistic_adv_Sigmadelta))
_hess_loss_Logistic_adv_Sigmadelta = jax.jit(jax.hessian(_loss_Logistic_adv_Sigmadelta))


def find_coefficients_Logistic_adv_Sigmadelta(
    ys, xs, reg_param: float, ε: float, wstar: ndarray, Sigmadelta
):
    _, d = xs.shape
    w = wstar.copy() + rng.standard_normal((d,), float32)
    xs_norm = divide(xs, sqrt(d))

    opt_res = minimize(
        _loss_Logistic_adv_Sigmadelta,
        w,
        method="Newton-CG",
        jac=_grad_loss_Logistic_adv_Sigmadelta,
        hess=_hess_loss_Logistic_adv_Sigmadelta,
        args=(xs_norm, ys, reg_param, ε, Sigmadelta, d),
        options={"maxiter": MAX_ITER_MINIMIZE},
    )

    if opt_res.status == 2:
        raise ValueError(
            f"LogisticRegressor Sigmadelta convergence failed: Newton-CG solver terminated with {opt_res.message}"
        )

    return opt_res.x


# ------------------------------ approximate L1 ------------------------------ #
@jax.jit
def approximalte_L1_regularsation(x, a):
    return (jnp.log(1 + jnp.exp(a * x)) + jnp.log(1 + jnp.exp(-a * x))) / a


v_approximalte_L1_regularsation = jax.jit(
    jax.vmap(approximalte_L1_regularsation, in_axes=(0, None))
)


@jax.jit
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


_grad_loss_Logistic_approx_L1 = jax.jit(jax.grad(_loss_Logistic_adv_approx_L1))
_hess_loss_Logistic_approx_L1 = jax.jit(jax.hessian(_loss_Logistic_adv_approx_L1))


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
    problem.solve(verbose=False)

    return w.value


# ---------------------- adversarial perturbation finder --------------------- #


def find_adversarial_perturbation_direct_space(
    ys: ndarray,
    xs: ndarray,
    w: ndarray,
    wstar: ndarray,
    ε: float,
    p: float,
):
    _, d = xs.shape
    x = Variable(len(w))

    constraints = [norm(x, p) <= ε, wstar @ x == 0]

    objective = Maximize(w @ x)

    problem = Problem(objective, constraints)
    problem.solve()

    return -ys[:, None] * np.tile(x.value, (len(ys), 1))


def find_adversarial_perturbation_RandomFeatures_space(
    ys: ndarray,
    cs: ndarray,
    w: ndarray,
    F: ndarray,
    wstar: ndarray,
    ε: float,
    p: float,
):
    _, d = cs.shape
    delta = Variable(d)

    constraints = [norm(delta, p) <= ε, wstar.T @ delta == 0]

    wtilde = F @ w
    objective = Maximize(wtilde.T @ delta)

    problem = Problem(objective, constraints)
    problem.solve()

    return -ys[:, None] * np.tile(delta.value, (len(ys), 1))
    # return delta.value


# ---------------------------------------------------------------------------- #
#                          Projected Gradient Descent                          #
# ---------------------------------------------------------------------------- #
@jax.jit
def total_loss_logistic(w, Xs, ys, reg_param):
    ys = ys.reshape(-1, 1) if ys.ndim == 1 else ys
    scores = jnp.matmul(Xs, w)
    loss_part = jnp.sum(jnp.log(1 + jnp.exp(-ys * scores)))
    reg_part = 0.5 * reg_param * jnp.dot(w, w)
    return loss_part + reg_part


@jax.jit
def linear_classif_loss(x, y, w):
    return -y * jnp.dot(x, w)


vec_linear_classif_loss = jax.jit(jax.vmap(linear_classif_loss, in_axes=(0, 0, None)))
grad_linear_classif_loss = jax.jit(jax.grad(linear_classif_loss, argnums=0))
vec_grad_linear_classif_loss = jax.jit(jax.vmap(grad_linear_classif_loss, in_axes=(0, 0, None)))


@jax.jit
def then_func(ops):
    x, ε, norm_x_projected = ops
    return ε * x / norm_x_projected


@jax.jit
def else_func(ops):
    return ops[0]


# jitted_norm = jax.jit(jnp.linalg.norm, static_argnums=["ord"])


@jax.jit
def project_and_normalize(x, wstar, ε, p):
    x -= jnp.dot(x, wstar) * wstar / jnp.dot(wstar, wstar)
    norm_x_projected = jnp.sum(jnp.abs(x) ** p) ** (1 / p)
    # norm_x_projected = jitted_norm(x, ord=p)

    return jax.lax.cond(norm_x_projected > ε, then_func, else_func, (x, ε, norm_x_projected))


vec_project_and_normalize = jax.jit(jax.vmap(project_and_normalize, in_axes=(0, None, None, None)))


@jax.jit
def projected_GA_step_jit(vs, ys, w, wstar, step_size, ε, p):
    g = vec_grad_linear_classif_loss(vs, ys, w)
    return vec_project_and_normalize(vs + step_size * g, wstar, ε, p)


def projected_GA(ys, w, wstar, step_size, n_steps, ε, p, adv_perturbation=None):
    if adv_perturbation is None:
        adv_perturbation = jnp.zeros((len(ys), len(w)))

    for _ in range(n_steps):
        adv_perturbation = projected_GA_step_jit(adv_perturbation, ys, w, wstar, step_size, ε, p)
    return adv_perturbation


def check_if_additional_step_improves(adv_perturbation, ys, w, ε, p):
    loss_before = linear_classif_loss(adv_perturbation, ys, w)
    adv_perturbation_after = projected_GA_step_jit(adv_perturbation, ys, w, ε, p)
    loss_after = linear_classif_loss(adv_perturbation_after, ys, w)
    return loss_after < loss_before


def projected_GA_untill_convergence(
    ys, w, wstar, step_size, ε, p, adv_perturbation=None, step_block=100
):
    if adv_perturbation is None:
        adv_perturbation = jnp.zeros((len(ys), len(w)))

    while not check_if_additional_step_improves(adv_perturbation, ys, w, ε, p):
        for _ in range(step_block):
            adv_perturbation = projected_GA_step_jit(
                adv_perturbation, ys, w, wstar, step_size, ε, p
            )

    return adv_perturbation


# def projected_GA(ys, w, wstar, step_size, n_steps, ε, p):
#     adv_perturbation = jnp.zeros((len(ys), len(w)))

#     @jax.jit
#     def cond_fun(state):
#         i, _ = state
#         return i < n_steps

#     @jax.jit
#     def body_fun(state):
#         i, adv_perturbation = state
#         adv_perturbation = projected_GA_step_jit(adv_perturbation, ys, w, wstar, step_size, ε, p)
#         return i + 1, adv_perturbation

#     initial_state = (0, adv_perturbation)
#     _, adv_perturbation = while_loop(cond_fun, body_fun, initial_state)

#     return adv_perturbation
