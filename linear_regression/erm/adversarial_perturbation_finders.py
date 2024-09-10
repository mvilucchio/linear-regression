from numpy import ndarray, tile
from cvxpy import Variable, Problem, Maximize, norm
from jax import jit, grad, vmap
from jax.lax import cond
import jax.numpy as jnp
from ..erm import (
    MAX_ITER_PDG,
    STEP_BLOCK_PDG,
    STEP_SIZE_PDG,
    TOL_PDG,
    TEST_ITERS_PDG,
)


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

    return -ys[:, None] * tile(x.value, (len(ys), 1))


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

    return -ys[:, None] * tile(delta.value, (len(ys), 1))


def find_adversarial_perturbation_NLRF(
    ys: ndarray,
    cs: ndarray,
    w: ndarray,
    F: ndarray,
    wstar: ndarray,
    ε: float,
    p: float,
    step_size: float = STEP_SIZE_PDG,
    abs_tol: float = TOL_PDG,
    step_block: int = STEP_BLOCK_PDG,
    max_iterations: int = MAX_ITER_PDG,
    adv_pert: ndarray = None,
    test_iters: int = TEST_ITERS_PDG,
    return_losses: bool = False,
):
    if adv_pert is None:
        print("adv_pert is None")
        adv_pert = jnp.zeros_like(cs)

    assert cs.shape == adv_pert.shape

    _, d = cs.shape

    if return_losses:
        losses = []

    for i in range(max_iterations):
        if return_losses:
            losses.append([])

        for _ in range(step_block):
            if return_losses:
                losses[i].append(vec_NLRF_data_perturb_loss(adv_pert, cs, ys, w, F, d))

            adv_pert = PGA_step_NLRF(adv_pert, cs, ys, w, wstar, F, d, step_size, ε, p)

        all_current_loss_val = vec_NLRF_data_perturb_loss(adv_pert, cs, ys, w, F, d)

        tmp_adv_pert = adv_pert.copy()

        for _ in range(test_iters):
            tmp_adv_pert = PGA_step_NLRF(tmp_adv_pert, cs, ys, w, wstar, F, d, step_size, ε, p)

        all_temp_loss = vec_NLRF_data_perturb_loss(tmp_adv_pert, cs, ys, w, F, d)

        if jnp.max(jnp.abs(all_current_loss_val - all_temp_loss)) <= abs_tol:
            break

    if return_losses:
        return adv_pert, losses
    return adv_pert


# ---------------------------------------------------------------------------- #
#                          Projected Gradient Descent                          #
# ---------------------------------------------------------------------------- #

# ------------------------ non-linear random features ------------------------ #


@jit
def linear_NLRF_data_perturb_loss(δ, c, y, w, F, d: int):
    return -y * jnp.dot(jnp.tanh((c + δ) @ F / jnp.sqrt(d)), w)


# @jit
# def linear_NLRF_data_perturb_loss(δ, c, y, w, F, d: int, activ_fun):
#     return -y * jnp.dot(activ_fun((c + δ) @ F / jnp.sqrt(d)), w)


@jit
def vec_NLRF_data_perturb_loss(δs, cs, ys, w, F, d: int):
    return -ys * jnp.dot(jnp.tanh(((cs + δs) @ F) / jnp.sqrt(d)), w)


grad_NLRF_data_perturb = jit(grad(linear_NLRF_data_perturb_loss, argnums=0))

vec_grad_NLRF_data_perturb = jit(vmap(grad_NLRF_data_perturb, in_axes=(0, 0, 0, None, None, None)))


@jit
def then_func(ops):
    x, ε, norm_x_projected = ops
    return ε * x / norm_x_projected


@jit
def else_func(ops):
    return ops[0]


def project_and_normalize(z, wstar, ε, p):
    z -= jnp.dot(z, wstar) * wstar / jnp.dot(wstar, wstar)
    norm_x_projected = jnp.sum(jnp.abs(z) ** p) ** (1 / p)

    return cond(norm_x_projected > ε, then_func, else_func, (z, ε, norm_x_projected))


vec_project_and_normalize = jit(vmap(project_and_normalize, in_axes=(0, None, None, None)))


# @jit
def PGA_step_NLRF(δs, xs, ys, w, wstar, F, d: int, step_size: float, ε: float, p):
    g = vec_grad_NLRF_data_perturb(δs, xs, ys, w, F, d)
    return vec_project_and_normalize(δs + step_size * g, wstar, ε, p)


# -------------------------------- linear loss ------------------------------- #
# @jit
# def total_loss_logistic(w, Xs, ys, reg_param):
#     ys = ys.reshape(-1, 1) if ys.ndim == 1 else ys
#     scores = jnp.matmul(Xs, w)
#     loss_part = jnp.sum(jnp.log(1 + jnp.exp(-ys * scores)))
#     reg_part = 0.5 * reg_param * jnp.dot(w, w)
#     return loss_part + reg_part


# @jit
# def linear_classif_loss(x, y, w):
#     return -y * jnp.dot(x, w)


# vec_linear_classif_loss = jit(vmap(linear_classif_loss, in_axes=(0, 0, None)))
# grad_linear_classif_loss = jit(grad(linear_classif_loss, argnums=0))
# vec_grad_linear_classif_loss = jit(vmap(grad_linear_classif_loss, in_axes=(0, 0, None)))


# # jitted_norm = jit(jnp.linalg.norm, static_argnums=["ord"])


# @jit
# def project_and_normalize(x, wstar, ε, p):
#     x -= jnp.dot(x, wstar) * wstar / jnp.dot(wstar, wstar)
#     norm_x_projected = jnp.sum(jnp.abs(x) ** p) ** (1 / p)
#     # norm_x_projected = jitted_norm(x, ord=p)

#     return jax.lax.cond(norm_x_projected > ε, then_func, else_func, (x, ε, norm_x_projected))


# vec_project_and_normalize = jit(vmap(project_and_normalize, in_axes=(0, None, None, None)))


# @jit
# def PGA_step_NLRF(vs, ys, w, wstar, step_size, ε, p):
#     g = vec_grad_linear_classif_loss(vs, ys, w)
#     return vec_project_and_normalize(vs + step_size * g, wstar, ε, p)


# # THERE IS A PROBLEM HERE IN THE IMPLEMENTATION
# def projected_GA(ys, w, wstar, step_size, n_steps, ε, p, adv_perturbation=None):
#     if adv_perturbation is None:
#         adv_perturbation = jnp.zeros((len(ys), len(w)))

#     for _ in range(n_steps):
#         adv_perturbation = PGA_step_NLRF(adv_perturbation, ys, w, wstar, step_size, ε, p)
#     return adv_perturbation


# def check_if_additional_step_improves(adv_perturbation, ys, w, ε, p):
#     loss_before = linear_classif_loss(adv_perturbation, ys, w)
#     adv_perturbation_after = PGA_step_NLRF(adv_perturbation, ys, w, ε, p)
#     loss_after = linear_classif_loss(adv_perturbation_after, ys, w)
#     return loss_after < loss_before


# def projected_GA_untill_convergence(
#     ys, w, wstar, step_size, ε, p, adv_perturbation=None, step_block=100
# ):
#     if adv_perturbation is None:
#         adv_perturbation = jnp.zeros((len(ys), len(w)))

#     while not check_if_additional_step_improves(adv_perturbation, ys, w, ε, p):
#         for _ in range(step_block):
#             adv_perturbation = PGA_step_NLRF(adv_perturbation, ys, w, wstar, step_size, ε, p)

#     return adv_perturbation
