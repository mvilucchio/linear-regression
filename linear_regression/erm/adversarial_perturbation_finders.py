from numpy import ndarray, tile
from math import inf, sqrt
from cvxpy import Variable, Problem, Minimize, norm
from typing import Optional
from numba import njit
import numpy as np
from ..erm import (
    MAX_ITER_PDG,
    STEP_BLOCK_PDG,
    STEP_SIZE_PDG,
    TOL_PDG,
    TEST_ITERS_PDG,
    N_ALTERNATING_PROJ,
    NOISE_SCALE,
)

from line_profiler import profile


# ---------------------------------------------------------------------------- #
#                                Main functions                                #
# ---------------------------------------------------------------------------- #

# ---------------------------- Linear Direct Space --------------------------- #


def find_adversarial_perturbation_direct_space(
    ys: ndarray,
    xs: ndarray,
    w: ndarray,
    wstar: ndarray,
    ε: float,
    p: float,
) -> ndarray:
    delta = Variable(len(w))

    if float(p) == inf:
        constraints = [norm(delta, "inf") <= ε, wstar.T @ delta == 0]
    else:
        constraints = [norm(delta, p) <= ε, wstar.T @ delta == 0]

    objective = Minimize(w @ delta)

    problem = Problem(objective, constraints)
    problem.solve()

    return ys[:, None] * tile(delta.value, (len(ys), 1))


# -------------------------- Linear Random Features -------------------------- #


def find_adversarial_perturbation_linear_rf(
    ys: ndarray,
    cs: ndarray,
    w: ndarray,
    F: ndarray,
    wstar: ndarray,
    ε: float,
    p: float,
) -> ndarray:
    _, d = cs.shape
    delta = Variable(d)
    if float(p) == inf:
        constraints = [norm(delta, "inf") <= ε, wstar.T @ delta == 0]
    else:
        constraints = [norm(delta, p) <= ε, wstar.T @ delta == 0]

    wtilde = F @ w
    objective = Minimize(wtilde.T @ delta)

    problem = Problem(objective, constraints)
    # problem.solve()
    # Solve with high accuracy
    solver_opts = {
        "abstol": 1e-10,  # Absolute tolerance
        "reltol": 1e-10,  # Relative tolerance
        "feastol": 1e-10,  # Feasibility tolerance
        "max_iters": 10_000,  # Maximum iterations
    }
    problem.solve(solver="ECOS", verbose=False, **solver_opts)

    return ys[:, None] * tile(delta.value, (len(ys), 1))


# ------------------------ Non-Linear Random Features ------------------------ #


# @profile
def find_adversarial_perturbation_non_linear_rf(
    ys: ndarray,
    cs: ndarray,
    w: ndarray,
    F: ndarray,
    wstar: ndarray,
    ε: float,
    p: float,
    non_linearity: callable,
    D_non_linearity: callable,
    step_size: float = STEP_SIZE_PDG,
    abs_tol: float = TOL_PDG,
    step_block: int = STEP_BLOCK_PDG,
    max_iterations: int = MAX_ITER_PDG,
    adv_pert: Optional[ndarray] = None,
    test_iters: int = TEST_ITERS_PDG,
) -> ndarray:
    if adv_pert is None:
        adv_pert = np.zeros_like(cs).astype(np.float32)
    elif cs.shape != adv_pert.shape:
        raise ValueError("The shape of the adv_pert is not the same as cs.")

    n_features, d = cs.shape
    d = np.array([d]).astype(np.float32)[0]
    n_features = np.array([n_features]).astype(np.float32)[0]

    wstar_norm = wstar / np.linalg.norm(wstar, ord=2)
    F_sqrtd = F / np.sqrt(d)
    w_sqrtp = w / np.sqrt(n_features)

    adv_pert = alternating_projections_vec(adv_pert, wstar_norm, ε, p)

    cur_i = 0
    for idx in range(max_iterations):
        for _ in range(step_block):
            adv_pert = proj_grad_descent_step(
                # adv_pert = stochastic_proj_grad_descent_step(
                adv_pert,
                cs,
                ys,
                w_sqrtp,
                wstar_norm,
                F_sqrtd,
                step_size,
                ε,
                p,
                np.float32(cur_i),
                D_non_linearity,
            )
            cur_i += 1
            # print(loss_vec_linear_data_perturb(adv_pert, cs, ys, w_sqrtp, F_sqrtd))

        # Record loss after block update
        current_loss = loss_vec_linear_data_perturb(
            adv_pert, cs, ys, w_sqrtp, F_sqrtd, non_linearity
        )

        # Check convergence
        for _ in range(test_iters):
            adv_pert = proj_grad_descent_step(
                adv_pert,
                cs,
                ys,
                w_sqrtp,
                wstar_norm,
                F_sqrtd,
                step_size,
                ε,
                p,
                cur_i,
                D_non_linearity,
            )
            cur_i += 1

        test_loss = loss_vec_linear_data_perturb(adv_pert, cs, ys, w_sqrtp, F_sqrtd, non_linearity)

        print(
            f"Iteration {idx}, Loss: {np.max(current_loss)} {np.min(current_loss)}, Test Loss: {np.max(test_loss)} {np.min(test_loss)}"
        )

        if np.max(np.abs(current_loss - test_loss)) <= abs_tol:
            break

    return adv_pert


# ---------------------------------------------------------------------------- #
#                          Projected Gradient Descent                          #
# ---------------------------------------------------------------------------- #


# ---------------------------- Non Linear Features --------------------------- #


@njit(error_model="numpy", fastmath=True)
def loss_vec_linear_data_perturb(
    δs: ndarray, cs: ndarray, ys: ndarray, w: ndarray, F_sqrtd: ndarray, non_linearity: callable
):
    preactivations = np.dot(cs + δs, F_sqrtd)  # (n, p)
    loss_vals = ys * np.dot(non_linearity(preactivations), w)  # (n,)
    return loss_vals


@njit(error_model="numpy", fastmath=True)
def grad_vec_linear_data_perturb(
    δs: ndarray, cs: ndarray, ys: ndarray, w: ndarray, F_sqrtd: ndarray, D_non_linearity: callable
):
    preactivations = np.dot(cs + δs, F_sqrtd)  # (n, p)
    grad_vals = ys[:, None] * np.dot(D_non_linearity(preactivations) * w, F_sqrtd.T)  # (n, d)
    return grad_vals


@njit(error_model="numpy", fastmath=True)
def project_onto_orthogonal_space_vec(zs: ndarray, wstar_normalized: ndarray):
    return zs - np.dot(zs, wstar_normalized)[:, None] * wstar_normalized[None, :]


@njit(error_model="numpy", fastmath=True)
def numba_compute_p_norm_vec(zs: ndarray, p: float):
    out = np.zeros(zs.shape[0]).astype(zs.dtype)
    if p == np.inf:
        for i in range(zs.shape[0]):
            out[i] = np.max(np.abs(zs[i]))
    else:
        for i in range(zs.shape[0]):
            out[i] = np.linalg.norm(zs[i], ord=p)
    return out


@njit(error_model="numpy", fastmath=True)
def project_onto_p_ball_vec(zs: ndarray, ε: float, p: float):
    if p == np.inf:
        # For L∞ norm, we clip each component independently to [-ε, ε]
        return np.clip(zs, -ε, ε)
    else:
        # For other Lp norms, we scale the vectors when they exceed the norm bound
        norms = numba_compute_p_norm_vec(zs, p)
        return np.where((norms > ε)[:, None], ε * zs / norms[:, None], zs)


@njit(error_model="numpy", fastmath=True)
def alternating_projections_vec(
    zs: ndarray, wstar_normalized: ndarray, ε: float, p: float, n_steps: int = N_ALTERNATING_PROJ
):
    for _ in range(n_steps):
        zs = project_onto_orthogonal_space_vec(zs, wstar_normalized)
        zs = project_onto_p_ball_vec(zs, ε, p)

    return zs


@njit(error_model="numpy", fastmath=True)
def dykstra_momentum_projections_vec(
    zs: ndarray, wstar_normalized: ndarray, ε: float, p: float, n_steps: int = N_ALTERNATING_PROJ
):
    """
    Dykstra's algorithm with momentum for alternating projections.

    This implementation uses existing projection functions and maintains
    correction vectors to prevent zigzagging between constraints. It also
    includes momentum for faster convergence.

    Args:
        zs: Array of shape (n,d) containing n vectors of dimension d
        wstar_normalized: Normalized vector defining orthogonal space
        ε: Epsilon for p-norm ball constraint
        p: Order of the norm (use np.inf for infinity norm)
        n_steps: Number of iteration steps

    Returns:
        Projected vectors maintaining (n,d) shape
    """
    # Initialize current point and previous point for momentum
    current = zs.copy()
    previous = zs.copy()

    # Initialize correction terms for both projections
    # p1 for orthogonal space, p2 for p-norm ball
    p1 = np.zeros_like(zs)
    p2 = np.zeros_like(zs)

    momentum = 0.9

    for t in range(n_steps):
        old_point = current.copy()

        # First projection (onto orthogonal space) with correction
        y = current + p1
        y_proj = project_onto_orthogonal_space_vec(y, wstar_normalized)
        p1 = y - y_proj  # Update correction term
        current = y_proj

        # Second projection (onto p-ball) with correction
        y = current + p2
        y_proj = project_onto_p_ball_vec(y, ε, p)
        p2 = y - y_proj  # Update correction term
        current = y_proj

        # Apply momentum with increasing schedule
        beta_t = t / (t + 3)  # Momentum grows with iterations
        current = current + beta_t * momentum * (current - previous)
        previous = old_point

        # Ensure feasibility after momentum by projecting onto both constraints
        current = project_onto_orthogonal_space_vec(current, wstar_normalized)
        current = project_onto_p_ball_vec(current, ε, p)

    return current


@njit(error_model="numpy", fastmath=True)
def proj_grad_descent_step(
    δs: ndarray,
    xs: ndarray,
    ys: ndarray,
    w: ndarray,
    wstar_normalized: ndarray,
    F: ndarray,
    step_size: float,
    ε: float,
    p: float,
    t: int,
    D_non_linearity: callable,
):
    gs = grad_vec_linear_data_perturb(δs, xs, ys, w, F, D_non_linearity)
    return alternating_projections_vec(δs - step_size * gs, wstar_normalized, ε, p)


@njit(error_model="numpy", fastmath=True)
def add_gradient_noise(gs: ndarray, noise_scale: float, t: int):
    noise = np.random.normal(0, 1, size=gs.shape).astype(gs.dtype)
    current_scale = np.float32(noise_scale / sqrt(1 + t))
    return gs + current_scale * noise


@njit(error_model="numpy", fastmath=True)
def stochastic_proj_grad_descent_step(
    δs: ndarray,
    xs: ndarray,
    ys: ndarray,
    w: ndarray,
    wstar_normalized: ndarray,
    F: ndarray,
    step_size: float,
    ε: float,
    p: float,
    t: int,
    D_non_linearity: callable,
    noise_scale: float = NOISE_SCALE,
):
    gs = grad_vec_linear_data_perturb(δs, xs, ys, w, F, D_non_linearity)
    noisy_gs = add_gradient_noise(gs, noise_scale, t)
    return alternating_projections_vec(δs - step_size * noisy_gs, wstar_normalized, ε, p)
