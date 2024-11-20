from numpy import ndarray, tile
from cvxpy import Variable, Problem, Maximize, norm
from jax import jit, grad, vmap
from jax.lax import cond
import jax.numpy as jnp
from typing import Tuple, Optional, List
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
) -> ndarray:
    """
    Find adversarial perturbations in the direct input space using convex optimization.
    Solves the problem: maximize w^T δ subject to ||δ||_p ≤ ε and δ ⊥ w*.
    Uses CVXPY to find the optimal perturbation and applies it to each sample.

    Parameters:
    -----------
    ys : ndarray of shape (n_samples,)
        Labels (+1/-1) for each sample
    xs : ndarray of shape (n_samples, n_features)
        Input samples to perturb
    w : ndarray of shape (n_features,)
        Model weights
    wstar : ndarray of shape (n_features,)
        Vector to maintain orthogonality with perturbations
    ε : float
        Maximum p-norm of perturbations
    p : float
        Norm type (usually 2 or inf)

    Returns:
    --------
    ndarray of shape (n_samples, n_features)
        Adversarial perturbations for each sample
    """
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
) -> ndarray:
    """
    Find adversarial perturbations for a linear random features model using convex optimization.
    Solves the problem: maximize (Fw)^T δ subject to ||δ||_p ≤ ε and δ ⊥ w*.
    Uses CVXPY to find the optimal perturbation for the linearized model.

    Parameters:
    -----------
    ys : ndarray of shape (n_samples,)
        Labels (+1/-1) for each sample
    cs : ndarray of shape (n_samples, n_features)
        Input samples to perturb
    w : ndarray of shape (n_random_features,)
        Model weights in random feature space
    F : ndarray of shape (n_features, n_random_features)
        Random feature matrix
    wstar : ndarray of shape (n_features,)
        Vector to maintain orthogonality with perturbations
    ε : float
        Maximum p-norm of perturbations
    p : float
        Norm type (usually 2 or inf)

    Returns:
    --------
    ndarray of shape (n_samples, n_features)
        Adversarial perturbations for each sample
    """
    _, d = cs.shape
    delta = Variable(d)

    constraints = [norm(delta, p) <= ε, wstar.T @ delta == 0]

    wtilde = F @ w
    objective = Maximize(wtilde.T @ delta)

    problem = Problem(objective, constraints)
    problem.solve()

    return -ys[:, None] * tile(delta.value, (len(ys), 1))


def find_adversarial_perturbation_NLRF(
    ys: jnp.ndarray,
    cs: jnp.ndarray,
    w: jnp.ndarray,
    F: jnp.ndarray,
    wstar: jnp.ndarray,
    ε: float,
    p: float,
    step_size: float = STEP_SIZE_PDG,
    abs_tol: float = TOL_PDG,
    step_block: int = STEP_BLOCK_PDG,
    max_iterations: int = MAX_ITER_PDG,
    adv_pert: Optional[jnp.ndarray] = None,
    test_iters: int = TEST_ITERS_PDG,
    return_losses: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Find adversarial perturbations using projected gradient ascent.

    Parameters:
    -----------
    ys : labels (+1/-1)
    cs : input samples
    w : model weights
    F : random feature matrix
    wstar : vector to maintain orthogonality with perturbations
    ε : maximum p-norm of perturbations
    p : norm type (usually 2 or inf)
    step_size : learning rate for gradient ascent
    abs_tol : tolerance for convergence
    step_block : number of steps between convergence checks
    max_iterations : maximum number of iterations
    adv_pert : initial perturbation (optional)
    test_iters : number of iterations for convergence test
    return_losses : if True, return loss history

    Returns:
    --------
    Tuple[jnp.ndarray, jnp.ndarray]:
        - Final adversarial perturbations
        - Either loss history (shape: (n_iterations, n_samples)) if return_losses=True
          or final loss values (shape: (n_samples,)) if return_losses=False
    """
    if adv_pert is None:
        adv_pert = jnp.zeros_like(cs)
    elif cs.shape != adv_pert.shape:
        raise ValueError("The shape of the adv_pert is not the same as cs.")

    adv_pert = vec_alternating_projections(adv_pert, wstar, ε, p)

    _, d = cs.shape
    all_losses = []

    current_loss = vec_NLRF_data_perturb_loss(adv_pert, cs, ys, w, F, d)
    all_losses.append(current_loss)

    for _ in range(max_iterations):
        for _ in range(step_block):
            adv_pert = PGA_step_NLRF(adv_pert, cs, ys, w, wstar, F, d, step_size, ε, p)

        # Record loss after block update
        current_loss = vec_NLRF_data_perturb_loss(adv_pert, cs, ys, w, F, d)
        all_losses.append(current_loss)

        # Check convergence
        tmp_adv_pert = adv_pert
        for _ in range(test_iters):
            tmp_adv_pert = PGA_step_NLRF(tmp_adv_pert, cs, ys, w, wstar, F, d, step_size, ε, p)

        test_loss = vec_NLRF_data_perturb_loss(tmp_adv_pert, cs, ys, w, F, d)

        if jnp.max(jnp.abs(current_loss - test_loss)) <= abs_tol:
            break

    # Convert losses to array
    losses = jnp.array(all_losses)  # Shape: (n_iterations, n_samples)

    if return_losses:
        return adv_pert, losses
    else:
        return adv_pert, current_loss


# ---------------------------------------------------------------------------- #
#                          Projected Gradient Descent                          #
# ---------------------------------------------------------------------------- #

# ------------------------ non-linear random features ------------------------ #


@jit
def linear_NLRF_data_perturb_loss(δ, c, y, w, F, d: int):
    """Single sample loss function"""
    feature = jnp.tanh((jnp.dot(c + δ, F)) / jnp.sqrt(d))
    return -y * jnp.dot(feature, w)


@jit
def vec_NLRF_data_perturb_loss(δs, cs, ys, w, F, d: int):
    """Vectorized loss function for multiple samples"""
    features = jnp.tanh((cs + δs) @ F / jnp.sqrt(d))
    return -ys * (features @ w)


grad_NLRF_data_perturb = jit(grad(linear_NLRF_data_perturb_loss, argnums=0))

vec_grad_NLRF_data_perturb = jit(vmap(grad_NLRF_data_perturb, in_axes=(0, 0, 0, None, None, None)))


@jit
def project_onto_orthogonal_space(z: jnp.ndarray, wstar: jnp.ndarray) -> jnp.ndarray:
    """Project vector z onto the space orthogonal to wstar."""
    wstar_normalized = wstar / jnp.sqrt(jnp.dot(wstar, wstar))
    return z - jnp.dot(z, wstar_normalized) * wstar_normalized


@jit
def project_onto_p_ball(z: jnp.ndarray, ε: float, p: float) -> jnp.ndarray:
    """Project vector z onto the p-norm ball of radius ε."""
    norm = jnp.sum(jnp.abs(z) ** p) ** (1 / p)
    return cond(norm > ε, lambda x: ε * x[0] / x[1], lambda x: x[0], (z, norm))


@jit
def alternating_projections(z: jnp.ndarray, wstar: jnp.ndarray, ε: float, p: float) -> jnp.ndarray:
    """
    Apply alternating projections between orthogonality constraint and p-norm ball.
    Uses multiple iterations to better approximate the projection onto the intersection.
    """
    n_projections = 7

    for _ in range(n_projections):
        z = project_onto_orthogonal_space(z, wstar)
        z = project_onto_p_ball(z, ε, p)
    return z


vec_alternating_projections = jit(vmap(alternating_projections, in_axes=(0, None, None, None)))


def PGA_step_NLRF(
    δs: jnp.ndarray,
    xs: jnp.ndarray,
    ys: jnp.ndarray,
    w: jnp.ndarray,
    wstar: jnp.ndarray,
    F: jnp.ndarray,
    d: int,
    step_size: float,
    ε: float,
    p: float,
) -> jnp.ndarray:
    """
    Single step of projected gradient ascent with proper handling of constraints.

    1. Compute gradient
    2. Project gradient onto orthogonal space (to maintain feasible direction)
    3. Take step in projected gradient direction
    4. Project result onto constraint set using alternating projections
    """
    g = vec_grad_NLRF_data_perturb(δs, xs, ys, w, F, d)
    g_proj = vec_alternating_projections(g, wstar, ε, p)
    δs_new = δs + step_size * g_proj
    return vec_alternating_projections(δs_new, wstar, ε, p)
