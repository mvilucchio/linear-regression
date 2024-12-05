from numpy import around, empty, sum, mean, std, square, divide, sqrt, dot, sign
from math import acos, pi
import numpy as np
from numba import njit


# --------------------------- Llinear Model Metrics -------------------------- #


def estimation_error_data(ys, xs, w, wstar):
    _, d = xs.shape
    return sum((wstar - w) ** 2) / d


def train_error_data(ys, xs, w, wstar, loss_function, loss_function_args):
    n, d = xs.shape
    xs_norm = xs / sqrt(d)
    tmp = loss_function(ys, xs_norm @ w, *loss_function_args)
    return sum(tmp) / n


def angle_teacher_student_data(ys, xs, w, wstar):
    tmp = dot(w, wstar) / sqrt(dot(w, w) * dot(wstar, wstar))
    return acos(tmp) / pi


def generalisation_error_classification(ys, xs, w, wstar):
    return mean(ys != sign(xs @ w))


def adversarial_error_data(ys, xs, w, wstar, eps, pstar):
    _, d = xs.shape
    tmp = sign(xs @ w / sqrt(d) - eps * ys * sum(abs(w) ** pstar) ** (1 / pstar) / d**pstar)
    return mean(ys != tmp)


def adversarial_error_data_Sigmaupsilon(ys, xs, w, wstar, Sigmaupsilon, eps):
    _, d = xs.shape
    tmp = sign(xs @ w / sqrt(d) - eps / sqrt(d) * sqrt(dot(w, Sigmaupsilon @ w)) * ys)
    return mean(ys != tmp)


# ---------------------------- Adversarial Errors ---------------------------- #


def percentage_different_labels_estim(
    ys,
    w,
    xs_pertubed,
    model_type: str = "linear",
    F: np.ndarray = None,
    non_linearity: callable = None,
) -> float:
    if model_type == "linear":
        return mean(ys != sign(xs_pertubed @ w))
    elif model_type == "linear_rf":
        if F is None:
            raise ValueError("Hidden model requires projection matrix")
        return mean(ys != sign(xs_pertubed @ F @ w))
    elif model_type == "non_linear_rf":
        if F is None:
            raise ValueError("Hidden model requires projection matrix")
        if non_linearity is None:
            return mean(ys != sign(xs_pertubed @ F @ w))
        d = xs_pertubed.shape[1]
        return mean(ys != sign(non_linearity(xs_pertubed @ F / sqrt(d)) @ w))
    else:
        raise ValueError("Model type not recognized")


def percentage_flipped_labels_estim(
    ys,
    xs,
    w,
    wstar,
    xs_pertubed,
    hidden_model=False,
    projection_matrix=None,
    non_linearity: callable = None,
) -> float:
    """
    Estimates the percentage of labels that have been flipped due to perturbation in the input features.

    This function compares the predictions made using the original features (xs) against those made
    using perturbed features (xs_pertubed) to determine what fraction of labels would change. It supports
    both standard linear models and hidden models with optional non-linear transformations.

    Parameters
    ----------
    ys : array-like
        True labels for the dataset.
    xs : array-like
        Original feature matrix of shape (n_samples, n_features).
    w : array-like
        Current model weights.
    wstar : array-like
        True/optimal model weights (unused in current implementation but kept for API consistency).
    xs_pertubed : array-like
        Perturbed feature matrix of the same shape as xs.
    hidden_model : bool, optional (default=False)
        If True, uses a hidden model architecture with projection.
    projection_matrix : array-like, optional (default=None)
        Projection matrix for hidden model. Required if hidden_model=True.
    non_linearity : callable, optional (default=None)
        Non-linear transformation function to apply in hidden model case.
        If None and hidden_model=True, uses linear projection only.

    Returns
    -------
    float
        Proportion of labels that would flip due to the perturbation (between 0 and 1).

    Raises
    ------
    ValueError
        If hidden_model=True and projection_matrix is None.

    Notes
    -----
    For hidden models with non-linearity, the projection is scaled by 1/sqrt(d) where d is the
    feature dimension to maintain stable variance.
    """
    if hidden_model:
        if projection_matrix is None:
            raise ValueError("Hidden model requires projection matrix")
        if non_linearity is None:
            return mean(
                sign(xs @ projection_matrix @ w) != sign(xs_pertubed @ projection_matrix @ w)
            )
        else:
            _, d = xs.shape
            return mean(
                sign(non_linearity(xs @ projection_matrix / sqrt(d)) @ w)
                != sign(non_linearity(xs_pertubed @ projection_matrix / sqrt(d)) @ w)
            )

    return mean(sign(xs @ w) != sign(xs_pertubed @ w))


def percentage_error_from_true(
    ys,
    xs,
    w,
    wstar,
    xs_pertubed,
    hidden_model=False,
    projection_matrix=None,
    non_linearity: callable = None,
) -> float:
    """
    Calculates the percentage of predictions that differ from the true labels when using perturbed features.

    This function measures the model's accuracy degradation when making predictions using perturbed
    features compared to the true labels. It supports both standard linear models and hidden models
    with optional non-linear transformations.

    Parameters
    ----------
    ys : array-like
        True labels for the dataset.
    xs : array-like
        Original feature matrix of shape (n_samples, n_features).
    w : array-like
        Current model weights.
    wstar : array-like
        True/optimal model weights (unused in current implementation but kept for API consistency).
    xs_pertubed : array-like
        Perturbed feature matrix of the same shape as xs.
    hidden_model : bool, optional (default=False)
        If True, uses a hidden model architecture with projection.
    projection_matrix : array-like, optional (default=None)
        Projection matrix for hidden model. Required if hidden_model=True.
    non_linearity : callable, optional (default=None)
        Non-linear transformation function to apply in hidden model case.
        If None and hidden_model=True, uses linear projection only.

    Returns
    -------
    float
        Proportion of predictions that differ from true labels (between 0 and 1).

    Raises
    ------
    ValueError
        If hidden_model=True and projection_matrix is None.

    Notes
    -----
    For hidden models with non-linearity, the projection is scaled by 1/sqrt(d) where d is the
    feature dimension to maintain stable variance.

    Unlike percentage_flipped_labels_estim, this function compares against the true labels (ys)
    rather than comparing predictions between original and perturbed features.
    """
    if hidden_model:
        if projection_matrix is None:
            raise ValueError("Hidden model requires projection matrix")
        if non_linearity is None:
            return mean(sign(ys) != sign(xs_pertubed @ projection_matrix @ w))
        else:
            _, d = xs.shape
            return mean(
                sign(ys) != sign(non_linearity(xs_pertubed @ projection_matrix / sqrt(d)) @ w)
            )

    return mean(sign(ys) != sign(xs_pertubed @ w))


# @jit
# def percentage_flipped_labels_NLRF(ys, xs, w, wstar, xs_pertubed, projection_matrix, d):
#     return jnp.mean(
#         jnp.sign(jnp.tanh(xs @ projection_matrix / jnp.sqrt(d)) @ w)
#         != jnp.sign(jnp.tanh(xs_pertubed @ projection_matrix / jnp.sqrt(d)) @ w)
#     )


def percentage_flipped_labels_NLRF(ys, xs, w, wstar, xs_pertubed, F, d, non_linearity):
    return np.mean(ys != np.sign(non_linearity(xs_pertubed @ F / np.sqrt(d)) @ w))


# @jit
# def single_percentage_flipped_labels_NLRF(y, x, w, wstar, x_pertubed, projection_matrix, d):
#     return jnp.heaviside(
#         -y * jnp.sign(jnp.dot(w, jnp.tanh(jnp.dot(x_pertubed, projection_matrix)))), 0
#     )


# percentage_flipped_labels_NLRF = jit(
#     vmap(single_percentage_flipped_labels_NLRF, in_axes=(0, 0, None, None, 0, None, None))
# )


def percentage_flipped_labels_estim_nonlinear(
    ys,
    xs,
    w,
    wstar,
    xs_pertubed,
    hidden_model=False,
    projection_matrix=None,
):
    if hidden_model:
        if projection_matrix is None:
            raise ValueError("Hidden model requires projection matrix")
        return mean(sign(xs @ projection_matrix @ w) != sign(xs_pertubed @ projection_matrix @ w))

    return mean(sign(xs @ w) != sign(xs_pertubed @ w))


# --------------------------- Overlaps Estimations --------------------------- #


def m_real_overlaps(ys, xs, w, wstar):
    d = xs.shape[1]
    m = dot(w, wstar) / d
    return m


def q_real_overlaps(ys, xs, w, wstar):
    d = xs.shape[1]
    q = sum(square(w)) / d
    return q
