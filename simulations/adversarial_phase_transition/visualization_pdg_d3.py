import numpy as np
import matplotlib.pyplot as plt
from linear_regression.erm.adversarial_perturbation_finders import (
    proj_grad_descent_step,
    loss_vec_linear_data_perturb,
    stochastic_proj_grad_descent_step,
)
from scipy.spatial import ConvexHull
from numba import njit
from scipy.spatial.transform import Rotation


def get_rotation_to_align_z(w):
    """
    Returns a rotation matrix that aligns the z-axis with vector w.

    Parameters:
    w (np.ndarray): Target vector (3D)

    Returns:
    np.ndarray: 3x3 rotation matrix
    """
    # Normalize the target vector
    w = w / np.linalg.norm(w)

    # Define the initial z-axis
    z = np.array([0, 0, 1])

    # If w is already aligned with z or pointing opposite to z,
    # handle these special cases
    if np.allclose(w, z):
        return np.eye(3)
    elif np.allclose(w, -z):
        # 180-degree rotation around x-axis
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    # Calculate the rotation axis (cross product of z and w)
    rotation_axis = np.cross(z, w)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Calculate the rotation angle
    cos_angle = np.dot(z, w)
    angle = np.arccos(cos_angle)

    # Create rotation using axis-angle representation
    rot = Rotation.from_rotvec(rotation_axis * angle)

    return rot.as_matrix()


@njit
def non_linearity(x):
    return np.tanh(x)


@njit
def d_non_linearity(x):
    return 1 - np.tanh(x) ** 2


# @njit
# def non_linearity(x):
#     # implement the relu function
#     return np.maximum(0, x)


# @njit
# def d_non_linearity(x):
#     # implement the derivative of the relu function
#     return (x > 0).astype(np.float64)


# get a random state
state = np.random.randint(0, 1000)
print(f"Random state: {state}")
np.random.seed(state)

n_samples = 2
dim = 3
features_dim = 30

n_big = 1000

epsilon = 3
n_steps = 100
step_size = 0.5
p = np.inf

# Generate data
w_star = np.random.randn(dim)
xs = np.random.randn(n_samples, dim)
xs[0, :] = xs[1, :]
ys = np.array([1, -1])

F = np.random.randn(dim, features_dim)
F = F / np.sqrt(dim)

w = np.random.randn(features_dim)
w = w / np.sqrt(features_dim)

wstar_normalized = w_star / np.linalg.norm(w_star)

# wstar_normalized = np.ones(dim)
# wstar_normalized = np.array([1.0, 1.0, 1.0])
# wstar_normalized = wstar_normalized / np.linalg.norm(wstar_normalized)

print(f"wstar_normalized: {wstar_normalized}")

rot_mat = get_rotation_to_align_z(wstar_normalized)
rot_mat_inv = rot_mat.T

print(f"rot_mat @ rot_mat_inv:\n {rot_mat @ rot_mat_inv}")
print(f"wstar_normalized: {wstar_normalized}")
print(f"z rotated {rot_mat @ np.array([0,0,1])}")
print(f"wstar rotated back: {rot_mat_inv @ wstar_normalized}")

# create the delta range on the plane orthogonal to wstar_normalized
delta_range = np.linspace(-epsilon * 1.7, epsilon * 1.7, 100)
X, Y = np.meshgrid(delta_range, delta_range)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        deltas = np.array([[X[i, j], Y[i, j], 0], [X[i, j], Y[i, j], 0]])
        deltas = (rot_mat @ deltas.T).T
        z = loss_vec_linear_data_perturb(deltas, xs, ys, w, F, non_linearity)
        assert np.isclose(-z[0], z[1])
        Z[i, j] = z[0]

delta_init = np.zeros((n_samples, dim))
trajectory = np.empty((n_steps + 1, n_samples, dim))
trajectory[0] = delta_init.copy()

current_delta = delta_init
for t in range(n_steps):
    # new_delta = proj_grad_descent_step(
    new_delta = stochastic_proj_grad_descent_step(
        current_delta,
        xs,
        ys,
        w,
        wstar_normalized,
        F,
        step_size,
        epsilon,
        p,
        t,
        d_non_linearity,
        0.1,
    )

    current_delta = new_delta
    trajectory[t + 1] = current_delta.copy()

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=30)
plt.colorbar(label="Loss")
plt.contour(X, Y, Z, levels=30, colors="k", alpha=0.5)

print(f"trajectory.shape: {trajectory.shape}")
for i in range(n_samples):
    for j in range(n_steps + 1):
        dot_wstar = np.dot(trajectory[j, i], wstar_normalized)
        if np.abs(dot_wstar) > 1e-6:
            raise ValueError(f"trajectory {j} for sample {i} is not orthogonal to wstar")
        if np.linalg.norm(trajectory[j, i], ord=p) > epsilon + 1e-6:
            raise ValueError(f"trajectory {j} for sample {i} is not of norm epsilon")
print("Trajectory is orthogonal to wstar and norm is epsilon")

for i in range(n_samples):
    sample_trajectory = trajectory[:, i, :]
    rot_sample_trajectory = (rot_mat_inv @ sample_trajectory.T).T
    assert np.allclose(rot_sample_trajectory[:, 2], 0.0, atol=1e-6)
    plt.plot(
        rot_sample_trajectory[:, 0],
        rot_sample_trajectory[:, 1],
        "o-",
        color=f"C{i}",
        label=f"y = {ys[i]}",
    )

# plot the limit of the constraint for Lp ball in the orthogonal space of wstar for any p
points = np.empty((n_big, dim))
for i, theta in enumerate(np.linspace(0, 2 * np.pi, n_big)):
    x = epsilon * np.cos(theta)
    y = epsilon * np.sin(theta)
    tmp_pt = np.array([x, y, 0.0])
    rotated_tmp_pt = rot_mat @ tmp_pt
    rotated_tmp_pt = rotated_tmp_pt / np.linalg.norm(rotated_tmp_pt, ord=p) * epsilon
    points[i] = rot_mat_inv @ rotated_tmp_pt

hull = ConvexHull(points[:, :2])
for simplex in hull.simplices:
    plt.plot(points[simplex, 0], points[simplex, 1], "r--")

plt.plot(points[:, 0], points[:, 1], "r--")

plt.title("PGD Trajectory on Loss Landscape")
plt.grid(True)
plt.legend()

plt.show()
