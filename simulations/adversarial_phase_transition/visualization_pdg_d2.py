import numpy as np
import matplotlib.pyplot as plt
from linear_regression.erm.adversarial_perturbation_finders import (
    proj_grad_descent_step,
    loss_vec_linear_data_perturb,
    stochastic_proj_grad_descent_step,
)
from numba import njit
from scipy.spatial import ConvexHull


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
dim = 2
features_dim = 30

epsilon = 3  # L∞ constraint
n_steps = 100
step_size = 0.5
p = 3

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

delta_range = np.linspace(-epsilon * 1.3, epsilon * 1.3, 100)
X, Y = np.meshgrid(delta_range, delta_range)
Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        deltas = np.array([[X[i, j], Y[i, j]], [X[i, j], Y[i, j]]])
        tmp = loss_vec_linear_data_perturb(deltas, xs, ys, w, F, non_linearity)

        assert np.isclose(-tmp[0], tmp[1])

        Z[i, j] = tmp[0]

delta_init = np.zeros((n_samples, dim))
trajectory = [delta_init.copy()]

current_delta = delta_init
for t in range(n_steps):
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
        1.0,
    )

    current_delta = new_delta
    trajectory.append(current_delta.copy())

# Plotting
plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=30)
plt.colorbar(label="Loss")
plt.contour(X, Y, Z, levels=30, colors="k", alpha=0.5)

# Plot trajectory
trajectory = np.array(trajectory)
for i in range(n_samples):
    sample_trajectory = trajectory[:, i, :]
    plt.plot(sample_trajectory[:, 0], sample_trajectory[:, 1], ".-", color=f"C{i}")

# plot the direction of wstar
plt.quiver(
    0,
    0,
    wstar_normalized[0],
    wstar_normalized[1],
    color="r",
    scale=1,
    scale_units="xy",
    angles="xy",
)

# plot the boundary of the Lp ball in red
if p == np.inf:
    square = np.array(
        [
            [-epsilon, -epsilon],
            [-epsilon, epsilon],
            [epsilon, epsilon],
            [epsilon, -epsilon],
            [-epsilon, -epsilon],
        ]
    )
    plt.plot(square[:, 0], square[:, 1], "r--")
else:
    points = []
    for theta in np.linspace(0, 2 * np.pi, 1000):
        x = epsilon * np.cos(theta)
        y = epsilon * np.sin(theta)
        norm = (np.abs(x) ** p + np.abs(y) ** p) ** (1 / p)
        points.append([epsilon * x / norm, epsilon * y / norm])

    points = np.array(points)
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], "r--")


plt.xlabel("δ₁")
plt.ylabel("δ₂")
plt.title("PGD Trajectory on Loss Landscape (L∞ Constraint)")
plt.grid(True)

plt.show()
