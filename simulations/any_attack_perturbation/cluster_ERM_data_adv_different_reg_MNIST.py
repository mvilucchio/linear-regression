import matplotlib.pyplot as plt
import numpy as np
from linear_regression.erm.erm_solvers import (
    find_coefficients_Logistic_adv,
    find_coefficients_Logistic_adv_Linf_L1,
)
from linear_regression.erm.metrics import (
    estimation_error_data,
    generalisation_error_classification,
    adversarial_error_data,
)
import pickle
from tqdm.auto import tqdm
from mpi4py import MPI
import os
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

# MNIST setup parameters
MNIST_DIR = "./datasets/MNIST"
# Choose two digits to classify (e.g., 0 vs 1)
DIGIT_1 = 0
DIGIT_2 = 1

alpha_min, alpha_max, n_alpha_pts = 0.06, 1.0, 12
reg_orders = [1.001, 2, 3]
eps_t = 2.0
eps_g = 2.0
reg_params = [1e-3]
pstar = 1.0

reps = 10
n_gen = 1000

data_folder = "./data"
file_name = f"MNIST_ERM_data_Linf_reg_order_{{:d}}_alpha_{alpha_min:.3f}_{alpha_max:.3f}_{n_alpha_pts:d}_digits_{DIGIT_1}vs{DIGIT_2}_reps_{reps:d}_reg_param_{{:.1e}}_eps_t_g_{eps_t:.1e}_{eps_g:.1e}.pkl"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert len(reg_orders) >= size

reg_order = reg_orders[rank]


def load_mnist():
    """Load MNIST dataset from local directory with PyTorch .pt files"""
    import torch

    processed_dir = os.path.join(MNIST_DIR, "processed")
    if os.path.exists(processed_dir):
        # Load training data
        train_data = torch.load(os.path.join(processed_dir, "training.pt"))
        train_images, train_labels = train_data

        # Load test data
        test_data = torch.load(os.path.join(processed_dir, "test.pt"))
        test_images, test_labels = test_data

        # Convert to numpy arrays
        train_images = train_images.numpy()
        train_labels = train_labels.numpy()
        test_images = test_images.numpy()
        test_labels = test_labels.numpy()

        # Flatten images and concatenate train and test sets
        train_images = train_images.reshape(train_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)

        X = np.vstack([train_images, test_images])
        y = np.concatenate([train_labels, test_labels])

        # Normalize pixel values to [0, 1]
        X = X / 255.0

        return X, y


def prepare_binary_mnist_data(digit1, digit2, n_train, n_test, seed=None):
    """
    Prepare binary classification data from MNIST for two specified digits.
    Returns training and test sets.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    # Load MNIST
    X, y = load_mnist()

    # Select only the two digits we want
    mask = (y == str(digit1)) | (y == str(digit2))
    if isinstance(y[0], str):
        mask = (y == str(digit1)) | (y == str(digit2))
    else:
        mask = (y == digit1) | (y == digit2)
    X_binary = X[mask]
    y_binary = y[mask]

    # Convert labels to -1, 1 (handle both string and int label formats)
    if isinstance(y_binary[0], str):
        y_binary = np.where(y_binary == str(digit1), -1, 1)
    else:
        y_binary = np.where(y_binary == digit1, -1, 1)

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_binary,
        y_binary,
        train_size=n_train,
        test_size=n_test,
        stratify=y_binary,
        random_state=seed,
    )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


# Remove get_ground_truth_weights function

for reg_param in reg_params:
    # For MNIST, d is fixed as the flattened image dimension (784 for MNIST)
    d = 784
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha_pts)

    q_mean = np.empty_like(alphas)
    q_std = np.empty_like(alphas)

    p_mean = np.empty_like(alphas)
    p_std = np.empty_like(alphas)

    train_error_mean = np.empty_like(alphas)
    train_error_std = np.empty_like(alphas)

    gen_error_mean = np.empty_like(alphas)
    gen_error_std = np.empty_like(alphas)

    adversarial_errors_mean = np.empty_like(alphas)
    adversarial_errors_std = np.empty_like(alphas)

    for j, alpha in enumerate(alphas):
        n = int(alpha * d)  # Number of training samples based on alpha

        print(
            f"process {rank}/{size} running reg_order = {reg_order}, reg_param = {reg_param:.1e}, alpha = {alpha:.4f} (= {n:d} samples / {d:d} features)"
        )

        tmp_train_errors = []
        tmp_gen_errors = []
        tmp_adversarial_errors = []
        tmp_qs = []
        tmp_ps = []

        iter = 0
        pbar = tqdm(total=reps)
        while iter < reps:
            # For each repetition, sample different images
            seed = 42 + iter  # Different seed for each repetition

            try:
                # Prepare MNIST data for the current alpha
                xs_train, ys_train, xs_gen, ys_gen = prepare_binary_mnist_data(
                    DIGIT_1, DIGIT_2, n, n_gen, seed=seed
                )

                # # Find coefficients using the appropriate solver
                # if reg_order == 1:
                #     w = find_coefficients_Logistic_adv_Linf_L1(ys_train, xs_train, reg_param, eps_t)
                # else:
                #     # For reg_order > 1, we need a substitute for wstar
                #     # Use a zero vector as a non-informative prior
                #     wstar_placeholder = np.zeros(xs_train.shape[1])

                #     w = find_coefficients_Logistic_adv(
                #         ys_train, xs_train, reg_param, eps_t, reg_order, pstar, wstar_placeholder
                #     )
                wstar_placeholder = np.zeros(xs_train.shape[1])

                w = find_coefficients_Logistic_adv(
                    ys_train, xs_train, reg_param, eps_t, reg_order, pstar, wstar_placeholder
                )

                # Calculate metrics
                tmp_qs.append(np.sum(w**2) / d)
                tmp_ps.append(np.sum(np.abs(w) ** pstar) / d)

                # Calculate classification errors using predictions instead of ground truth comparison
                y_train_pred = np.sign(xs_train @ w)
                train_error = np.mean(y_train_pred != ys_train)
                tmp_train_errors.append(train_error)

                y_gen_pred = np.sign(xs_gen @ w)
                gen_error = np.mean(y_gen_pred != ys_gen)
                tmp_gen_errors.append(gen_error)

                # For adversarial error, we still need a reference but we can use w itself
                # This measures robustness to perturbations
                adv_error = adversarial_error_data(ys_gen, xs_gen, w, w, eps_g, pstar)
                tmp_adversarial_errors.append(adv_error)

                iter += 1
                pbar.update(1)

            except ValueError as e:
                print(f"Error in iteration {iter}: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error in iteration {iter}: {e}")
                continue

        pbar.close()

        if len(tmp_train_errors) > 0:
            q_mean[j] = np.mean(tmp_qs)
            q_std[j] = np.std(tmp_qs) / np.sqrt(reps)

            p_mean[j] = np.mean(tmp_ps)
            p_std[j] = np.std(tmp_ps) / np.sqrt(reps)

            train_error_mean[j] = np.mean(tmp_train_errors)
            train_error_std[j] = np.std(tmp_train_errors) / np.sqrt(reps)

            gen_error_mean[j] = np.mean(tmp_gen_errors)
            gen_error_std[j] = np.std(tmp_gen_errors) / np.sqrt(reps)

            adversarial_errors_mean[j] = np.mean(tmp_adversarial_errors)
            adversarial_errors_std[j] = np.std(tmp_adversarial_errors) / np.sqrt(reps)
        else:
            print(f"Warning: No successful iterations for alpha={alpha:.4f}")
            # Fill with NaN for this alpha
            q_mean[j] = np.nan
            q_std[j] = np.nan
            p_mean[j] = np.nan
            p_std[j] = np.nan
            train_error_mean[j] = np.nan
            train_error_std[j] = np.nan
            gen_error_mean[j] = np.nan
            gen_error_std[j] = np.nan
            adversarial_errors_mean[j] = np.nan
            adversarial_errors_std[j] = np.nan

    # Save results
    data_dict = {
        "alphas": alphas,
        "q_mean": q_mean,
        "q_std": q_std,
        "p_mean": p_mean,
        "p_std": p_std,
        "train_error_mean": train_error_mean,
        "train_error_std": train_error_std,
        "gen_error_mean": gen_error_mean,
        "gen_error_std": gen_error_std,
        "adversarial_error_mean": adversarial_errors_mean,
        "adversarial_error_std": adversarial_errors_std,
    }

    save_path = os.path.join(data_folder, file_name.format(int(reg_order), reg_param))
    with open(save_path, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"Results saved to {save_path}")
