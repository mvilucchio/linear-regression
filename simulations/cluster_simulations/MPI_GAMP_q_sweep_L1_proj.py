import linear_regression.regression_numerics.amp_funcs as amp
from linear_regression.sweeps.q_sweeps import sweep_fw_first_arg_GAMP
import linear_regression.regression_numerics.data_generation as dg
from linear_regression.aux_functions.prior_regularization_funcs import (
    f_w_projection_on_sphere,
    Df_w_projection_on_sphere,
)
from linear_regression.fixed_point_equations.fpe_L1_loss import var_hat_func_L1_decorrelated_noise
from linear_regression.fixed_point_equations.fpe_projection_denoising import (
    var_func_projection_denoising,
)
from linear_regression.aux_functions.misc import damped_update
from linear_regression.aux_functions.likelihood_channel_functions import f_out_L1, Df_out_L1
from linear_regression.aux_functions.loss_functions import l1_loss
from linear_regression.regression_numerics.amp_funcs import (
    GAMP_algorithm_unsimplified_mod_3,
)
import numpy as np
import linear_regression.fixed_point_equations as fpe
from linear_regression.utils.errors import ConvergenceError
from linear_regression.regression_numerics.numerics import gen_error_data, train_error_data
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

abs_tol = 1e-8
min_iter = 100
max_iter = 10000
blend_fpe = 0.85

n_features = 10_000
max_nb_iters = 100_000
d = n_features
repetitions = 3
multiplier = int(1)
tol_gamp = 1e-2

alpha, delta_in, delta_out, percentage, beta = 2.0, 1.0, 5.0, 0.3, 0.0
n_samples = max(int(np.around(n_features * alpha)), 1)

filename = f"./results/AMP_results_decorrelated_noise_d{n_features}_{delta_in}_{delta_out}_{percentage}_{beta}_{alpha}_{repetitions}.csv"
print(filename)

# to create the file if it does not exist
if rank == 0 and not os.path.exists(filename):
    with open(filename, "w") as f:
        f.write("#q,gen_err_mean,gen_err_std,train_err_mean,train_err_std,iters_nb_mean,iters_nb_std\n")

# to spread the values of q
if rank == 0:
    qs_amp = np.logspace(-1, 1, size * multiplier, dtype=np.float64)
    chunk_size = len(qs_amp) // size
else:
    qs_amp = None
    chunk_size = None

chunk_size = comm.bcast(chunk_size, root=0)

q_chunk = np.empty(chunk_size, dtype=np.float64)
comm.Scatter(qs_amp, q_chunk, root=0)
print(f"Rank {rank} has chunk {q_chunk}")

gen_err_mean_chunk = np.empty_like(q_chunk)
gen_err_std_chunk = np.empty_like(q_chunk)

train_err_mean_chunk = np.empty_like(q_chunk)
train_err_std_chunk = np.empty_like(q_chunk)

iters_nb_mean_chunk = np.empty_like(q_chunk)
iters_nb_std_chunk = np.empty_like(q_chunk)


for idx, q in enumerate(q_chunk):
    print(f"Rank {rank} is calculating {q}")
    all_gen_err = list()
    all_train_err = list()
    all_iters_nb = list()

    for _ in range(repetitions):
        xs, ys, _, _, ground_truth_theta = dg.data_generation(
            dg.measure_gen_decorrelated,
            n_features=n_features,
            n_samples=max(int(np.around(n_features * alpha)), 1),
            n_generalization=1,
            measure_fun_args=(delta_in, delta_out, percentage, beta),
        )

        while True:
            m = 10 * np.random.random() + 0.01
            sigma = 10 * np.random.random() + 0.01
            if np.square(m) < q + delta_in * q and np.square(m) < q + delta_out * q:
                break

        iter_nb = 0
        err = 100.0
        while err > abs_tol or iter_nb < min_iter:
            m_hat, q_hat, sigma_hat = var_hat_func_L1_decorrelated_noise(
                m, q, sigma, alpha, delta_in, delta_out, percentage, beta
            )
            new_m, _, new_sigma = var_func_projection_denoising(m_hat, q_hat, sigma_hat, q)

            err = max([abs(new_m - m), abs(new_sigma - sigma)])

            m = damped_update(new_m, m, blend_fpe)
            sigma = damped_update(new_sigma, sigma, blend_fpe)

            iter_nb += 1
            if iter_nb > max_iter:
                raise ConvergenceError("fixed_point_finder", iter_nb)

        # we want to initialize them at the fixed point so:
        estimated_theta, iters_nb = GAMP_algorithm_unsimplified_mod_3(
            sigma,
            f_w_projection_on_sphere,
            Df_w_projection_on_sphere,
            f_out_L1,
            Df_out_L1,
            ys,
            xs,
            (q,),
            list(),
            m * ground_truth_theta + np.sqrt(q - m**2) * np.random.normal(size=n_features),
            ground_truth_theta,
            abs_tol=tol_gamp,
            max_iter=max_nb_iters,
            blend=1.0,
        )

        all_gen_err.append(gen_error_data(ys, xs, estimated_theta, ground_truth_theta))

        all_train_err.append(
            train_error_data(ys, xs, estimated_theta, ground_truth_theta, l1_loss, list())
        )

        all_iters_nb.append(iters_nb)

        del xs
        del ys
        del ground_truth_theta

    gen_err_mean_chunk[idx] = np.mean(all_gen_err)
    gen_err_std_chunk[idx] = np.std(all_gen_err)

    train_err_mean_chunk[idx] = np.mean(all_train_err)
    train_err_std_chunk[idx] = np.std(all_train_err)

    iters_nb_mean_chunk[idx] = np.mean(all_iters_nb)
    iters_nb_std_chunk[idx] = np.std(all_iters_nb)


fh = MPI.File.Open(comm, filename, MPI.MODE_WRONLY | MPI.MODE_CREATE | MPI.MODE_APPEND)
try:
    # Write results to the file atomically
    for idx, q in enumerate(q_chunk):
        result_data = bytearray(
            "{:.7e},{:.7e},{:.7e},{:.7e},{:.7e},{:.7e},{:.7e}\n".format(
                q,
                gen_err_mean_chunk[idx],
                gen_err_std_chunk[idx],
                train_err_mean_chunk[idx],
                train_err_std_chunk[idx],
                iters_nb_mean_chunk[idx],
                iters_nb_std_chunk[idx],
            ), 'utf-8'
        )
        fh.Write_ordered(result_data)
finally:
    fh.Close()

MPI.Finalize()