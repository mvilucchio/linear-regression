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
from linear_regression.utils.errors import ConvergenceError
from linear_regression.regression_numerics.numerics import gen_error_data, train_error_data
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

abs_tol = 1e-8
min_iter = 100
max_iter = 10000
blend_fpe = 0.85

n_features = 100
d = n_features
repetitions = 10
multiplier = int(3)

alpha, delta_in, delta_out, percentage, beta = 2.0, 1.0, 5.0, 0.3, 0.0
n_samples = max(int(np.around(n_features * alpha)), 1)

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
            abs_tol=1e-2,
            max_iter=100,
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

if rank == 0:
    results_gen_err_mean = np.empty(len(qs_amp), dtype=np.float64)
    results_gen_err_std = np.empty(len(qs_amp), dtype=np.float64)
    
    results_train_err_mean = np.empty(len(qs_amp), dtype=np.float64)
    results_train_err_std = np.empty(len(qs_amp), dtype=np.float64)

    results_iters_nb_mean = np.empty(len(qs_amp), dtype=np.float64)
    results_iters_nb_std = np.empty(len(qs_amp), dtype=np.float64)
else:
    results_gen_err_mean = None
    results_gen_err_std = None
    
    results_train_err_mean = None
    results_train_err_std = None

    results_iters_nb_mean = None
    results_iters_nb_std = None

print(f"Rank {rank} is gathering")
comm.Gather(gen_err_mean_chunk, results_gen_err_mean, root=0)
comm.Gather(gen_err_std_chunk, results_gen_err_std, root=0)

comm.Gather(train_err_mean_chunk, results_train_err_mean, root=0)
comm.Gather(train_err_std_chunk, results_train_err_std, root=0)

comm.Gather(iters_nb_mean_chunk, results_iters_nb_mean, root=0)
comm.Gather(iters_nb_std_chunk, results_iters_nb_std, root=0)

if rank == 0:
    # save the results of AMP in a file with the delta_in, delta_out, percentage, beta, alpha,n_features, repetitions parameters
    np.savetxt(
        f"./results/AMP_results_decorrelated_noise_{len(qs_amp)}_params_{delta_in}_{delta_out}_{percentage}_{beta}_{alpha}_{n_features}_{repetitions}.csv",
        np.vstack(
            [
                qs_amp,
                results_gen_err_mean,
                results_gen_err_std,
                results_train_err_mean,
                results_train_err_std,
                results_iters_nb_mean,
                results_iters_nb_std,
            ]
        ).T,
        delimiter=",",
        header="q,gen_err_mean,gen_err_std,train_err_mean,train_err_std,iters_nb_mean,iters_nb_std",
    )
