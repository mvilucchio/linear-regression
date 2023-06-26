import linear_regression.regression_numerics.data_generation as dg
from linear_regression.regression_numerics.erm_solvers import vanillaGD_Huber
import numpy as np
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"Rank {rank} of {size} is alive")

delta_in, delta_out, percentage, beta, a = 1.0, 5.0, 0.3, 0.0, 1.0
reg_param = -1.50
alpha = 30.0
multiplier = int(1)

d = 2_000
lr = 0.01
T = 120

if rank == 0:
    params = (delta_in, delta_out, percentage, beta)
    xs, ys, _, _, ground_truth_theta = dg.data_generation(
        dg.measure_gen_decorrelated,
        n_features=d,
        n_samples=max(int(np.around(d * alpha)), 1),
        n_generalization=1,
        measure_fun_args=params,
    )
    xs_norm = np.divide(xs, np.sqrt(d))
else:
    xs = None
    ys = None
    ground_truth_theta = None

xs = comm.bcast(xs, root=0)
ys = comm.bcast(ys, root=0)
ground_truth_theta = comm.bcast(ground_truth_theta, root=0)

# to spread the values of q
if rank == 0:
    q_inits = np.logspace(np.log(1) / np.log(10), np.log(20) / np.log(10), size * multiplier, dtype=np.float64)
    chunk_size = len(q_inits) // size
else:
    q_inits = None
    chunk_size = None

chunk_size = comm.bcast(chunk_size, root=0)

q_inits_chunk = np.empty(chunk_size, dtype=np.float64)
comm.Scatter(q_inits, q_inits_chunk, root=0)
print(f"Rank {rank} has chunk {q_inits_chunk}")

#
for i, q_init in enumerate(q_inits_chunk):
    w = np.random.normal(loc=0.0, scale=1.0, size=(d,))
    w = w / np.linalg.norm(w) * q_init * np.sqrt(d)

    filename = f"./simulations/data/vanilla_GD/vanilla_GD_Huber_d{d}_q{q_init}_{delta_in}_{delta_out}_{percentage}_{beta}_{alpha}.csv"
    print(filename)

    # to create the file if it does not exist
    if not os.path.exists(filename):
        file = open(filename, "w")
        file.close()

    print(f"Rank {rank} is running q={q_init}")
    w_hat, loss_list, q_list, E_list = vanillaGD_Huber(
        ys,
        xs,
        reg_param=reg_param,
        a=a,
        lr=lr,
        w_init=w,
        max_iters=T,
        save_run=True,
        ground_truth_theta=ground_truth_theta,
    )
    print(f"Rank {rank} is saving q={q_init}")

    np.savetxt(filename, np.c_[np.arange(len(loss_list)), loss_list, q_list, E_list], delimiter=",")

MPI.Finalize()
