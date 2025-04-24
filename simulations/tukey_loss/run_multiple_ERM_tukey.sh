#!/bin/bash

# alpha_min, alpha_max, n_alpha_pts, d, reps, tau, c, reg_param, delta_in, delta_out, percentage, beta

python3 ./simulations/tukey_loss/erm_sweep_alpha_tukey.py 0.5 1000 20 500 10 1.0 0.001 2.0 0.1 1.0 0.1 0.0
python3 ./simulations/tukey_loss/erm_sweep_alpha_tukey.py 0.5 1000 20 500 10 1.0 0.001 2.0 1.0 1.0 0.1 0.0
python3 ./simulations/tukey_loss/erm_sweep_alpha_tukey.py 0.5 1000 20 500 10 1.0 0.001 2.0 1.0 1.0 0.1 0.1
python3 ./simulations/tukey_loss/erm_sweep_alpha_tukey.py 0.5 1000 20 500 10 1.0 0.001 2.0 0.1 1.0 0.1 0.1
