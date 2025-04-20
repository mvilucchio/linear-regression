#!/bin/bash

# alpha_min, alpha_max, n_alpha_pts, d, reps, tau, c, reg_param,delta_in, delta_out, percentage, beta

python ./simulations/tukey_loss/erm_sweep_alpha_tukey.py 1.0 300 30 500 25 1.0 0.001 2.0 0.1 1.0 0.1 0.0
python ./simulations/tukey_loss/erm_sweep_alpha_tukey.py 1.0 300 30 500 25 1.0 0.001 2.0 0.1 1.0 0.2 0.0
python ./simulations/tukey_loss/erm_sweep_alpha_tukey.py 1.0 300 30 500 25 1.0 0.001 2.0 0.1 1.0 0.3 0.0

python ./simulations/tukey_loss/erm_sweep_alpha_tukey.py 1.0 500 30 300 25 1.0 0.001 2.0 0.1 1.0 0.1 0.0
python ./simulations/tukey_loss/erm_sweep_alpha_tukey.py 1.0 500 30 300 25 1.0 0.001 2.0 0.1 1.0 0.2 0.0
python ./simulations/tukey_loss/erm_sweep_alpha_tukey.py 1.0 500 30 300 25 1.0 0.001 2.0 0.1 1.0 0.3 0.0


