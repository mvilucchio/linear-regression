#!/bin/bash

# eps_min, eps_max, n_epss, alpha, eps_training, reg_param, reg, pstar, d

python ./simulations/adversarial_phase_transition/direct_space/finite_vals/sweep_eps_erm_se.py 0.1 10.0 15 0.5 0.0 0.001 2.0 2.0 500 &
python ./simulations/adversarial_phase_transition/direct_space/finite_vals/sweep_eps_erm_se.py 0.1 10.0 15 1.0 0.0 0.001 2.0 2.0 500 &
python ./simulations/adversarial_phase_transition/direct_space/finite_vals/sweep_eps_erm_se.py 0.1 10.0 15 1.5 0.0 0.001 2.0 2.0 500 &

python ./simulations/adversarial_phase_transition/direct_space/finite_vals/sweep_eps_erm_se.py 0.1 10.0 15 0.5 0.0 0.001 2.0 1.0 500 & 
python ./simulations/adversarial_phase_transition/direct_space/finite_vals/sweep_eps_erm_se.py 0.1 10.0 15 1.0 0.0 0.001 2.0 1.0 500 &
python ./simulations/adversarial_phase_transition/direct_space/finite_vals/sweep_eps_erm_se.py 0.1 10.0 15 1.5 0.0 0.001 2.0 1.0 500 &

wait
