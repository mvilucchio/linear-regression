#!/bin/bash

# alpha_min, alpha_max, n_alphas, gamma, eps_t, reg_param

python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 50 0.1 0.1 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 50 1.0 0.1 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 50 2.0 0.1 0.01

# alpha_min, alpha_max, n_alphas, gamma, eps_t, reg_param

python ./simulations/adversarial_phase_transition/hastie_model/SE_gamma_sweep_hastie.py 0.1 3.0 50 0.1 0.1 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_gamma_sweep_hastie.py 0.1 3.0 50 1.0 0.1 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_gamma_sweep_hastie.py 0.1 3.0 50 2.0 0.1 0.01
