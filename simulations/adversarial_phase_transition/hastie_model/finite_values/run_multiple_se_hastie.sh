#!/bin/bash

# alpha_min, alpha_max, n_alphas, gamma, eps_t, reg_param

python ./simulations/adversarial_phase_transition/hastie_model/finite_values/SE_alpha_sweep_hastie.py 0.1 5.0 50 0.1 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/finite_values/SE_alpha_sweep_hastie.py 0.1 5.0 50 1.0 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/finite_values/SE_alpha_sweep_hastie.py 0.1 5.0 50 1.5 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/finite_values/SE_alpha_sweep_hastie.py 0.1 5.0 50 2.0 0.0 0.01

python ./simulations/adversarial_phase_transition/hastie_model/finite_values/SE_alpha_sweep_hastie.py 0.1 5.0 50 0.1 0.0 0.1
python ./simulations/adversarial_phase_transition/hastie_model/finite_values/SE_alpha_sweep_hastie.py 0.1 5.0 50 1.0 0.0 0.1
python ./simulations/adversarial_phase_transition/hastie_model/finite_values/SE_alpha_sweep_hastie.py 0.1 5.0 50 1.5 0.0 0.1
python ./simulations/adversarial_phase_transition/hastie_model/finite_values/SE_alpha_sweep_hastie.py 0.1 5.0 50 2.0 0.0 0.1

# alpha_min, alpha_max, n_alphas, gamma, eps_t, reg_param

python ./simulations/adversarial_phase_transition/hastie_model/finite_values/SE_gamma_sweep_hastie.py 0.1 3.0 50 0.5 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/finite_values/SE_gamma_sweep_hastie.py 0.1 3.0 50 1.0 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/finite_values/SE_gamma_sweep_hastie.py 0.1 3.0 50 1.5 0.0 0.01

python ./simulations/adversarial_phase_transition/hastie_model/finite_values/SE_gamma_sweep_hastie.py 0.1 3.0 50 0.5 0.0 0.1
python ./simulations/adversarial_phase_transition/hastie_model/finite_values/SE_gamma_sweep_hastie.py 0.1 3.0 50 1.0 0.0 0.1
python ./simulations/adversarial_phase_transition/hastie_model/finite_values/SE_gamma_sweep_hastie.py 0.1 3.0 50 1.5 0.0 0.1
