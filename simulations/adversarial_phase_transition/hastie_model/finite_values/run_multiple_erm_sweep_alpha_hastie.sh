#!/bin/bash

# alpha_min, alpha_max, n_alphas, gamma, eps_t, delta, reg_param

python ./simulations/adversarial_phase_transition/hastie_model/finite_values/erm_hastie_sweep_alpha.py 0.1 3.0 10 500 0.5 0.0 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/finite_values/erm_hastie_sweep_alpha.py 0.1 3.0 10 500 1.0 0.0 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/finite_values/erm_hastie_sweep_alpha.py 0.1 3.0 10 500 1.5 0.0 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/finite_values/erm_hastie_sweep_alpha.py 0.1 3.0 10 500 2.0 0.0 0.0 0.01

python ./simulations/adversarial_phase_transition/hastie_model/finite_values/erm_hastie_sweep_alpha.py 0.1 5.0 10 500 0.5 0.0 0.0 0.1
python ./simulations/adversarial_phase_transition/hastie_model/finite_values/erm_hastie_sweep_alpha.py 0.1 5.0 10 500 1.0 0.0 0.0 0.1
python ./simulations/adversarial_phase_transition/hastie_model/finite_values/erm_hastie_sweep_alpha.py 0.1 5.0 10 500 1.5 0.0 0.0 0.1
python ./simulations/adversarial_phase_transition/hastie_model/finite_values/erm_hastie_sweep_alpha.py 0.1 5.0 10 500 2.0 0.0 0.0 0.1
