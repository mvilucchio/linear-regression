#!/bin/bash

# gamma_min, gamma_max, n_gammas, alpha, eps_t, delta, reg_param

# python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_gamma.py 0.5 3.0 10 500 0.1 0.1 0.0 0.01
# python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_gamma.py 0.5 3.0 10 500 1.0 0.1 0.0 0.01
# python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_gamma.py 0.5 3.0 10 500 2.0 0.1 0.0 0.01

python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.5 3.0 10 500 1.0 0.1 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.5 3.0 10 500 2.0 0.1 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.5 3.0 10 500 0.1 0.1 0.0 0.01
