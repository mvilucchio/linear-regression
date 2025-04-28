#!/bin/bash

# alpha_min, alpha_max, n_alphas, gamma, eps_t, delta, reg_param

python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 15 500 0.5 0.0 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 15 500 1.0 0.0 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 15 500 1.5 0.0 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 15 500 2.0 0.0 0.0 0.01

python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 15 500 0.5 0.1 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 15 500 1.0 0.1 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 15 500 1.5 0.1 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 15 500 2.0 0.1 0.0 0.01

python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 15 500 0.5 0.2 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 15 500 1.0 0.2 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 15 500 1.5 0.2 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 15 500 2.0 0.2 0.0 0.01

python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 15 500 0.5 0.3 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 15 500 1.0 0.3 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 15 500 1.5 0.3 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 15 500 2.0 0.3 0.0 0.01
