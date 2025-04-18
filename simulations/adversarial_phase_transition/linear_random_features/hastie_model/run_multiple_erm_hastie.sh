#!/bin/bash

# alpha_min, alpha_max, n_alphas, gamma, eps_t, delta, reg_param

python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 20 0.5 0.0 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 20 1.0 0.0 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 20 1.5 0.0 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 20 2.0 0.0 0.0 0.01

python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 20 0.5 0.1 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 20 1.0 0.1 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 20 1.5 0.1 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 20 2.0 0.1 0.0 0.01

python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 20 0.5 0.2 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 20 1.0 0.2 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 20 1.5 0.2 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_alpha.py 0.1 5.0 20 2.0 0.2 0.0 0.01

# gamma_min, gamma_max, n_gammas, alpha, eps_t, delta, reg_param

python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_gamma.py 0.5 2.0 20 0.5 0.0 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_gamma.py 0.5 2.0 20 1.0 0.0 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_gamma.py 0.5 2.0 20 1.5 0.0 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_gamma.py 0.5 2.0 20 2.0 0.0 0.0 0.01

python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_gamma.py 0.5 2.0 20 0.5 0.1 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_gamma.py 0.5 2.0 20 1.0 0.1 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_gamma.py 0.5 2.0 20 1.5 0.1 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_gamma.py 0.5 2.0 20 2.0 0.1 0.0 0.01

python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_gamma.py 0.5 2.0 20 0.5 0.2 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_gamma.py 0.5 2.0 20 1.0 0.2 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_gamma.py 0.5 2.0 20 1.5 0.2 0.0 0.01
python ./simulations/adversarial_phase_transition/linear_random_features/hastie_model/erm_hastie_sweep_gamma.py 0.5 2.0 20 2.0 0.2 0.0 0.01
