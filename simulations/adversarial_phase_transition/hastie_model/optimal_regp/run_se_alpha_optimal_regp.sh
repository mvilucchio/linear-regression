#!/bin/bash

# alpha_min, alpha_max, n_alphas, gamma, eps_t, reg_param

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 3.0 50 0.1
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 3.0 50 0.5
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 3.0 50 1.0
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 3.0 50 1.5
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 3.0 50 2.0
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_alpha_sweep_optimal_regp.py 0.25 3.0 50 2.5
