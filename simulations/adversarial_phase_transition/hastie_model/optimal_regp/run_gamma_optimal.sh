#!/bin/bash

# gamma_min, gamma_max, n_gammas, alpha
# gamma_min_se, gamma_max_se, n_gammas_se, d, alpha, delta

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp.py 0.5 2.0 50 0.5
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_gamma_sweep_optimal_regp.py 0.5 2.0 50 500 0.5 0.0

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp.py 0.5 2.0 50 1.0
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_gamma_sweep_optimal_regp.py 0.5 2.0 50 500 1.0 0.0

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp.py 0.5 2.0 50 1.5
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_gamma_sweep_optimal_regp.py 0.5 2.0 50 500 1.5 0.0

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_gamma_sweep_optimal_regp.py 0.5 2.0 50 2.0
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_gamma_sweep_optimal_regp.py 0.5 2.0 50 500 2.0 0.0

