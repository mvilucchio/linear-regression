#!/bin/bash

# alpha_min_se, alpha_max_se, n_alphas_se, d, gamma, delta

# python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_alpha_sweep_optimal_regp.py 0.2 3.0 50 500 0.1 0.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_alpha_sweep_optimal_regp.py 0.2 3.0 50 500 0.5 0.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_alpha_sweep_optimal_regp.py 0.2 3.0 50 500 1.0 0.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_alpha_sweep_optimal_regp.py 0.2 3.0 50 500 1.5 0.0 &
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_alpha_sweep_optimal_regp.py 0.2 3.0 50 500 2.0 0.0 &
# python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_alpha_sweep_optimal_regp.py 0.2 3.0 50 500 2.5 0.0 &

wait
