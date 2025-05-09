#!/bin/bash

# gamma_min, gamma_max, n_gammas, d, alpha, eps_t, delta, reg_param

# python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_gamma_sweep_optimal_regp.py 0.5 2.0 50 0.1 & 
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_gamma_sweep_optimal_regp.py 0.5 2.0 50 0.5 & 
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_gamma_sweep_optimal_regp.py 0.5 2.0 50 1.0 & 
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_gamma_sweep_optimal_regp.py 0.5 2.0 50 1.5 & 
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_gamma_sweep_optimal_regp.py 0.5 2.0 50 2.0 & 
# python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_gamma_sweep_optimal_regp.py 0.5 2.0 50 2.5 & 

wait
