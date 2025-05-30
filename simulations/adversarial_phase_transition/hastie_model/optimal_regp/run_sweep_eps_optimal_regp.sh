#!/bin/bash

# eps_min, eps_max, n_epss, alpha, gamma, eps_training
# eps_min, eps_max, n_epss, alpha, gamma, eps_training

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_sweep_eps_optimal_regp.py 0.1 10.0 15 0.5 0.5 0.0
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_sweep_eps_optimal_regp.py 0.1 10.0 15 0.5 0.5 0.0

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_sweep_eps_optimal_regp.py 0.1 10.0 15 0.5 1.0 0.0
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_sweep_eps_optimal_regp.py 0.1 10.0 15 0.5 1.0 0.0

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_sweep_eps_optimal_regp.py 0.1 10.0 15 0.5 1.5 0.0 
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_sweep_eps_optimal_regp.py 0.1 10.0 15 0.5 1.5 0.0

# ---

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_sweep_eps_optimal_regp.py 0.1 10.0 15 1.0 0.5 0.0
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_sweep_eps_optimal_regp.py 0.1 10.0 15 1.0 0.5 0.0

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_sweep_eps_optimal_regp.py 0.1 10.0 15 1.0 1.0 0.0
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_sweep_eps_optimal_regp.py 0.1 10.0 15 1.0 1.0 0.0

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_sweep_eps_optimal_regp.py 0.1 10.0 15 1.0 1.5 0.0
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_sweep_eps_optimal_regp.py 0.1 10.0 15 1.0 1.5 0.0

# ---

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_sweep_eps_optimal_regp.py 0.1 10.0 15 1.5 0.5 0.0
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_sweep_eps_optimal_regp.py 0.1 10.0 15 1.5 0.5 0.0

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_sweep_eps_optimal_regp.py 0.1 10.0 15 1.5 1.0 0.0
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_sweep_eps_optimal_regp.py 0.1 10.0 15 1.5 1.0 0.0

python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/SE_sweep_eps_optimal_regp.py 0.1 10.0 15 1.5 1.5 0.0
python ./simulations/adversarial_phase_transition/hastie_model/optimal_regp/erm_sweep_eps_optimal_regp.py 0.1 10.0 15 1.5 1.5 0.0
