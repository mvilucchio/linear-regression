#!/bin/bash

# eps_min, eps_max, n_epss, alpha, gamma, reg_param, eps_training

python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 0.5 0.1
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 1.0 0.1
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 1.5 0.1
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 2.0 0.1

python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 0.5 0.5
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 1.0 0.5
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 1.5 0.5
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 2.0 0.5

python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 0.5 1.0
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 1.0 1.0
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 1.5 1.0
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 2.0 1.0

python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 0.5 1.5
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 1.0 1.5
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 1.5 1.5
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 2.0 1.5

python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 0.5 2.0
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 1.0 2.0
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 1.5 2.0
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 2.0 2.0

python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 0.5 3.0
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 1.0 3.0
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 1.5 3.0
python ./simulations/adversarial_phase_transition/hastie_model/erm_perc_flipped_sweep_eps_optimal.py 0.1 10.0 15 2.0 3.0


