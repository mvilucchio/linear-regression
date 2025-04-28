#!/bin/bash

# alpha_min, alpha_max, n_alphas, gamma, eps_t, reg_param

python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 100 0.5 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 100 1.0 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 100 1.5 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 100 2.0 0.0 0.01

python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 100 0.5 0.1 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 100 1.0 0.1 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 100 1.5 0.1 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 100 2.0 0.1 0.01

python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 100 0.5 0.2 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 100 1.0 0.2 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 100 1.5 0.2 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 100 2.0 0.2 0.01

python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 100 0.5 0.3 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 100 1.0 0.3 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 100 1.5 0.3 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_alpha_sweep_hastie.py 0.1 5.0 100 2.0 0.3 0.01

# alpha_min, alpha_max, n_alphas, gamma, eps_t, reg_param

python ./simulations/adversarial_phase_transition/hastie_model/SE_gamma_sweep_hastie.py 0.5 2.0 100 0.5 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_gamma_sweep_hastie.py 0.5 2.0 100 1.0 0.0 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_gamma_sweep_hastie.py 0.5 2.0 100 1.5 0.0 0.01

python ./simulations/adversarial_phase_transition/hastie_model/SE_gamma_sweep_hastie.py 0.5 2.0 100 0.5 0.1 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_gamma_sweep_hastie.py 0.5 2.0 100 1.0 0.1 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_gamma_sweep_hastie.py 0.5 2.0 100 1.5 0.1 0.01

python ./simulations/adversarial_phase_transition/hastie_model/SE_gamma_sweep_hastie.py 0.5 2.0 100 0.5 0.2 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_gamma_sweep_hastie.py 0.5 2.0 100 1.0 0.2 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_gamma_sweep_hastie.py 0.5 2.0 100 1.5 0.2 0.01

python ./simulations/adversarial_phase_transition/hastie_model/SE_gamma_sweep_hastie.py 0.5 2.0 100 0.5 0.3 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_gamma_sweep_hastie.py 0.5 2.0 100 1.0 0.3 0.01
python ./simulations/adversarial_phase_transition/hastie_model/SE_gamma_sweep_hastie.py 0.5 2.0 100 1.5 0.3 0.01
