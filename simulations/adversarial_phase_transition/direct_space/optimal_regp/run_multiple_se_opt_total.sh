#!/bin/bash

# alpha_min, alpha_max, n_alphas, pstar, reg_p, metric_type_chosen
# alpha_min_se, alpha_max_se, n_alphas_se, d, delta

# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_both_regp_epst.py 0.25 3.0 50 1.0 1.0 misclass > output_misclass_p1.0_reg1.0.log 2>error_misclass_p1.0_reg1.0.err &
# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_both_regp_epst.py 0.25 3.0 50 1.0 1.0 bound > output_bound_p1.0_reg1.0.log 2>error_bound_p1.0_reg1.0.err &
# python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_both_regp_epst.py 0.25 3.0 50 1.0 1.0 adv > output_adv_p1.0_reg1.0.log 2>error_adv_p1.0_reg1.0.err &

python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_both_regp_epst.py 0.25 3.0 50 2.0 2.0 misclass > output_misclass_p2.0_reg2.0.log 2>error_misclass_p2.0_reg2.0.err &
python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_both_regp_epst.py 0.25 3.0 50 2.0 2.0 bound > output_bound_p2.0_reg2.0.log 2>error_bound_p2.0_reg2.0.err &
python ./simulations/adversarial_phase_transition/direct_space/optimal_regp/SE_alpha_sweep_optimal_both_regp_epst.py 0.25 3.0 50 2.0 2.0 adv > output_adv_p2.0_reg2.0.log 2>error_adv_p2.0_reg2.0.err &

wait
