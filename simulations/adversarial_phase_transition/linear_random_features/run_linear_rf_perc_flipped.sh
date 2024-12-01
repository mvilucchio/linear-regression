#!/bin/bash
#SBATCH --chdir=/home/vilucchi/projects/linear-regression
#SBATCH --job-name=linear_rf
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --output=linear_rf_perc_flipped.out
#SBATCH --error=linear_rf_perc_flipped.err
#SBATCH --time=12:00:00
#SBATCH --qos=serial

module load gcc openmpi python/3.10.4
source /home/vilucchi/venvs/my-venv/bin/activate

srun python3 ./simulations/adversarial_phase_transition/linear_random_features/cluster_linear_rf_perc_flipped.py

deactivate
