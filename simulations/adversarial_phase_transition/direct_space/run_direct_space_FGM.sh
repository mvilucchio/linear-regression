#!/bin/bash
#SBATCH --chdir=/home/vilucchi/projects/linear-regression
#SBATCH --job-name=direct_space
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --output=direct_space_FGM.out
#SBATCH --error=direct_space_FGM.err
#SBATCH --time=12:00:00
#SBATCH --qos=serial

module load gcc openmpi python/3.10.4
source /home/vilucchi/venvs/my-venv/bin/activate

srun python3 ./simulations/adversarial_phase_transition/direct_space/cluster_direct_space_erm_FGM.py

deactivate
