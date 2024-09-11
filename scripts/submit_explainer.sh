#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=bd1179
#SBATCH --nodes=1
#SBATCH --time=07:00:00

srun python /work/bd1179/b309246/phd_thesis/sarauer23_microphysics_parametrization/pytorch_nextgems/src/plot_classifier_and_explain.py