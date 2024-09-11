#!/bin/bash
#SBATCH --partition=compute
#SBATCH --account=bd1179
#SBATCH --nodes=1
#SBATCH --time=07:00:00

srun python ../../src/train_classifier.py