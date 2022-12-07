#!/bin/bash
#SBATCH --job-name="env adl_hw2"
#SBATCH --partition=cpu-2gl
#SBATCH --ntasks=1
#SBATCH --time=1-0:0
#SBATCH --output=ENVcout.txt
#SBATCH --error=ENVcerr.txt
#SBATCH --chdir=.
###SBATCH --test-only

module load opt gcc mpi 

pip3 install torch torchvision torchaudio
pip install datasets
pip install transformers
pip install datasets
pip install accelerate
pip install scikit-learn
