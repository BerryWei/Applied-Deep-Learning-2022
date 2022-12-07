#!/bin/bash
#SBATCH --job-name="transformerQA"
#SBATCH --partition=v100-32g
#SBATCH --ntasks=2
#SBATCH --gres=gpu:1
#SBATCH --time=1-0:10
#SBATCH --output=QAcout.txt
#SBATCH --error=QAcerr.txt
#SBATCH --chdir=.
###SBATCH --test-only



bash download.sh
bash run.sh ./data/context.json ./test.json ./submission.csv
