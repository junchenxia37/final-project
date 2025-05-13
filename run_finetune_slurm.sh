#!/bin/bash

#SBATCH -A trn040
#SBATCH -J sft-finetune
#SBATCH -o .cache/sbatch_logs/%x-%j.out
#SBATCH -e .cache/sbatch_logs/%x-%j.err
#SBATCH -t 01:00:00 
#SBATCH -N 1

# Only necessary if submitting like: sbatch --export=NONE ... (recommended)
# Do NOT include this line when submitting without --export=NONE
unset SLURM_EXPORT_ENV

# load software and activate environment
source setup_env.sh

# run the finetuning
# Output some useful logging info
echo "Running on node: $(hostname)"
echo "Python3 executable path: $(which python3)"
echo "Current working directory: $(pwd)"


# run 
python3 finetune.py
