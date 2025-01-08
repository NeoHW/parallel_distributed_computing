#!/bin/bash

## This is an example Slurm template job script for A1 that just runs the script and arguments you pass in via `srun`.

#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=4gb
#SBATCH --partition=i7-7700
#SBATCH --time=00:10:00
#SBATCH --output=%x_%j.slurmlog
#SBATCH --error=%x_%j.slurmlog

echo "We are running on $(hostname)"
echo "Job started at $(date)"
echo "Arguments to srun: $@"

# Runs your script with the arguments you passed in
srun $@

echo "Job ended at $(date)"