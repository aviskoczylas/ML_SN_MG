#!/bin/bash

#SBATCH --job-name=sn_H_test_datagen
#SBATCH --mail-type=All
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --account=bckiedro0
#SBATCH --partition=standard
#SBATCH --export=ALL
#SBATCH --output=sn_H_test_datagen.out

srun --cpu-bind=cores python sn_solver_fixed_source.py
