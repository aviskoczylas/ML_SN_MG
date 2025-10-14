#!/bin/bash

#SBATCH --job-name=sn_datagen
#SBATCH --mail-type=All
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --account=bckiedro0
#SBATCH --partition=standard
#SBATCH --export=ALL
#SBATCH --output=sn_datagen.out

python sn_solver_class.py
