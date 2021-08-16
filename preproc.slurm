#!/bin/bash
#SBATCH --job-name=preproc
#SBATCH --output=%x-%j.out
#SBATCH --partition=rtx-dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00

hostname; pwd; date

inpath="/home1/06431/yueyingn/scratch/dmo-100MPC/dmo-64/test-set0/output/PART_006"
outpath="catnorm"

python preproc.py --inpath "$inpath" --outpath "$outpath"

date

