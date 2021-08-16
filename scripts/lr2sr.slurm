#!/bin/bash
#SBATCH -p rtx-dev
#SBATCH -N 1
#SBATCH --ntasks-per-node 1
#SBATCH --job-name=gen-sr
#SBATCH --output=%x-%j.out
#SBATCH --time=1:00:00

hostname; pwd; date

model_path="SRmodel/G_z0.pt" # path of the SR model
redshift=0 
lr_input="catnorm.npy" # normalized LR input in shape Nc,Ng,Ng,Ng, with Nc=6
sr_path="SR-output" # output directory of SR columns
Boxsize=100000 # Boxsize of LR/HR/SR, in kpc/h
nsplit=4 # split the LR box into nsplit^3 chunks to perform SR, increase nsplit if out of GPU memory, make sure that nsplit devides Ng_lr

python lr2sr.py --model-path "$model_path" --redshift $redshift --lr-input "$lr_input" --sr-path "$sr_path" 

date

