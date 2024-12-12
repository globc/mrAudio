#!/bin/bash -l
#
#SBATCH -e test.err
#SBATCH -o test.out

#SBATCH -n 1 # 1 process
#SBATCH -c 4 # 4 CPU cores per process

#SBATCH --time=00:10:00

#SBATCH --gres=gpu:A100:2
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_NUM_DEVICES=$SLURM_GPUS_ON_NODE

export HPC_SCRATCH=$WORK

ml gcc/11 cuda/11.8
source ~/miniconda3/bin/activate
conda activate mraudio
pip install git+https://github.com/salesforce/LAVIS --no-deps
pip install -r requirements_xinstructblip.txt

./scripts/train/X-InstructBLIP/qvh.sh
conda deactivate


