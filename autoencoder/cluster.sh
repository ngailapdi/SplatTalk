#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


#SBATCH --job-name=freesplat-autoenc

#SBATCH --output=/u/anhthai/develop/FreeSplatLang/srun_out/%x_%j.out

#SBATCH --error=/u/anhthai/develop/FreeSplatLang/srun_out/%x_%j.err

#SBATCH --ntasks=1

#SBATCH --partition=HCESC-H100

#SBATCH --gpus=1

#SBATCH --cpus-per-task=12



#SBATCH --time=48:00:00

#SBATCH --requeue

EXPORT PYTHONUNBUFFERED=True


source /u/anhthai/miniconda3/bin/activate
conda activate freesplat

EXPORT WANDB_API_KEY=f360e0a2ef2980d00dc08522aa2b5453383ffe92
# EXPORT HOME=/coc/flash6/athai6

# module load cuda/11.6

# export CUDA_HOME=/public/apps/cuda/11.6/
# export CUDA_LAUNCH_BLOCKING=1.



cd /u/anhthai/develop/FreeSplatLang/autoencoder


# srun -u bash scripts/train/finetune_ov.sh
# srun -u python train.py --dataset_path /projects/illinois/eng/cs/jrehg/users/anhthai/scannet/scanqa --dataset_name train_scanqa
srun -u python test.py --dataset_path /projects/illinois/eng/cs/jrehg/users/anhthai/scannet/test_scanqa --dataset_name train_scanqa --start $1 --end $2


