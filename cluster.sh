#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree


#SBATCH --job-name=fvt_full_100v_32r_256_mse_cos

#SBATCH --output=/u/anhthai/develop/FreeSplatLang/outputs/train_fvt_full_100v_32r_256_mse_cos/%x_%j.out

#SBATCH --error=/u/anhthai/develop/FreeSplatLang/outputs/train_fvt_full_100v_32r_256_mse_cos/%x_%j.err


#SBATCH --ntasks=1

#SBATCH --partition=HCESC-H100

#SBATCH --gpus=1


#SBATCH --mem=64G

#SBATCH --time=7-00:00:00

#SBATCH --requeue
EXPORT PYTHONUNBUFFERED=True


source /u/anhthai/miniconda3/bin/activate
conda activate freesplat

EXPORT WANDB_API_KEY=f360e0a2ef2980d00dc08522aa2b5453383ffe92
# EXPORT HOME=/coc/flash6/athai6

# module load cuda/11.6

# export CUDA_HOME=/public/apps/cuda/11.6/
# export CUDA_LAUNCH_BLOCKING=1.



cd /u/anhthai/develop/FreeSplatLang


# srun -u python -m src.main +experiment=scannet/fvt +output_dir=train_fvt_full_100v_32r_256_mse_cos_learned_feat_dec

# srun -u python -m src.main +experiment=scannet/fvt +output_dir=train_fvt_full_100v_32r_256 checkpointing.load='outputs/train_fvt_full_100v_32r_256/checkpoints/epoch\=37-step\=22400.ckpt'
# srun -u python -m src.main +experiment=scannet/fvt +output_dir=train_fvt_full_10v_consec_32_256_new dataset.view_sampler.num_context_views=10 dataset.view_sampler.num_target_views=10 checkpointing.load='outputs/train_fvt_full_10v_consec_32_256_new/checkpoints/epoch\=39-step\=23600.ckpt'

# srun -u python -m src.main +experiment=scannet/fvt +output_dir=test_scannet_fvt mode=test dataset/view_sampler=bounded checkpointing.load='outputs/train_fvt_full_100v_32r_256/checkpoints/epoch\=47-step\=28200.ckpt'
srun -u python -m src.main +experiment=scannet/fvt +output_dir=test_scannet_fvt mode=test dataset/view_sampler=bounded checkpointing.load='outputs/train_fvt_full_100v_32r_256/checkpoints/epoch\=75-step\=44700.ckpt'


# srun -u python -m src.main +experiment=scannet/fvt +output_dir=train_fvt_full_100v_32r_256_new
# srun -u python -m src.main +experiment=scannet/fvt +output_dir=train_fvt_full_10v_consec_32_256_new dataset.view_sampler.num_context_views=10 dataset.view_sampler.num_target_views=10

# srun -u python -m src.main +experiment=scannet/fvt +output_dir=test_scannet_fvt_10v mode=test dataset/view_sampler=bounded dataset.view_sampler.num_context_views=10 dataset.view_sampler.num_target_views=10 checkpointing.load='outputs/train_fvt_full_10v_consec_32_256_new/checkpoints/epoch\=91-step\=54300.ckpt'


