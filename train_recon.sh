#!/bin/bash

set -x

CUDA_VISIBLE_DEVICES=0 python src/train_reconstruction.py \
    --resolution 32 \
    --data_path "../data/FashionMNIST" \
    --exp_dir "results/PG_Fashion_MNIST_Reconstructor_UNet" \
    --accelerations 32 16 10.6667 8 6.4 5.3333 \
    --center_fractions 0.0625 0.0625 0.0625 0.0625 0.0625 0.0625
    

# CUDA_VISIBLE_DEVICES=0 python src/train_reconstruction.py \
#     --data_path "/scratch/imaging/projects/active_acquisition/data/fastMRI" \
#     --exp_dir "results/PG_Fashion_MNIST_Reconstructor"
