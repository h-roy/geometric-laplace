#!/bin/bash
#BSUB -J really_big_sample
#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err


module load python3/3.9.11 cuda/12.2 cudnn/v8.9.1.23-prod-cuda-12.X
source geom/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python3 src/sampling/sample.py --checkpoint_path "./checkpoints/MNIST/LeNet/OOD_MNIST_seed420" --run_name "sample_MNIST" --diffusion_steps 20 --num_samples 5 --lanczos_iters 1000
python3 src/sampling/sample.py --checkpoint_path "./checkpoints/MNIST/LeNet/OOD_FMNIST_seed420" --run_name "sample_FMNIST" --diffusion_steps 20 --num_samples 5 --lanczos_iters 1000
python3 src/sampling/sample.py --checkpoint_path "./checkpoints/MNIST/LeNet/OOD_CIFAR_seed420" --run_name "sample_CIFAR" --diffusion_steps 20 --num_samples 5 --lanczos_iters 1000