#!/bin/bash
#BSUB -J really_big_sample
#BSUB -q p1
#BSUB -R "select[gpu80gb]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err


module load python3/3.11.4 cuda/12.2 cudnn/v8.9.1.23-prod-cuda-12.X
source geom/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# python3 src/sampling/sample.py --checkpoint_path "./checkpoints/MNIST/LeNet/OOD_MNIST_seed420" --run_name "sample_MNIST" --diffusion_steps 20 --num_samples 5 --lanczos_iters 1000
python3 src/sampling/sample.py --checkpoint_path "./checkpoints/FMNIST/LeNet/OOD_FMNIST_seed420" --run_name "alpha_5" --diffusion_steps 25 --num_samples 5 --lanczos_iters 1000 --posthoc_precision 5.0
python3 src/sampling/sample.py --checkpoint_path "./checkpoints/FMNIST/LeNet/OOD_FMNIST_seed420" --run_name "alpha_05" --diffusion_steps 25 --num_samples 5 --lanczos_iters 1000 --posthoc_precision 0.5
python3 src/sampling/sample.py --checkpoint_path "./checkpoints/CIFAR-10/LeNet/OOD_CIFAR_seed420" --run_name "sample_CIFAR" --diffusion_steps 20 --num_samples 5 --lanczos_iters 1000