#!/bin/bash
#BSUB -q gpuh100
#BSUB -J run_diffusion
#BSUB -n 4
#BSUB -gpu "num=1"
#BSUB -W 4:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -R "select[gpu16gb]"
#BSUB -R "span[hosts=1]"
#BSUB -o /work3/hroy/geometric-laplace/logs/jobscript_test_0.out
#BSUB -e /work3/hroy/geometric-laplace/logs/jobscript_test_0.err
module load python3/3.9.11 cuda/12.2 cudnn/v8.9.1.23-prod-cuda-12.X
source geom/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python3 experiments/sine_curve.py