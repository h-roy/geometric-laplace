#!/bin/bash
#BSUB -q gpuv100
#BSUB -J test_segmentation_error
#BSUB -n 4
#BSUB -gpu "num=1"
#BSUB -W 0:05
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "select[gpu16gb]"
#BSUB -R "span[hosts=1]"
#BSUB -o /work3/hroy/geometric-laplace/logs/jobscript_test_0.out
#BSUB -e /work3/hroy/geometric-laplace/logs/jobscript_test_0.err
module load python3/3.9.11 cuda/11.4 cudnn/v8.6.0.163-prod-cuda-11.X
module swap cudnn/v8.6.0.163-prod-cuda-11.X
source geom/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python3 src/training/train_fc.py