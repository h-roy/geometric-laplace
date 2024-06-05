#!/bin/bash
#BSUB -J resnet_proj_sample
#BSUB -q p1
#BSUB -R "select[gpu80gb]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 20:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err


module load python3/3.11.4 cuda/12.2 cudnn/v8.9.1.23-prod-cuda-12.X
source geom/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# python src/sampling/sample_proj.py --num_samples 5 --posthoc_precision 2.0 --run_name "prec2"
python src/sampling/sample_resnet.py --diffusion_steps 10 --num_samples 5 --lanczos_iters 1000 --posthoc_precision 1.0 --posterior_type "non-kernel-eigvals" --run_name "posterior_1000iters_prec1"
