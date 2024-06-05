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

# python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 5 --lanczos_iters 1500 --posthoc_precision 1.0 --posterior_type "non-kernel-eigvals" --run_name "seed4" --sample_seed 4 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed4.pickle"
# python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 5 --lanczos_iters 2500 --posthoc_precision 1.0 --posterior_type "non-kernel-eigvals" --run_name "seed4" --sample_seed 4 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed4.pickle"
# python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 5 --lanczos_iters 1500 --posthoc_precision 0.5 --posterior_type "non-kernel-eigvals" --run_name "seed4" --sample_seed 4 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed4.pickle"
python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 5 --lanczos_iters 2500 --posthoc_precision 2.5 --posterior_type "non-kernel-eigvals" --run_name "seed4" --sample_seed 4 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed4.pickle"
