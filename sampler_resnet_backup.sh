#!/bin/bash
#BSUB -J resnet_backup_sample
#BSUB -q p1
#BSUB -R "select[gpu80gb]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=2:mode=exclusive_process"
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err


module load python3/3.11.4 cuda/12.2 cudnn/v8.9.1.23-prod-cuda-12.X
source geom/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 2 --lanczos_iters 5000 --posterior_type "non-kernel-eigvals" --posthoc_precision 0.5 --run_name "new0.5_5000_1" --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed0.pickle" --sample_seed 0
python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 2 --lanczos_iters 5000 --posterior_type "non-kernel-eigvals" --posthoc_precision 0.5 --run_name "new0.5_5000_2" --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed1.pickle" --sample_seed 1
# python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 1 --lanczos_iters 5000 --posterior_type "non-kernel-eigvals" --posthoc_precision 0.5 --run_name "pray3" --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed2.pickle" --sample_seed 2

# python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 3 --lanczos_iters 2500 --posterior_type "non-kernel-eigvals" --posthoc_precision 1.0 --run_name "deep_resnet_prec05"

# python src/sampling/sample_resnet.py --diffusion_steps 2 --num_samples 5 --lanczos_iters 1500 --posterior_type "full-ggn" --run_name "posterior_full_ggn"
