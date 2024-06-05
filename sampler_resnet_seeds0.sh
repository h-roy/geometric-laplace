#!/bin/bash
#BSUB -J try_again
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

python src/sampling/sample_resnet.py --diffusion_steps 2 --num_samples 1 --lanczos_iters 3500 --posthoc_precision 1.5 --posterior_type "non-kernel-eigvals" --run_name "newnew_1500_1.5_3" --sample_seed 0 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed0.pickle"
python src/sampling/sample_resnet.py --diffusion_steps 2 --num_samples 1 --lanczos_iters 3500 --posthoc_precision 1.5 --posterior_type "non-kernel-eigvals" --run_name "newnew_1500_1.5_3" --sample_seed 1 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed1.pickle"
# python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 15 --lanczos_iters 3500 --posthoc_precision 1.5 --posterior_type "non-kernel-eigvals" --run_name "new_3500_1.5_15" --sample_seed 2 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed2.pickle"
# python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 15 --lanczos_iters 3500 --posthoc_precision 1.5 --posterior_type "non-kernel-eigvals" --run_name "new_3500_1.5_15" --sample_seed 3 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed3.pickle"
# python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 15 --lanczos_iters 3500 --posthoc_precision 1.5 --posterior_type "non-kernel-eigvals" --run_name "new_3500_1.5_15" --sample_seed 4 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed4.pickle"


# python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 1 --lanczos_iters 1500 --posthoc_precision 10.0 --posterior_type "non-kernel-eigvals" --run_name "_1500_10" --sample_seed 0 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed0.pickle"
# python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 1 --lanczos_iters 1000 --posthoc_precision 10.0 --posterior_type "non-kernel-eigvals" --run_name "_1000_10" --sample_seed 0 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed0.pickle"
# python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 1 --lanczos_iters 500 --posthoc_precision 10.0 --posterior_type "non-kernel-eigvals" --run_name "_500_10" --sample_seed 0 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed0.pickle"

# python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 5 --lanczos_iters 1000 --posthoc_precision 5.0 --posterior_type "non-kernel-eigvals" --run_name "seed1" --sample_seed 1 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed1.pickle"
# python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 5 --lanczos_iters 1000 --posthoc_precision 5.0 --posterior_type "non-kernel-eigvals" --run_name "seed2" --sample_seed 2 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed2.pickle"
# python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 5 --lanczos_iters 1000 --posthoc_precision 5.0 --posterior_type "non-kernel-eigvals" --run_name "seed3" --sample_seed 3 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed3.pickle"
# python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 5 --lanczos_iters 1000 --posthoc_precision 5.0 --posterior_type "non-kernel-eigvals" --run_name "seed4" --sample_seed 4 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed4.pickle"

# python src/sampling/sample_resnet.py --diffusion_steps 5 --num_samples 5 --lanczos_iters 2500 --posthoc_precision 5.0 --posterior_type "non-kernel-eigvals" --run_name "seed0" --sample_seed 0 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed0.pickle"
# python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 5 --lanczos_iters 1500 --posthoc_precision 0.5 --posterior_type "non-kernel-eigvals" --run_name "seed0" --sample_seed 0 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed0.pickle"
# python src/sampling/sample_resnet.py --diffusion_steps 3 --num_samples 5 --lanczos_iters 2500 --posthoc_precision 0.5 --posterior_type "non-kernel-eigvals" --run_name "seed0" --sample_seed 0 --checkpoint_path "./checkpoints/CIFAR-10/ResNet/good_params_seed0.pickle"
