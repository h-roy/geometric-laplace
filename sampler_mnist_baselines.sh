#!/bin/bash
#BSUB -J baselines_mnist
#BSUB -q p1
#BSUB -R "select[gpu80gb]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -W 18:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err


module load python3/3.11.4 cuda/12.2 cudnn/v8.9.1.23-prod-cuda-12.X
source geom/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python src/sampling/sample_lapalce_baselines_lenet.py --method Hutchinson_Diag --checkpoint_path "./checkpoints/MNIST/LeNet/seed_0_seed0"  --sample_seed 0 --run_name "baseline_0" --posthoc_precision 50.0
python src/sampling/sample_lapalce_baselines_lenet.py --method Hutchinson_Diag --checkpoint_path "./checkpoints/MNIST/LeNet/seed_1_seed1"  --sample_seed 1 --run_name "baseline_1" --posthoc_precision 50.0
python src/sampling/sample_lapalce_baselines_lenet.py --method Hutchinson_Diag --checkpoint_path "./checkpoints/MNIST/LeNet/seed_2_seed2"  --sample_seed 2 --run_name "baseline_2" --posthoc_precision 50.0


python src/sampling/sample_lapalce_baselines_lenet.py --method Hutchinson_Diag --checkpoint_path "./checkpoints/FMNIST/LeNet/seed_0_seed0"  --sample_seed 0 --run_name "baseline_0" --posthoc_precision 50.0
python src/sampling/sample_lapalce_baselines_lenet.py --method Hutchinson_Diag --checkpoint_path "./checkpoints/FMNIST/LeNet/seed_1_seed1"  --sample_seed 1 --run_name "baseline_1" --posthoc_precision 50.0
python src/sampling/sample_lapalce_baselines_lenet.py --method Hutchinson_Diag --checkpoint_path "./checkpoints/FMNIST/LeNet/seed_2_seed2"  --sample_seed 2 --run_name "baseline_2" --posthoc_precision 50.0


python src/sampling/sample_lapalce_baselines_lenet.py --method Subnetwork --checkpoint_path "./checkpoints/MNIST/LeNet/seed_0_seed0"  --sample_seed 0 --run_name "baseline_0" --posthoc_precision 1.0
python src/sampling/sample_lapalce_baselines_lenet.py --method Subnetwork --checkpoint_path "./checkpoints/MNIST/LeNet/seed_1_seed1"  --sample_seed 1 --run_name "baseline_1" --posthoc_precision 1.0
python src/sampling/sample_lapalce_baselines_lenet.py --method Subnetwork --checkpoint_path "./checkpoints/MNIST/LeNet/seed_2_seed2"  --sample_seed 2 --run_name "baseline_2" --posthoc_precision 1.0

python src/sampling/sample_lapalce_baselines_lenet.py --method Subnetwork --checkpoint_path "./checkpoints/FMNIST/LeNet/seed_0_seed0"  --sample_seed 0 --run_name "baseline_0" --posthoc_precision 1.0
python src/sampling/sample_lapalce_baselines_lenet.py --method Subnetwork --checkpoint_path "./checkpoints/FMNIST/LeNet/seed_1_seed1"  --sample_seed 1 --run_name "baseline_1" --posthoc_precision 1.0
python src/sampling/sample_lapalce_baselines_lenet.py --method Subnetwork --checkpoint_path "./checkpoints/FMNIST/LeNet/seed_2_seed2"  --sample_seed 2 --run_name "baseline_2" --posthoc_precision 1.0

python src/sampling/sample_lapalce_baselines_lenet.py --method SWAG --checkpoint_path "./checkpoints/MNIST/LeNet/seed_0_seed0"  --sample_seed 0 --run_name "baseline_0" --posthoc_precision 1.0
python src/sampling/sample_lapalce_baselines_lenet.py --method SWAG --checkpoint_path "./checkpoints/MNIST/LeNet/seed_1_seed1"  --sample_seed 1 --run_name "baseline_1" --posthoc_precision 1.0
python src/sampling/sample_lapalce_baselines_lenet.py --method SWAG --checkpoint_path "./checkpoints/MNIST/LeNet/seed_2_seed2"  --sample_seed 2 --run_name "baseline_2" --posthoc_precision 1.0

python src/sampling/sample_lapalce_baselines_lenet.py --method SWAG --checkpoint_path "./checkpoints/FMNIST/LeNet/seed_0_seed0"  --sample_seed 0 --run_name "baseline_0" --posthoc_precision 1.0
python src/sampling/sample_lapalce_baselines_lenet.py --method SWAG --checkpoint_path "./checkpoints/FMNIST/LeNet/seed_1_seed1"  --sample_seed 1 --run_name "baseline_1" --posthoc_precision 1.0
python src/sampling/sample_lapalce_baselines_lenet.py --method SWAG --checkpoint_path "./checkpoints/FMNIST/LeNet/seed_2_seed2"  --sample_seed 2 --run_name "baseline_2" --posthoc_precision 1.0

