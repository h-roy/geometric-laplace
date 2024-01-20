#!/bin/bash
#BSUB -J big_train
#BSUB -q gpuv100
#BSUB -R "select[gpu32gb]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1"
#BSUB -n 8
#BSUB -W 24:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

module load python3/3.9.11 cuda/11.4 cudnn/v8.6.0.163-prod-cuda-11.X
module swap cudnn/v8.6.0.163-prod-cuda-11.X
source geom/bin/activate
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

datasets=('MNIST' 'FMNIST' 'CIFAR-10' 'SVHN' 'CIFAR-100')
models=('LeNet' 'MLP')

for seed in {0..4}
do
    for dataset in "${datasets[@]}"
    do
        for model in "${models[@]}"
        do 
            echo "Running $dataset with $model (seed $seed)"
            if (($model=='MLP'))
            then
                python src/training/train_model.py --dataset $dataset --likelihood "classification" --model $model --mlp_hidden_dim 10 --mlp_num_layers 1 --seed $seed --n_epochs 50 --run_name epoch50
                python src/training/train_model.py --dataset $dataset --likelihood "classification" --model $model --mlp_hidden_dim 20 --mlp_num_layers 2 --seed $seed --n_epochs 50 --run_name epoch50
                python src/training/train_model.py --dataset $dataset --likelihood "classification" --model $model --mlp_hidden_dim 50 --mlp_num_layers 2 --seed $seed --n_epochs 50 --run_name epoch50
                python src/training/train_model.py --dataset $dataset --likelihood "classification" --model $model --mlp_hidden_dim 300 --mlp_num_layers 2 --seed $seed --n_epochs 50 --run_name epoch50
            else
                python src/training/train_model.py --dataset $dataset --likelihood "classification" --model $model --seed $seed --n_epochs 50 --run_name epoch50
            fi
        done
    done
done