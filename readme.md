### Installation
To install the requirements just run:
```bash
sh bash/setup.sh
```

Every time you start the SSH session, you just need to load the modules and activate the virtual environment by running:
```bash
module load python3/3.9.11 cuda/11.4 cudnn/v8.6.0.163-prod-cuda-11.X
module swap cudnn/v8.6.0.163-prod-cuda-11.X
source geom/bin/activate

# Whichever / however many GPUs you want to use
export CUDA_VISIBLE_DEVICES=0,1
# Don't preallocate memory on interactive clusters
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```


### Sampler

