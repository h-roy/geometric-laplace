module load python3/3.9.11 cuda/11.4 cudnn/v8.6.0.163-prod-cuda-11.X
python3 -m pip install --upgrade pip
python3 -m venv geom/
source ./geom/bin/activate
python3 -m pip install -e .
python3 -m pip install -U pip setuptools wheel
python3 -m pip install -r requirements.txt

python3 -m pip install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
python3 -m pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
cp -n .env.example .env
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
