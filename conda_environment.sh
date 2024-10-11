module load gcccuda
module load OpenMPI/4.0.5-gcccuda-2020b
module load cuda11.8/toolkit

conda create -n train python=3.11
conda activate train

conda install openmpi intel-openmp
pip install mpi4py
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets evaluate sacrebleu scipy accelerate deepspeed fastobo sentencepiece protobuf
pip install flash-attn --no-build-isolation
