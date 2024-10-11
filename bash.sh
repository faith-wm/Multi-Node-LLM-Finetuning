#!/bin/bash
#SBATCH --job-name=finetuning  
#SBATCH --nodes=2     #adjust to your number of nodes
#SBATCH --ntasks-per-node=1  # crucial - only 1 task per node!
#SBATCH --cpus-per-task=50
#SBATCH --mem=980G     #Adjust memory per node here 
#SBATCH --partition=gpu-xe9680q
#SBATCH --gres=gpu:8        # Adjust number of GPUs per node here
#SBATCH --output=%x-%j.out

set -x -e

# Load required modules
# module load gcccuda OpenMPI/4.0.5-gcccuda-2020b cuda11.8/toolkit/11.8.0  (uncomment if needed)

# NCCL debugging options (uncomment if needed)
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1  
# export NCCL_P2P_DISABLE=1  
# export NCCL_ASYNC_ERROR_HANDLING=1  
# export CUDA_LAUNCH_BLOCKING=1

# Host management
export PARENT=$(hostname -s)
export CHILDREN=$(scontrol show hostnames $SLURM_JOB_NODELIST | grep -v $PARENT)
export HOSTLIST="$PARENT $CHILDREN"


# Set MASTER_ADDR and MASTER_PORT explicitly
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=1234 #add your masterport here

# Print for debugging
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

# Generate the hostfile for DeepSpeed
makehostfile() {
  perl -e '
    $slots = $ENV{"SLURM_STEP_GPUS"} ? scalar(split /,/, $ENV{"SLURM_STEP_GPUS"}) : 8;
    @nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
    foreach my $node (@nodes) {
      print "$node slots=$slots\n";
    }
  '
}
makehostfile > hostfile

# Check if hostfile is generated correctly
if [ ! -s hostfile ]; then
  echo "Error: Hostfile is empty or not formatted correctly, unable to proceed with launching."
  exit 1
fi

echo "Hostfile generated successfully:"
cat hostfile

# Launch DeepSpeed with OpenMPI
deepspeed -H hostfile \
  --master_port $MASTER_PORT \
  --master_addr $MASTER_ADDR \
  --no_ssh_check \
  --launcher OPENMPI \
  --launcher_args="--oversubscribe" \
  train.py -m $PARENT
