#!/bin/bash -l 
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --reservation=A100 
#SBATCH --mem=32GB 
#SBATCH --time=4:00:00 
#SBATCH --partition=batch
##SBATCH --account=conf-iclr-2023.09.29-elhosemh

# Load environment which has Jupyter installed. It can be one of the following:
# - Machine Learning module installed on the system (module load machine_learning)
# - your own conda environment on Ibex
# - a singularity container with python environment (conda or otherwise)  

module load machine_learning 

# get tunneling info 
export XDG_RUNTIME_DIR="" node=$(hostname -s) 
user=$(whoami) 
submit_host=${SLURM_SUBMIT_HOST} 
port=10035
echo $node pinned to port $port 
# print tunneling instructions 

echo -e " 
To connect to the compute node ${node} on IBEX running your jupyter notebook server, you need to run following two commands in a terminal 1. 
Command to create ssh tunnel from you workstation/laptop to glogin: 

ssh -L ${port}:${node}:${port} ${user}@glogin.ibex.kaust.edu.sa 

Copy the link provided below by jupyter-server and replace the NODENAME with localhost before pasting it in your browser on your workstation/laptop " 

# Run Jupyter 
jupyter notebook --no-browser --port=${port} --port-retries=50 --ip=${node} 
