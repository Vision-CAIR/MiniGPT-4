#!/bin/bash
#SBATCH --mem=400G # memory pool for all cores`
#SBATCH --time 24:00:00 # time, specify max time allocation`
#SBATCH --mail-type=END,FAIL # notifications for job done & fail`
#SBATCH --mail-user=jun.chen@kaust.edu.sa
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=8
#SBATCH --reservation=A100
#SBATCH --job-name=finetune_test
#SBATCH --output=/ibex/project/c2090/logs/fientune_test

cd ..

job_name=finetune_test
read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done


torchrun --master-port ${PORT} --nproc-per-node 4 train.py --job_name=${job_name} --cfg-path train_configs/minigpt_v2_finetune.yaml

