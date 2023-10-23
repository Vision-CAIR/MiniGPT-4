
cd ..

job_name=minigpt4_v2_test
read LOWERPORT UPPERPORT < /proc/sys/net/ipv4/ip_local_port_range
while :
do
        PORT="`shuf -i $LOWERPORT-$UPPERPORT -n 1`"
        ss -lpn | grep -q ":$PORT " || break
done


#torchrun --master-port ${PORT} --nproc-per-node 2 train.py --job_name ${job_name}  --cfg-path train_configs_llama2/336_final_v1_gqa.yaml


#torchrun --master-port ${PORT} --nproc-per-node 2 train.py --job_name ${job_name}  --cfg-path train_configs_llama2/448_final_v1_gqa_ablation2.yaml
torchrun --master-port ${PORT} --nproc-per-node 2 train.py --job_name ${job_name} --cfg-path train_configs/minigpt_v2_finetune.yaml

#torchrun --master-port ${PORT} --nproc-per-node 2 train.py --job_name ${job_name}  --cfg-path finetune_conversation_ablation/conversation_v2_last_336_test.yaml

#torchrun --master-port ${PORT} --nproc-per-node 2 train.py --job_name ${job_name}  --cfg-path train_configs_llama2/336_final_v1_13B.yaml

# torchrun --master-port ${PORT} --nproc-per-node 2 train.py --job_name ${job_name}  --cfg-path train_configs_final_ablations/448_v2_llama2.yaml
#accelerate launch train.py --job_name ${job_name}  --cfg-path train_configs_final_ablations/336_v2_llama2.yaml


# torchrun --master-port ${PORT} --nproc-per-node 2 train.py --job_name ${job_name}  --cfg-path train_configs_final_ablations/336_v2_llama2_clip_encoder.yaml

#best_data_ratio_336_full_dataset_lr2e4_v1.yaml

