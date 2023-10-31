## Evaluation Instruction for MiniGPT-v2

### Data preparation



### environment setup

```
export PYTHONPATH=$PYTHONPATH:/path/to/directory/of/MiniGPT-4
```

### start evalauting RefCOCO, RefCOCO+, RefCOCOg
port=port_number
cfg_path=/path/to/eval_configs/minigptv2_eval.yaml
eval_file_path=/path/to/eval/image/path
save_path=/path/to/save/path
ckpt=/path/to/evaluation/checkpoint


split=/evaluation/data/split/type  # e.g. val, testA, testB, test
dataset=/data/type  #refcoco, refcoco+, refcocog

```
torchrun --master-port ${port} --nproc_per_node 1 eval_ref.py \
 --cfg-path ${cfg_path} --img_path ${IMG_PATH} --eval_file_path ${eval_file_path} --save_path ${save_path} \
 --ckpt ${ckpt} --split ${split}  --dataset ${dataset} --lora_r 64 --lora_alpha 16 \
 --batch_size 10 --max_new_tokens 20 --resample
```


### start evaluating visual question answering

port=port_number
cfg_path=/path/to/eval_configs/minigptv2_eval.yaml
eval_file_path=/path/to/eval/image/path
save_path=/path/to/save/path
ckpt=/path/to/evaluation/checkpoint


split=/evaluation/data/split/type  # e.g. val,test
dataset=/data/type  # vqa data types: okvqa, vizwiz, iconvqa, gqa, vsr, hm

```
torchrun --master-port ${port} --nproc_per_node 1 eval_ref.py \
 --cfg-path ${cfg_path} --img_path ${IMG_PATH} --eval_file_path ${eval_file_path} --save_path ${save_path} \
 --ckpt ${ckpt} --split ${split}  --dataset ${dataset} --lora_r 64 --lora_alpha 16 \
 --batch_size 10 --max_new_tokens 20 --resample
```




