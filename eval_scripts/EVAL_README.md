## Evaluation Instruction for MiniGPT-v2

### Data preparation
Images download
Image source | Download path
--- | :---:
OKVQA| <a href="https://drive.google.com/drive/folders/1jxIgAhtaLu_YqnZEl8Ym11f7LhX3nptN?usp=sharing">annotations</a> &nbsp;&nbsp;  <a href="http://images.cocodataset.org/zips/train2017.zip"> images</a>
gqa | <a href="https://drive.google.com/drive/folders/1-dF-cgFwstutS4qq2D9CFQTDS0UTmIft?usp=drive_link">annotations</a> &nbsp;&nbsp;  <a href="https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip">images</a> 
hateful meme |  <a href="https://github.com/faizanahemad/facebook-hateful-memes">images and annotations</a> 
iconqa |  <a href="https://iconqa.github.io/#download">images and annotation</a>
vizwiz |  <a href="https://vizwiz.org/tasks-and-datasets/vqa/">images and annotation</a>
RefCOCO | <a href="https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip"> annotations </a>
RefCOCO+ | <a href="https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip"> annotations </a>
RefCOCOg | <a href="https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip"> annotations </a>

### Evaluation dataset structure

```
${MINIGPTv2_EVALUATION_DATASET}
├── gqa
│   └── test_balanced_questions.json
│   ├── testdev_balanced_questions.json
│   ├── gqa_images
├── hateful_meme
│   └── hm_images
│   ├── dev.jsonl
├── iconvqa
│   └── iconvqa_images
│   ├── choose_text_val.json
├── vizwiz
│   └── vizwiz_images
│   ├── val.json
├── vsr
│   └── vsr_images
├── okvqa
│   ├── okvqa_test_split.json
│   ├── mscoco_val2014_annotations_clean.json
│   ├── OpenEnded_mscoco_val2014_questions_clean.json
├── refcoco
│   └── instances.json
│   ├── refs(google).p
│   ├── refs(unc).p
├── refcoco+
│   └── instances.json
│   ├── refs(unc).p
├── refercocog
│   └── instances.json
│   ├── refs(google).p
│   ├── refs(und).p
...
```


### environment setup

```
export PYTHONPATH=$PYTHONPATH:/path/to/directory/of/MiniGPT-4
```

### config file setup

Set **llama_model** to the path of LLaMA model.  
Set **ckpt** to the path of our pretrained model.  
Set **eval_file_path** to the path of the annotation files for each evaluation data.  
Set **img_path** to the img_path for each evaluation dataset.  
Set **save_path** to the save_path for evch evaluation dataset.    

in [minigpt4/eval_configs/minigptv2_benchmark_evaluation.yaml](../minigpt4/eval_configs/minigptv2_benchmark_evaluation.yaml) 




### start evalauting RefCOCO, RefCOCO+, RefCOCOg
port=port_number  
cfg_path=/path/to/eval_configs/minigptv2_benchmark_evaluation.yaml  

dataset names:  
| refcoco | refcoco+ | refcocog |
| ------- | -------- | -------- |

```
torchrun --master-port ${port} --nproc_per_node 1 eval_ref.py \
 --cfg-path ${cfg_path} --dataset refcoco,refcoco+,refcocog --resample
```


### start evaluating visual question answering

port=port_number  
cfg_path=/path/to/eval_configs/minigptv2_benchmark_evaluation.yaml 

dataset names:  
| okvqa | vizwiz | iconvqa | gqa | vsr | hm |
| ------- | -------- | -------- |-------- | -------- | -------- |


```
torchrun --master-port ${port} --nproc_per_node 1 eval_vqa.py \
 --cfg-path ${cfg_path} --dataset okvqa,vizwiz,iconvqa,gqa,vsr,hm
```




