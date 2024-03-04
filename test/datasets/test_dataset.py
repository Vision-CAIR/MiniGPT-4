import datasets
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import random
from tqdm import tqdm

# path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/multimodal/PromptMoE/minigpt4/models/cmrc2018_trial.json"
# dataset = load_dataset("json", data_files=[path], field="data",  split="train")
# tokenizer = AutoTokenizer.from_pretrained("/mnt/pfs-guan-ssai/nlu/wanghanzi/models/bert-base-uncased")
# def preprocess_function(example):
#     import pdb; pdb.set_trace()
#     model_inputs = tokenizer(example["content"], max_length=512, truncation=True)
#     labels = tokenizer(example["title"], max_length=32, truncation=True)
#     # label就是title编码的结果
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs
# processed_datasets = dataset.map(preprocess_function)

dataset = load_dataset("/mnt/pfs-guan-ssai/nlu/wanghanzi/data/alpaca_20k")
train_dataset = dataset['train']

    
for i in tqdm(range(1, len(train_dataset))):
    import pdb; pdb.set_trace()

    idx = random.randint(0,i)
    memory = train_dataset[idx]
    memory_text = f"Instruction: {memory['instruction']}\n Answer: {memory['output']} \n"
    train_dataset[i]['text'] = f"{memory_text} Instruction:{train_dataset[i]['instruction']}"


import pdb; pdb.set_trace()


model_path = "/mnt/pfs-guan-ssai/nlu/wanghanzi/models/opt_350m"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


def formatting_prompts_func(example):
    import pdb; pdb.set_trace()
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['output'][i]}"
        output_texts.append(text)
    return output_texts

response_template = " ### Answer:"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator,
)
trainer.train()