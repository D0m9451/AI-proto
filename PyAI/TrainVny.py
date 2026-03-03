from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os

# === Config === #
modelname = "./models/Qwen2.5-3B"
datasetpath = r"C:\Users\Domin\Downloads\Programminn\repo\AI-proto\trainingData\QAdataset44k.csv"
datasetformat = "csv"  # csv, json, jsonl
datasetcolumns = ["prompt", "completion"]  
delimiter = ","  # for CSV only

outputdir = "./vinny-lora-results"
num_epochs = 1
batch_size = 2
learning_rate = 2e-4
max_length = 512

# === Load Model & Tokenizer === #
tokenizer = AutoTokenizer.from_pretrained(modelname, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    modelname,
    device_map="cpu"
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# === Load Dataset === #
if datasetformat == "csv":
    dataset = load_dataset(
        "csv",
        data_files={"train": datasetpath},
        delimiter=delimiter
    )
elif datasetformat == "json":
    dataset = load_dataset("json", data_files={"train": datasetpath})
elif datasetformat == "jsonl":
    dataset = load_dataset("json", data_files={"train": datasetpath})

# === Tokenization Function === #
def format_and_tokenize(example):
    text = example[datasetcolumns[0]] + " " + example[datasetcolumns[1]]
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=max_length)
    labels = tokenized["input_ids"].copy()
    labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(format_and_tokenize, batched=False)

# === Training Arguments === #
training_args = TrainingArguments(
    output_dir=outputdir,
    per_device_train_batch_size=batch_size,
    num_train_epochs=num_epochs,
    save_steps=50,                  
    save_total_limit=3,              
    logging_steps=5,
    learning_rate=learning_rate,
    fp16=False,
    report_to="none",                
)

# === Trainer Setup === #
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
)

# === Check for Existing Checkpoint === #
last_checkpoint = None
if os.path.isdir(outputdir):
    checkpoints = [d for d in os.listdir(outputdir) if d.startswith("checkpoint-")]
    if checkpoints:
        last_checkpoint = os.path.join(outputdir, sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1])
        print(f"Resuming from checkpoint: {last_checkpoint}")

# === Train === #
trainer.train(resume_from_checkpoint=last_checkpoint)

# === Save Adapter === #
model.save_pretrained(outputdir + "/adapter")
tokenizer.save_pretrained(outputdir + "/adapter")
