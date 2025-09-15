from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch
import os

# === Config === #
model_name = "./models/Qwen2.5-3B"
dataset_path = r"C:\Users\Domin\Downloads\Programminn\repo\AI-proto\trainingData\vinnypersonality(10K).jsonl"
dataset_format = "jsonl"  # csv, json, jsonl
dataset_columns = ["prompt", "completion"]  # adjust this depending on your dataset
delimiter = ","  # for CSV only

output_dir = "./vinny-lora-results"
num_epochs = 1
batch_size = 2
learning_rate = 2e-4
max_length = 512

# === Load Model & Tokenizer === #
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
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
if dataset_format == "csv":
    dataset = load_dataset(
        "csv",
        data_files={"train": dataset_path},
        delimiter=delimiter
    )
elif dataset_format == "json":
    dataset = load_dataset("json", data_files={"train": dataset_path})
elif dataset_format == "jsonl":
    dataset = load_dataset("json", data_files={"train": dataset_path})

# === Tokenization Function === #
def format_and_tokenize(example):
    text = example[dataset_columns[0]] + " " + example[dataset_columns[1]]
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=max_length)
    labels = tokenized["input_ids"].copy()
    labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(format_and_tokenize, batched=False)

# === Training Arguments === #
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    num_train_epochs=num_epochs,
    save_steps=50,                  # Save every 50 steps (adjust if needed)
    save_total_limit=3,              # Keep last 3 checkpoints
    logging_steps=5,
    learning_rate=learning_rate,
    fp16=False,
    report_to="none"
)

# === Trainer Setup === #
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# === Check for Existing Checkpoint === #
last_checkpoint = None
if os.path.isdir(output_dir):
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if checkpoints:
        # Get latest checkpoint by highest step number
        last_checkpoint = os.path.join(output_dir, sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1])
        print(f"Resuming from checkpoint: {last_checkpoint}")

# === Train === #
trainer.train(resume_from_checkpoint=last_checkpoint)

# === Save Adapter === #
model.save_pretrained(output_dir + "/adapter")
tokenizer.save_pretrained(output_dir + "/adapter")
