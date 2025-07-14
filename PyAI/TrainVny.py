from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# Load model & tokenizer
model_name = "./models/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",    # force CPU
    # no load_in_4bit, load full precision (fp32)
)

# Prepare model for LoRA - note this function expects kbit (quantized) model, but we skip quantization
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # adjust if needed
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Load your dataset
dataset = load_dataset("json", data_files={"train": r"C:\Users\Domin\Downloads\ComputerScience\Programs\AI-proto\trainingData\vinny_personality.jsonl"})

# Tokenize prompt-completion pairs
def format_and_tokenize(example):
    text = example["prompt"] + example["completion"]
    tokenized = tokenizer(text, truncation=True, padding="max_length", max_length=512)
    
    labels = tokenized["input_ids"].copy()
    labels = [label if label != tokenizer.pad_token_id else -100 for label in labels]
    tokenized["labels"] = labels
    
    return tokenized

# Map tokenization to dataset
tokenized_dataset = dataset.map(format_and_tokenize, batched=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./vinny-lora-results",
    per_device_train_batch_size=1,   # reduce batch size for CPU RAM limits
    num_train_epochs=3,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
    learning_rate=2e-4,
    fp16=False,                     # disable fp16 for CPU
    report_to="none"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# Train model
trainer.train()

# Save LoRA adapter weights
model.save_pretrained("./vinny-lora-adapter")
tokenizer.save_pretrained("./vinny-lora-adapter")
