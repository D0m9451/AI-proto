from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# Load model & tokenizer
model_name = "./models/Qwen2.5-3B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    load_in_4bit=True  # memory efficient
)

# Prepare model for LoRA
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # or adjust for Qwen specifics if needed
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Load your dataset
dataset = load_dataset("json", data_files={"train": r"C:\Users\Domin\Downloads\Programminn\repo\AI-proto\trainingData\vinny_personality.jsonl"})

# Tokenize prompt-completion pairs
def format_and_tokenize(example):
    text = example["prompt"] + example["completion"]
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(format_and_tokenize, batched=False)

# Training arguments
training_args = TrainingArguments(
    output_dir="./vinny-lora-results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
    learning_rate=2e-4,
    fp16=True,
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
