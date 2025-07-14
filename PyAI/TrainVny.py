from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

model_path = "./models/Qwen2.5-3B"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

dataset = load_dataset(
    "json",
    data_files={"train": r"C:\Users\Domin\Downloads\Programminn\repo\AI-proto\trainingData\revision_techniques.jsonl"},
)

def preprocess_function(examples):
    full_texts = [p + c for p, c in zip(examples["prompt"], examples["completion"])]
    tokenized = tokenizer(full_texts, truncation=True, padding="max_length", max_length=512)
    labels = []
    for i, (prompt_text, _) in enumerate(zip(examples["prompt"], examples["completion"])):
        prompt_len = len(tokenizer(prompt_text)["input_ids"])
        label_ids = tokenized["input_ids"][i].copy()
        label_ids[:prompt_len] = [-100] * prompt_len
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["prompt", "completion"])

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
)

trainer.train()
