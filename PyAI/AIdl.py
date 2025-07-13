from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-3B"

print(f"Downloading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

print("Saving model and tokenizer locally...")
tokenizer.save_pretrained("./models/qwen2.5-3b")
model.save_pretrained("./models/qwen2.5-3b")

print("Done! Model saved in ./models/qwen2.5-3b")
