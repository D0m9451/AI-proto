from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import time

start = time.time()

model_path = "./models/Qwen2.5-3B"
adapter_path = "./vinny-lora-adapter"

tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

# Load base model and LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto", load_in_4bit=True)
model = PeftModel.from_pretrained(base_model, adapter_path)

while True:
    prompt = input("User: ")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    output = model.generate(
        **inputs,
        max_new_tokens=50,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    model_reply = response[len(prompt):].strip().split("\n")[0]

    print(f"Generated in {time.time() - start:.2f} seconds")
    print(f"Vinny 1.0: {model_reply}")
