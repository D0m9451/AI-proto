from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import time


model_path = "./models/Qwen2.5-3B"
adapter_path = "./vinny-lora-adapter"

# Load tokenizer from adapter folder (if saved there)
tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

# Load base model WITHOUT quantization, force to CPU
base_model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    trust_remote_code=True, 
    device_map="cpu"   # <-- CPU only
)

# Attach LoRA adapter to base model
model = PeftModel.from_pretrained(base_model, adapter_path)

# Optional: Move model to CPU explicitly
model = model.to("cpu")

while True:
    prompt = input("User: ")
    start = time.time()
    # Tokenize prompt and move tensors to CPU
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    # Generate response
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
    print(f"Vinny 2.0: {model_reply}")
