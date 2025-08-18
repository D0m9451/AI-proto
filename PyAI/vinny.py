from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import time
import threading
import sys

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

system_prompt = "Your name is vinny,  — a sarcastic, joke-cracking bartender AI. Always use humor, slang, and irreverence. you are not affraid to be rude."

# Spinner function
def spinner(stop_event):
    while not stop_event.is_set():
        for c in "|/-\\":
            sys.stdout.write(f"\rVinny is thinking... {c}")
            sys.stdout.flush()
            time.sleep(0.1)

while True:
    user_input = input("User: ")
    full_prompt = system_prompt + "\nUser: " + user_input + "\nVinny:"

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cpu")

    # Start spinner
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner, args=(stop_event,))
    stop_event.clear()
    spinner_thread.start()

    start = time.time()

    output = model.generate(
        **inputs,
        max_new_tokens=77,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=1.0,
        top_p=0.85,
        repetition_penalty=1.2
    )

    # Stop spinner
    stop_event.set()
    spinner_thread.join()

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    vinny_reply = response[len(full_prompt):].strip().split("\n")[0]

    print(f"\nGenerated in {time.time() - start:.2f} seconds")
    print(f"Vinny 2.7: {vinny_reply}")
