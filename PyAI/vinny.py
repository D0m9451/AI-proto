from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import os
import time
import threading
import sys


torch.set_num_threads(os.cpu_count())

modelpath = "./models/Qwen2.5-3B"
adapterpath = "./vinny-lora-adapter"

# Load tokenizer from adapter folder (if saved there)
tokenizer = AutoTokenizer.from_pretrained(adapterpath, trust_remote_code=True)

# Load base model WITHOUT quantization, force to CPU
basemodel = AutoModelForCausalLM.from_pretrained(
    modelpath, 
    trust_remote_code=True, 
    device_map="cpu"   # <-- CPU only
)

# Attach LoRA adapter to base model
model = PeftModel.from_pretrained(basemodel, adapterpath)

# Optional: Move model to CPU explicitly
model = model.to("cpu")

systemprompt = "Your name is vinny,  — a desktop assistant AI. You are helpful, creative, clever, and very friendly. Your responses should be in-depth and detailed."

# Spinner function
def spinner(stop_event):
    while not stop_event.is_set():
        for c in "|/-\\":
            sys.stdout.write(f"\rVinny is thinking... {c}")
            sys.stdout.flush()
            time.sleep(0.1)

while True:
    userinput = input("User: ")
    fullprompt = systemprompt + "\nUser: " + userinput + "\nVinny:"

    inputs = tokenizer(fullprompt, return_tensors="pt").to("cpu")

    # Start spinner
    stop_event = threading.Event()
    spinner_thread = threading.Thread(target=spinner, args=(stop_event,))
    stop_event.clear()
    spinner_thread.start()

    start = time.time()

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens = 77,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.eos_token_id,
            do_sample = True,
            temperature = 0.5,
            top_p = 0.85,
            repetition_penalty = 1.2,
            use_cache = True
        )

    # Stop spinner
    stop_event.set()
    spinner_thread.join()

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    vinnyreply = response[len(fullprompt):].strip().split("\n")[0]

    print(f"\nGenerated in {time.time() - start:.2f} seconds")
    print(f"Vinny 2.7: {vinnyreply}")
