from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
from pathlib import Path
from threading import Thread
import torch
import os
import time
import sys
import socket  

thesocket = None  # Global socket handle (reserved for future use)

# Change working directory to the script's own folder so all relative paths (models/, adapters/) resolve correctly.
os.chdir(Path(__file__).parent)
torch.set_num_threads(os.cpu_count())# Allow PyTorch to use every available CPU core for inference.

modelpath = "./models/Qwen2.5-3B"   # Base model weights
adapterpath = "./vinny-lora-adapter"    # LoRA fine-tune adapter

# Load the tokenizer from the adapter directory (it may contain custom tokens added during fine-tuning).
tokenizer = AutoTokenizer.from_pretrained(adapterpath, trust_remote_code=True)


# Load the base model onto the CPU.  
basemodel = AutoModelForCausalLM.from_pretrained(
    modelpath, 
    trust_remote_code=True, 
    device_map="cpu"   
)

# Merge the LoRA adapter weights on top of the base model, then explicitly move the merged model to CPU
model = PeftModel.from_pretrained(basemodel, adapterpath)
model = model.to("cpu")

# This prefix is prepended to every conversation turn so the model stays in character as "Vinny" regardless of what the user asks.
systemprompt = "Your name is Vinny,  — a desktop assistant AI. You are helpful, creative, clever, and very friendly. Your responses should be in-depth and detailed."

def transmit(token):
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('localhost', 8080)) # Connect to localhost only
        client.sendall(token.encode('utf-8')) #Encode into UTF-8 bytes before sending
        client.close()
        print(f"[SENT: {repr(token)}]", end="", flush=True) # Log the sent token inline for debugging purposes

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('localhost', 8081)) # Connect to localhost only
        client.sendall(token.encode('utf-8')) #Encode into UTF-8 bytes before sending
        client.close()
        print(f"[SENT: {repr(token)}]", end="", flush=True)

    except Exception as e: # Log the error inline without interrupting the token stream
        print(f"[ERROR: {e}]", end="", flush=True)
        pass  

def listen():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 9091)) # Bind to localhost only
    server.listen(1) # Queue up to 1 pending connection
    print("Listening for user input on port 9091...")

    while True:
        conn, addr = server.accept() # Block until a client connects
        chunks = []
        while True:
            chunk = conn.recv(1024).decode('utf-8') #Read up to 1KB at a time
            if not chunk:
                break # Empty read signals the sender closed
            chunks.append(chunk)

        data = ''.join(chunks) # Reassemble all chunks into one string
      
        return data #return to loop
    



while True:
    userinput = listen() # Block until a message arrives
    print(f"\n{userinput}\n")  # Echo raw input to the local console
    fullprompt = systemprompt + "\nUser: " + userinput + "\nVinny:" # Construct the full prompt string
    inputs = tokenizer(fullprompt, return_tensors="pt").to("cpu") # Tokenize and move tensors to CPU.
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    start = time.time() # Timer for latency recording

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=100, # max limit of generated tokens
        do_sample=True, # Enable stochastic sampling
        eos_token_id=tokenizer.eos_token_id, # Stop token
        pad_token_id=tokenizer.eos_token_id, # Reuse EOS as padding token
        temperature=0.7, # Sampling temperature
        top_p=0.85, # Top-p sampling
        repetition_penalty=1.2, # Repetition penalty
        streamer=streamer, # Hook for streaming output
        use_cache=True # KV-cache for faster generation
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start() # Run model.generate() on a daemon thread so the main thread can consume tokens from the streamer as they are produced.

    print("Vinny:", end=" ", flush=True) 
    for token in streamer: # Blocks until each token is ready
        sys.stdout.write(token) # Print to local console
        sys.stdout.flush() 
        transmit(token) #Send to GUI and memory
    transmit("<<END>>")

    print()
    print(f"\nGenerated in {time.time() - start:.2f} seconds") #latency report