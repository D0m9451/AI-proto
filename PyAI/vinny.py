from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
from pathlib import Path
from threading import Thread
import torch
import os
import time
import sys
import socket  

thesocket = None  
'''
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', 8080))
    s.send(b"TEST")
    s.close()
    print("Connection worked!")
except Exception as e:
    print(f"Connection failed: {e}")
'''

os.chdir(Path(__file__).parent)
torch.set_num_threads(os.cpu_count())

modelpath = "./models/Qwen2.5-3B"
adapterpath = "./vinny-lora-adapter"


tokenizer = AutoTokenizer.from_pretrained(adapterpath, trust_remote_code=True)

basemodel = AutoModelForCausalLM.from_pretrained(
    modelpath, 
    trust_remote_code=True, 
    device_map="cpu"   
)


model = PeftModel.from_pretrained(basemodel, adapterpath)
model = model.to("cpu")

systemprompt = "Your name is Vinny,  — a desktop assistant AI. You are helpful, creative, clever, and very friendly. Your responses should be in-depth and detailed."

def transmit(token):
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('localhost', 8080)) 
        client.sendall(token.encode('utf-8'))
        client.close()
        print(f"[SENT: {repr(token)}]", end="", flush=True)

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(('localhost', 8081)) 
        client.sendall(token.encode('utf-8'))
        client.close()
        print(f"[SENT: {repr(token)}]", end="", flush=True)

    except Exception as e:
        print(f"[ERROR: {e}]", end="", flush=True)
        pass  

def listen():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('localhost', 9091)) 
    server.listen(1)
    print("Listening for user input on port 9091...")

    while True:
        conn, addr = server.accept()
        data = conn.recv(1024).decode('utf-8')
        return data



while True:
    userinput = listen()
    print(f"\n{userinput}\n") 
    fullprompt = systemprompt + "\nUser: " + userinput + "\nVinny:"
    inputs = tokenizer(fullprompt, return_tensors="pt").to("cpu")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    start = time.time()

    generation_kwargs = dict(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.85,
        repetition_penalty=1.2,
        streamer=streamer,
        use_cache=True
    )

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    print("Vinny:", end=" ", flush=True)
    for token in streamer:
        sys.stdout.write(token)
        sys.stdout.flush()
        transmit(token)
    transmit("<<END>>")

    print()
    print(f"\nGenerated in {time.time() - start:.2f} seconds")