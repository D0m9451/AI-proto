from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
start = time.time()


model_path = "./models/Qwen2.5-3B"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto")


while True:
    prompt = input("User: ")
    inputs = tokenizer(prompt, return_tensors="pt")
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

    model_reply = response[len(prompt):].strip()

    model_reply = model_reply.split("\n")[0]

    print(f"Generated in {time.time() - start:.2f} seconds")
    print(f"Vinny 1.0: {model_reply}")