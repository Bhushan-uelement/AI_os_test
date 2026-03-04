import requests
import subprocess
import os

# Configuration
SLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
LLM_PATH = "/root/qwen-2b.gguf"  # Ensure you have this file

def ask_slm(query):
    """Instant response for chat and simple stats."""
    payload = {
        "messages": [{"role": "system", "content": "You are MerlinOS, a fast assistant."},
                    {"role": "user", "content": query}],
        "max_tokens": 150
    }
    try:
        r = requests.post(SLM_URL, json=payload)
        return r.json()['choices'][0]['message']['content']
    except:
        return "SLM Server is offline."

def analyze_with_llm(query, context):
    """Deep analysis using the larger 2B model."""
    print("[Routing to Qwen-2B for Deep Analysis...]")
    prompt = f"<|im_start|>system\nYou are the MerlinOS Log Expert. Analyze this data:\n{context}<|im_end|>\n<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"
    
    # We call llama-cli directly for the 2B model to save RAM (it closes after finishing)
    cmd = f'llama-cli -m {LLM_PATH} -p "{prompt}" -n 256 --quiet'
    return subprocess.check_output(cmd, shell=True, text=True)

print("--- MerlinOS Dual-Engine Active ---")

while True:
    user_input = input("\nUser @ MerlinOS: ")
    
    # 1. ROUTING LOGIC
    if any(word in user_input.lower() for word in ["analyze", "log", "debug", "error"]):
        # Fetch the logs to give to the 2B model
        logs = subprocess.check_output("journalctl -n 20 --no-pager", shell=True, text=True)
        response = analyze_with_llm(user_input, logs)
    else:
        # Fast path
        response = ask_chat(user_input)

    print(f"\nMerlin AI: {response}")
