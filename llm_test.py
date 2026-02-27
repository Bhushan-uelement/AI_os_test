import subprocess
import json
from llama_cpp import Llama

# 1. Initialize the AI with System Context
# We tell the model EXACTLY what MerlinOS is using Qwen's system tag format.
SYSTEM_PROMPT = """You are MerlinOS AI, the central intelligence for this Linux distribution.
System Details:
- OS: MerlinOS (Yocto-based Enterprise AI Distro)
- Hostname: merlinos
- Purpose: Edge AI system and system management.

You have access to terminal tools. If a user asks about system status, logs, or 
hardware, you MUST use the provided context to answer concisely."""

# Load the model - verbose=False keeps the terminal clean of loading logs
llm = Llama(model_path="/root/qwen.gguf", n_ctx=2048, verbose=False)

def get_system_data(command_type):
    """Executes safe diagnostic commands to provide live context."""
    try:
        safe_commands = {
            "logs": "journalctl -n 20 --no-pager",
            "storage": "df -h",
            "ram": "free -h",
            "status": "hostnamectl",
            "network": "ip addr"
        }
        cmd = safe_commands.get(command_type, "uptime")
        output = subprocess.check_output(cmd, shell=True).decode()
        return output
    except Exception as e:
        return f"Error fetching system data: {str(e)}"

print("--- MerlinOS Intelligence Agent Online ---")

while True:
    user_query = input("\nUser @ MerlinOS: ")
    if user_query.lower() in ["exit", "quit"]:
        break

    # 2. Fetch Context if needed
    context_data = ""
    if any(word in user_query.lower() for word in ["log", "error", "ram", "storage", "who", "stat", "cpu"]):
        print("[Retrieving MerlinOS System Context...]")
        # Determine best tool to use
        cmd_type = "status"
        if "log" in user_query.lower(): cmd_type = "logs"
        elif "ram" in user_query.lower(): cmd_type = "ram"
        elif "storage" in user_query.lower(): cmd_type = "storage"
        
        context_data = f"\nLIVE SYSTEM DATA:\n{get_system_data(cmd_type)}"

    # 3. Format using ChatML (Strictly required for Qwen 2.5)
    # This structure is what triggers the model to respond correctly.
    full_prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}\n{context_data}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # 4. Generate with Streaming
    print("\nMerlin AI: ", end="", flush=True)
    
    # We use stream=True so you see the text word-by-word
    response_stream = llm(
        full_prompt, 
        max_tokens=512, 
        stop=["<|im_end|>", "<|endoftext|>"], 
        stream=True
    )

    for chunk in response_stream:
        if "choices" in chunk:
            text = chunk["choices"][0]["text"]
            print(text, end="", flush=True)
    
    print() # New line after response
