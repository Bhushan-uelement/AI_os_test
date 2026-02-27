import subprocess
import json
from llama_cpp import Llama

# 1. Initialize the AI with System Context
# We tell the model EXACTLY what MerlinOS is.
SYSTEM_PROMPT = """
You are MerlinOS AI, the central intelligence for this Linux distribution.
System Details:
- OS: MerlinOS (Yocto-based Enterprise AI Distro)
- Hostname: merlinos
- Purpose: Edge ai system and system management.

You have access to terminal tools. If a user asks about system status, logs, or 
hardware, you MUST use the provided context to answer.
"""

llm = Llama(model_path="/root/qwen.gguf", n_ctx=2048)

def get_system_data(command):
    """Executes safe diagnostic commands to provide live context."""
    try:
        # Define allowed 'safe' commands for the AI
        safe_commands = {
            "logs": "journalctl -n 20 --no-pager",
            "storage": "df -h",
            "ram": "free -h",
            "status": "hostnamectl",
            "network": "ip addr"
        }
        cmd = safe_commands.get(command, "uptime")
        output = subprocess.check_output(cmd, shell=True).decode()
        return output
    except Exception as e:
        return str(e)

# 2. The Interactive Loop
print("--- MerlinOS Intelligence Agent Online ---")
while True:
    user_query = input("\nUser @ MerlinOS: ")
    
    # Identify if the query needs system context
    context = ""
    if any(word in user_query.lower() for word in ["log", "error", "ram", "storage", "who"]):
        print("[System Info Requested...]")
        context = f"\nLIVE SYSTEM DATA:\n{get_system_data('logs' if 'log' in user_query else 'status')}"

    # Generate Response
    full_prompt = f"{SYSTEM_PROMPT}\n{context}\nUser: {user_query}\nAssistant:"
    response = llm(full_prompt, max_tokens=256, stop=["User:"])
    
    print(f"\nMerlin AI: {response['choices'][0]['text'].strip()}")
