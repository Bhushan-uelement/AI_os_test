import subprocess
import time
from llama_cpp import Llama

# 1. SETUP: Conservative Memory & Threading
# Using n_threads=4 for your IdeaPad; n_threads_batch=1 prevents sync hangs.
print("[SYSTEM] Initializing MerlinOS Engine...")
llm = Llama(
    model_path="/root/qwen.gguf",
    n_ctx=1024,         # Sufficient for OS context
    n_threads=4,        # Matches your CPU physical cores
    n_threads_batch=1,  # Critical: Prevents threading deadlocks
    verbose=False       # Keeps terminal clean
)

def fetch_os_context(query):
    """Bridge between AI and MerlinOS Kernel."""
    try:
        # Map keywords to safe OS commands
        if "log" in query.lower():
            return subprocess.check_output("journalctl -n 10 --no-pager", shell=True, text=True)
        elif "ram" in query.lower() or "stat" in query.lower():
            return subprocess.check_output("free -h && uptime", shell=True, text=True)
        return ""
    except Exception as e:
        return f"System Error: {str(e)}"

print("--- MerlinOS Intelligence Agent Online ---")

while True:
    user_query = input("\nUser @ MerlinOS: ")
    if user_query.lower() in ["exit", "quit"]: break

    # 2. CONTEXT INJECTION
    context = fetch_os_context(user_query)
    
    # Qwen 2.5 ChatML format - Exact tagging is vital for Qwen models
    prompt = (
        f"<|im_start|>system\nYou are MerlinOS. "
        f"Current System State:\n{context}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )

    # 3. GENERATION WITH HEARTBEAT
    print("Merlin is thinking...", end="\r")
    
    # We use stream=True so you see the first word immediately
    response_stream = llm(
        prompt, 
        max_tokens=256, 
        stop=["<|im_end|>", "<|endoftext|>"], 
        stream=True
    )

    print("Merlin AI: ", end="", flush=True)
    for chunk in response_stream:
        # Check if the chunk contains text
        if "choices" in chunk and len(chunk["choices"]) > 0:
            token = chunk["choices"][0].get("text", "")
            print(token, end="", flush=True)
    
    print() # New line after the response
