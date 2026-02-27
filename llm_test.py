import subprocess
from llama_cpp import Llama

# 1. Initialize the AI
# chat_format="chatml" tells llama-cpp to handle the Qwen/ChatML tags for you.
llm = Llama(
    model_path="/root/qwen.gguf", 
    n_ctx=2048, 
    verbose=False,
    chat_format="chatml" 
)

def get_system_data(command_type):
    try:
        cmds = {
            "logs": "journalctl -n 15 --no-pager",
            "ram": "free -h",
            "storage": "df -h /",
            "status": "uptime"
        }
        return subprocess.check_output(cmds.get(command_type, "uptime"), shell=True).decode()
    except:
        return "Unable to fetch system data."

print("--- MerlinOS Intelligence Agent Online ---")

while True:
    user_query = input("\nUser @ MerlinOS: ")
    
    # 2. Prepare System Context
    sys_context = "You are MerlinOS, a system-aware AI. Use the following data if relevant:\n"
    if any(w in user_query.lower() for w in ["stat", "ram", "log", "health"]):
        sys_context += get_system_data("ram" if "ram" in user_query else "logs")

    # 3. Use the High-Level Chat API
    # This method is much more stable for streaming
    stream = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": sys_context},
            {"role": "user", "content": user_query}
        ],
        stream=True
    )

    print("\nMerlin AI: ", end="", flush=True)
    for chunk in stream:
        delta = chunk["choices"][0]["delta"]
        if "content" in delta:
            print(delta["content"], end="", flush=True)
    print()
