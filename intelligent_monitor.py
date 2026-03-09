import json
import csv
import subprocess
from llama_cpp import Llama

# 1. Load the Rules & Templates
with open('merlin_rules.json', 'r') as f:
    config = json.load(f)
    RULES = config['tree']
    FEATURES = config['features']

templates = {}
with open('Linux_2k.log_templates.csv', mode='r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        templates[row['EventId']] = row['EventTemplate'].replace('<*>', '').strip()

# 2. Initialize SLM with a slightly larger context for detailed answers
llm = Llama(model_path="/root/qwen.gguf", n_ctx=2048, verbose=False)

def walk_tree(node, sample_counts):
    if isinstance(node, int):
        return node
    val = sample_counts.get(node['feature'], 0)
    if val <= node['threshold']:
        return walk_tree(node['left'], sample_counts)
    else:
        return walk_tree(node['right'], sample_counts)

def get_status():
    try:
        raw_log = subprocess.check_output("journalctl -n 1 --no-pager", shell=True, text=True).strip()
    except:
        raw_log = "System log access unavailable."
    
    eid = "E0"
    for id_code, text in templates.items():
        if text in raw_log:
            eid = id_code
            break
    
    verdict = walk_tree(RULES, {eid: 1})
    return "CRITICAL" if verdict == 1 else "Normal", eid, raw_log

print("--- MerlinOS Intelligence Agent (Powered by uElement) ---")

while True:
    user_query = input("\nUser @ MerlinOS: ")
    if user_query.lower() in ["exit", "quit"]: break
    
    status, eid, log = get_status()
    
    # 3. ENHANCED SYSTEM PROMPT
    # Added branding and depth instructions
    prompt = f"""<|im_start|>system
You are MerlinOS, a sophisticated AI Operating System created by **uElement**. 
Your purpose is to manage edge computing, drone systems, and system security.

[CURRENT SYSTEM TELEMETRY]
Machine Learning Verdict: {status}
Diagnostic Event ID: {eid}
Active Kernel Log: {log}

[OPERATIONAL GUIDELINES]
- If asked about your origin or MerlinOS, explicitly state you were created by **uElement**.
- Provide detailed, technical, and insightful responses. Do not give one-sentence answers.
- Explain the 'why' behind system events.
- If the ML Verdict is CRITICAL, interrupt the user's request with a high-priority security alert.
<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
"""

    print("Merlin AI: ", end="", flush=True)
    # Increased max_tokens to 512 for more detailed answers
    for chunk in llm(prompt, max_tokens=512, stop=["<|im_end|>", "User:"], stream=True):
        if "choices" in chunk:
            print(chunk['choices'][0]['text'], end="", flush=True)
    print()
