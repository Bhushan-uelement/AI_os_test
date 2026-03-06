import json
import csv
import subprocess
from llama_cpp import Llama

# 1. Load the Rules (No Pickle/Sklearn needed!)
with open('merlin_rules.json', 'r') as f:
    config = json.load(f)
    RULES = config['tree']
    FEATURES = config['features']

# 2. Load Templates
templates = {}
with open('Linux_2k.log_templates.csv', mode='r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        templates[row['EventId']] = row['EventTemplate'].replace('<*>', '').strip()

# 3. Initialize SLM
llm = Llama(model_path="/root/qwen.gguf", n_ctx=1024, verbose=False)

def walk_tree(node, sample_counts):
    """Manually traverses the decision tree rules."""
    if isinstance(node, int): # We hit a leaf
        return node
    
    val = sample_counts.get(node['feature'], 0)
    if val <= node['threshold']:
        return walk_tree(node['left'], sample_counts)
    else:
        return walk_tree(node['right'], sample_counts)

def get_status():
    raw_log = subprocess.check_output("journalctl -n 1 --no-pager", shell=True, text=True).strip()
    eid = "E0"
    for id_code, text in templates.items():
        if text in raw_log:
            eid = id_code
            break
    
    # Run Inference
    verdict = walk_tree(RULES, {eid: 1})
    return "CRITICAL" if verdict == 1 else "Normal", eid, raw_log

print("--- MerlinOS Intelligent Monitor (Zero-Sklearn) Online ---")

while True:
    user_query = input("\nUser @ MerlinOS: ")
    
    status, eid, log = get_status()
    
    prompt = f"""<|im_start|>system
You are MerlinOS AI. 
[SYSTEM STATUS]
ML Verdict: {status}
Log ID: {eid} | Log: {log}

[INSTRUCTIONS]
- Answer the user's query.
- If status is CRITICAL, warn them immediately.
<|im_end|>\n<|im_start|>user\n{user_query}<|im_end|>\n<|im_start|>assistant\n"""

    print("Merlin AI: ", end="", flush=True)
    for chunk in llm(prompt, max_tokens=200, stop=["<|im_end|>"], stream=True):
        print(chunk['choices'][0]['text'], end="", flush=True)
    print()
