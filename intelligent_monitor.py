import pickle
import csv
import subprocess
from llama_cpp import Llama

# 1. Load the Model & Templates (Standard Python Logic)
with open('merlin_decision_tree.pkl', 'rb') as f:
    model_data = pickle.load(f)
    # Extract model and features from the pickle you made on IdeaPad
    clf = model_data['model']
    feature_names = model_data['features']

templates = {}
with open('Linux_2k.log_templates.csv', mode='r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        templates[row['EventId']] = row['EventTemplate'].replace('<*>', '').strip()

# 2. Load the SLM (Qwen 0.5B)
llm = Llama(model_path="/root/qwen.gguf", n_ctx=1024, verbose=False)

def get_event_id(log_line):
    for eid, template in templates.items():
        if template in log_line: return eid
    return "E0"

print("--- MerlinOS Intelligence Agent Online ---")

while True:
    user_query = input("\nUser @ MerlinOS: ")
    
    # A. FETCH REAL-TIME CONTEXT
    # We grab the last few logs to see if an anomaly is happening RIGHT NOW
    raw_logs = subprocess.check_output("journalctl -n 5 --no-pager", shell=True, text=True)
    last_log_line = raw_logs.strip().split('\n')[-1]
    eid = get_event_id(last_log_line)
    
    # B. RUN ML INFERENCE (The "Hidden" Check)
    vector = [1 if feat == eid else 0 for feat in feature_names]
    is_anomaly = clf.predict([vector])[0]
    
    # C. CONSTRUCT THE "BRAIN" PROMPT
    # We tell the AI what the ML model found, but let the AI answer the user.
    status_report = "ANOMALY DETECTED" if is_anomaly == 1 else "System is Healthy"
    
    prompt = f"""<|im_start|>system
You are MerlinOS AI. 
ML MONITOR VERDICT: {status_report} (Last Event: {eid})
CURRENT LOG: {last_log_line}

INSTRUCTIONS:
1. Answer the user's question.
2. If the ML Verdict is an anomaly, WARN the user regardless of their question.
3. Keep it technical and concise.<|im_end|>
<|im_start|>user
{user_query}<|im_end|>
<|im_start|>assistant
"""

    # D. GENERATE NL RESPONSE
    print("Merlin AI: ", end="", flush=True)
    output = llm(prompt, max_tokens=200, stop=["<|im_end|>"], stream=True)
    
    for chunk in output:
        print(chunk['choices'][0]['text'], end="", flush=True)
    print()
