import pickle
import pandas as pd
import subprocess
from llama_cpp import Llama

# 1. Load the Model and Templates
with open('merlin_decision_tree.pkl', 'rb') as f:
    clf = pickle.load(f)

templates = pd.read_csv('Linux_2k.log_templates.csv')
# Load the SLM (assuming llama-server isn't running, otherwise use requests)
llm = Llama(model_path="/root/qwen.gguf", n_ctx=512, verbose=False)

def get_event_id(log_line):
    """Simple keyword matcher to find the EventID from templates."""
    for _, row in templates.iterrows():
        # Clean the template to make matching easier
        template_clean = row['EventTemplate'].replace('<*>', '')
        if template_clean in log_line:
            return row['EventId']
    return "E0" # Unknown event

print("--- MerlinOS Live ML Monitor Active ---")

while True:
    # 2. Capture a live log line (simulating journalctl tail)
    # For testing, we'll just take a simulated input
    raw_log = input("\n[Live Log Input]: ")
    
    eid = get_event_id(raw_log)
    
    # 3. Decision Tree Inference
    # We create a simple feature vector (1 for the detected event, 0 for others)
    # Note: In a production version, you'd use a window of multiple events.
    prediction = clf.predict([[1 if i == eid else 0 for i in clf.feature_names_in_]])[0]

    if prediction == 1:
        print("ML VERDICT: ANOMALY DETECTED")
        
        # 4. SLM Explanation
        prompt = f"<|im_start|>system\nYou are MerlinOS. The ML model detected an anomaly: {eid}.\nLog: {raw_log}\nExplain the risk and suggest a fix.<|im_end|>\n<|im_start|>user\nStatus report?<|im_end|>\n<|im_start|>assistant\n"
        output = llm(prompt, max_tokens=128, stop=["<|im_end|>"])
        print(f"Merlin AI: {output['choices'][0]['text'].strip()}")
    else:
        print("ML VERDICT: SYSTEM HEALTHY")
