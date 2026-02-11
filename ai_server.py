from flask import Flask, request, jsonify
import threading
import time

app = Flask(__name__)

# Global variable to store the latest result
# This prevents the AI loop from blocking the web server
latest_ai_result = {"status": "initializing", "value": 0}

def ai_inference_loop():
    """Simulates the Interceptor AI detection loop in a separate thread"""
    global latest_ai_result
    while True:
        # Simulate an Interceptor task (like processing a camera frame)
        # In your final version, replace this with your ONNX/OpenCV detection
        time.sleep(0.5) 
        latest_ai_result = {
            "status": "active",
            "timestamp": time.time(),
            "interceptor_status": "searching_for_targets"
        }

@app.route('/')
def home():
    """GET route for browser testing"""
    return jsonify({
        "message": "KnowledgeOS Interceptor Server Online",
        "current_state": latest_ai_result
    })

@app.route('/predict', methods=['POST'])
def predict():
    """POST route for drone data processing"""
    data = request.get_json()
    if not data or 'input' not in data:
        return jsonify({"error": "Invalid input"}), 400
    
    processed = sum(data['input'])
    return jsonify({
        "status": "success",
        "processed_value": processed,
        "node": "interceptor-01"
    })

if __name__ == '__main__':
    # Start AI loop in background
    threading.Thread(target=ai_inference_loop, daemon=True).start()
    
    # Run server - 'threaded=True' is vital for responsiveness
    app.run(host='0.0.0.0', port=5000, threaded=True)
