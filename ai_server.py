from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Simulate AI processing latency
    data = request.get_json()
    return jsonify({"status": "success", "processed_value": sum(data['input'])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
