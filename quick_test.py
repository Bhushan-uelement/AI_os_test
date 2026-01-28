import cv2
import numpy as np
import time
import sys

# 1. Setup paths
model_proto = "MobileNetSSD_deploy.prototxt"
model_weights = "MobileNetSSD_deploy.caffemodel"
input_video = "input.mp4"
output_video = "output.mp4"

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# 2. Load the model
print(f"[INFO] Loading model: {model_weights}...")
try:
    net = cv2.dnn.readNetFromCaffe(model_proto, model_weights)
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    sys.exit(1)

# 3. Initialize Video
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print(f"[ERROR] Could not open {input_video}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Setup VideoWriter - Using XVID for better Yocto compatibility
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print(f"[INFO] Video: {width}x{height} | Total Frames: {total_frames}")
print("[INFO] Starting detection... (Press Ctrl+C to abort)")

# 4. Processing Loop
start_time = time.time()
frame_count = 0

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        (h, w) = frame.shape[:2]
        
        # Detection logic
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        objects_found = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                label_text = CLASSES[idx]
                objects_found.append(label_text)
                
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)

        # MANUAL PROGRESS BAR AND LOGGING
        percent = (frame_count / total_frames) * 100
        elapsed = time.time() - start_time
        fps_current = frame_count / elapsed
        
        # Create a visual bar [#####.....]
        bar_length = 20
        filled = int(bar_length * frame_count // total_frames)
        bar = '█' * filled + '-' * (bar_length - filled)
        
        # Print status update to the same line
        msg = f"\r[PROGRESS] |{bar}| {percent:.1f}% | Frame: {frame_count}/{total_frames} | Detected: {len(objects_found)} objects"
        sys.stdout.write(msg)
        sys.stdout.flush()

except KeyboardInterrupt:
    print("\n[WARN] Interrupted by user. Saving progress...")

# 5. Cleanup
total_time = time.time() - start_time
print(f"\n[INFO] Done!")
print(f"[INFO] Total Time: {total_time:.2f}s | Average Speed: {frame_count/total_time:.2f} FPS")
print(f"[INFO] Result saved as: {output_video}")

cap.release()
out.release()
