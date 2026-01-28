import cv2
import numpy as np
from tqdm import tqdm
import time

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
net = cv2.dnn.readNetFromCaffe(model_proto, model_weights)

# 3. Initialize Video
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("[ERROR] Could not open video file.")
    exit()

# Get video properties for logging
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"[INFO] Video Stats: {width}x{height} | {fps} FPS | Total Frames: {total_frames}")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

# 4. Processing with Progress Bar
print("[INFO] Starting detection...")
start_time = time.time()

# 'unit' defines what each iteration represents (frames)
with tqdm(total=total_frames, unit='frame', desc="Processing Video") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        
        net.setInput(blob)
        detections = net.forward()

        objects_in_frame = 0
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                objects_in_frame += 1
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                label = f"{CLASSES[idx]}: {confidence:.2f}%"
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        pbar.set_postfix({"Detected": objects_in_frame}) # Show detections in progress bar
        pbar.update(1)

total_time = time.time() - start_time
print(f"\n[INFO] Done! Output saved to {output_video}")
print(f"[INFO] Total Time: {total_time:.2f}s | Average FPS: {total_frames/total_time:.2f}")

cap.release()
out.release()
