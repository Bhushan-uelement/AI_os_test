import cv2
import numpy as np

# 1. Setup paths (Ensure these files are on your device)
model_proto = "MobileNetSSD_deploy.prototxt"
model_weights = "MobileNetSSD_deploy.caffemodel"
input_video = "input.mp4"
output_video = "output.mp4"

# List of classes MobileNet SSD can detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# 2. Load the model
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(model_proto, model_weights)

# 3. Initialize Video
cap = cv2.VideoCapture(input_video)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print("[INFO] Processing video...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]
    # Resize and normalize frame for the model
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    
    net.setInput(blob)
    detections = net.forward()

    # Loop over detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # 50% threshold
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw label and rectangle
            label = f"{CLASSES[idx]}: {confidence:.2f}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)

print(f"[INFO] Done! Output saved to {output_video}")
cap.release()
out.release()
