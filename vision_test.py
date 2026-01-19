import cv2
import numpy as np

def test_camera():
    # Test 1: Check OpenCV Build
    print(f"OpenCV Version: {cv2.__version__}")
    
    # Test 2: Attempt to open a virtual or physical camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device. (Expected if no camera is attached)")
    else:
        ret, frame = cap.read()
        print(f"Frame Captured: {ret}, Shape: {frame.shape if ret else 'N/A'}")
        cap.release()

    # Test 3: Synthetic Image Processing (No camera needed)
    blank_image = np.zeros((480, 640, 3), np.uint8)
    cv2.circle(blank_image, (320, 240), 100, (255, 0, 0), -1)
    cv2.imwrite('test_output.jpg', blank_image)
    print("Synthetic Image Test: Saved test_output.jpg")

if __name__ == "__main__":
    test_camera()
