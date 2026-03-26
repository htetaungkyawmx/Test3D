import cv2
import time

print("Testing camera with OpenCV...")

# Try camera 0
cap0 = cv2.VideoCapture(0)
if cap0.isOpened():
    print("✓ Camera 0 opened successfully!")
    ret, frame = cap0.read()
    if ret:
        print(f"✓ Frame captured! Resolution: {frame.shape[1]}x{frame.shape[0]}")
        cv2.imwrite("test_capture.jpg", frame)
        print("✓ Test image saved as test_capture.jpg")
    cap0.release()
else:
    print("✗ Camera 0 failed")

print("\n" + "="*50)
print("Now testing with cv2.VideoCapture(0) in loop...")
print("Press 'q' to quit")
print("="*50)

# Test with display
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Cannot open camera")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read frame")
        break
    
    cv2.imshow("Camera Test", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Test completed")