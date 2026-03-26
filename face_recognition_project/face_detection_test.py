#!/usr/bin/env python3
"""
Face Detection Test - Check if camera and face detection work
"""

import cv2
import numpy as np
import time

# Try to import insightface
try:
    import insightface
    from insightface.app import FaceAnalysis
    print("✓ InsightFace imported successfully")
except ImportError as e:
    print(f"❌ InsightFace import failed: {e}")
    exit(1)

print("\n" + "="*60)
print("FACE DETECTION TEST")
print("="*60)

# Initialize face models
print("\n1. Loading face models...")
app = FaceAnalysis(name='buffalo_l', root='saved_data')
app.prepare(ctx_id=-1, det_size=(320, 320))  # Smaller size for speed
print("✓ Models loaded")

# Open camera
print("\n2. Opening camera...")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera!")
    exit(1)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("✓ Camera opened")

# Test frame capture
print("\n3. Testing frame capture...")
ret, test_frame = cap.read()
if not ret:
    print("❌ Cannot capture frame!")
    cap.release()
    exit(1)

print(f"✓ Frame captured: {test_frame.shape}")

print("\n" + "="*60)
print("TESTING FACE DETECTION")
print("="*60)
print("Please look at the camera...")
print("Press 'q' to quit")
print("Press 's' to save current frame")
print("="*60 + "\n")

frame_count = 0
detection_count = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Cannot read frame")
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # Detect faces
        faces = app.get(frame)
        
        # Draw results
        if len(faces) > 0:
            detection_count += 1
            for face in faces:
                bbox = face.bbox.astype(int)
                score = face.det_score
                
                # Draw bounding box
                color = (0, 255, 0)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Draw score
                cv2.putText(frame, f"Face: {score:.2f}", (bbox[0], bbox[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw center point
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)
        
        # Show detection status
        status_color = (0, 255, 0) if len(faces) > 0 else (0, 0, 255)
        status_text = f"Faces: {len(faces)}" if len(faces) > 0 else "No Face Detected"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Show instructions
        cv2.putText(frame, "Press 'q' to quit | 's' to save", (10, frame.shape[0]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow("Face Detection Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"test_frame_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Saved: {filename}")
        
        # Print detection rate every 30 frames
        if frame_count % 30 == 0:
            detection_rate = (detection_count / frame_count) * 100 if frame_count > 0 else 0
            print(f"Detection rate: {detection_rate:.1f}% ({detection_count}/{frame_count})")

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    if detection_count > 0:
        print(f"✅ Face detection is WORKING!")
        print(f"   Detected faces in {detection_count} out of {frame_count} frames")
        print(f"   Success rate: {(detection_count/frame_count)*100:.1f}%")
    else:
        print(f"❌ Face detection is NOT WORKING!")
        print(f"   No faces detected in {frame_count} frames")
        print("\nTroubleshooting:")
        print("1. Make sure your face is clearly visible")
        print("2. Check lighting - bright, even lighting works best")
        print("3. Look directly at the camera")
        print("4. Remove glasses or mask if wearing")
    print("="*60)