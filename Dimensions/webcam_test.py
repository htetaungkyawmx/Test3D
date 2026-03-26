import cv2
from ultralytics import YOLO

print("=" * 60)
print("WEBCAM OBJECT DETECTION")
print("=" * 60)

# Model ကို Load လုပ်မယ်
print("\nLoading YOLO model...")
model = YOLO('yolov8n.pt')
print("✓ Model loaded!")

# Webcam ကိုဖွင့်မယ်
print("\nOpening webcam...")
print("Press 'q' to quit")
print("Press 's' to save current frame")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam မဖွင့်နိုင်ပါ")
    exit()

frame_count = 0

while True:
    # Frame ဖတ်မယ်
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Object detection လုပ်မယ် (frame ၃ ခုခံမှတစ်ခါလုပ်ပါ)
    if frame_count % 3 == 0:
        results = model(frame)
        
        # Result တွေကိုဆွဲမယ်
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                confidence = float(box.conf[0])
                
                # အကွက်ဆွဲမယ်
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{confidence:.2f}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Frame ကိုပြမယ်
    cv2.imshow('Webcam Object Detection - Press "q" to quit', frame)
    
    # Key press ကိုစစ်ဆေးမယ်
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):  # 'q' နှိပ်ရင်ရပ်မယ်
        break
    elif key == ord('s'):  # 's' နှိပ်ရင်ပုံသိမ်းမယ်
        cv2.imwrite(f'webcam_capture_{frame_count}.jpg', frame)
        print(f"✓ Frame {frame_count} ကိုသိမ်းပြီးပါပြီ")

# Clean up
cap.release()
cv2.destroyAllWindows()
print("\n✓ Webcam detection stopped")