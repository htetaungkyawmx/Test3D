import cv2
from ultralytics import YOLO
import numpy as np
from datetime import datetime

# မြန်မာလိုအမည်များ
myanmar_names = {
    'person': 'လူ',
    'bicycle': 'စက်ဘီး',
    'car': 'ကား',
    'motorcycle': 'ဆိုင်ကယ်',
    'bus': 'ဘတ်စ်ကား',
    'truck': 'ထရပ်ကား',
    'cat': 'ကြောင်',
    'dog': 'ခွေး',
    'bird': 'ငှက်',
    'book': 'စာအုပ်',
    'cell phone': 'ဖုန်း',
    'laptop': 'လက်တော့ပ်',
    'bottle': 'ပုလင်း',
    'cup': 'ခွက်',
    'chair': 'ထိုင်',
    'table': 'စားပွဲ'
}

print("=" * 60)
print("ADVANCED OBJECT DETECTION")
print("=" * 60)

# Model ကို Load လုပ်မယ်
model = YOLO('yolov8n.pt')

# Webcam ကိုဖွင့်မယ်
cap = cv2.VideoCapture(0)

# Video recorder ဆောက်မယ်
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
recording = False

# Settings
confidence_threshold = 0.5
show_myanmar = True
show_fps = True

# FPS calculation
fps_start_time = datetime.now()
fps_frame_count = 0
fps = 0

print("\nCommands:")
print("  q - ရပ်ရန်")
print("  s - ပုံသိမ်းရန်")
print("  r - recording စရန်/ရပ်ရန်")
print("  c - ယုံကြည်မှုပြောင်းရန်")
print("  m - မြန်မာ/အင်္ဂလိပ် ပြောင်းရန်")
print("  f - FPS ပြရန်/မပြရန် ပြောင်းရန်")
print("-" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # FPS တွက်မယ်
    fps_frame_count += 1
    if (datetime.now() - fps_start_time).total_seconds() >= 1:
        fps = fps_frame_count
        fps_frame_count = 0
        fps_start_time = datetime.now()
    
    # Object detection လုပ်မယ်
    results = model(frame)
    
    # Result တွေကိုဆွဲမယ်
    for r in results:
        boxes = r.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < confidence_threshold:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            # အရောင်ရွေးမယ် (class ID အလိုက်)
            color = (0, 255, 0)  # Default green
            if class_name in ['person']:
                color = (0, 0, 255)  # Red for person
            elif class_name in ['car', 'truck', 'bus']:
                color = (255, 0, 0)  # Blue for vehicles
            elif class_name in ['cat', 'dog']:
                color = (0, 255, 255)  # Yellow for animals
            
            # အကွက်ဆွဲမယ်
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # အမည်ပြမယ်
            if show_myanmar and class_name in myanmar_names:
                display_name = myanmar_names[class_name]
            else:
                display_name = class_name
            
            label = f'{display_name}: {confidence:.2f}'
            
            # စာသားနောက်ခံထည့်မယ်
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            
            # စာသားထည့်မယ်
            cv2.putText(frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Information display
    info_y = 30
    cv2.putText(frame, f"Confidence: {confidence_threshold:.2f}", (10, info_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    if show_fps:
        cv2.putText(frame, f"FPS: {fps}", (10, info_y + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    if recording:
        cv2.putText(frame, "REC", (frame.shape[1] - 100, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        out.write(frame)
    
    # Frame ကိုပြမယ်
    cv2.imshow('Advanced Object Detection', frame)
    
    # Key press ကိုစစ်ဆေးမယ်
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f'detection_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg'
        cv2.imwrite(filename, frame)
        print(f"✓ ပုံသိမ်းပြီးပါပြင်: {filename}")
    elif key == ord('r'):
        recording = not recording
        if recording:
            filename = f'recording_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
            out = cv2.VideoWriter(filename, fourcc, 20.0, (frame.shape[1], frame.shape[0]))
            print(f"✓ Recording စတင်ပါပြီ: {filename}")
        else:
            out.release()
            print("✓ Recording ရပ်ဆိုင်းပါပြီ")
    elif key == ord('c'):
        confidence_threshold = (confidence_threshold + 0.1) % 1.0
        print(f"✓ ယုံကြည်မှုပမာဏ: {confidence_threshold:.2f}")
    elif key == ord('m'):
        show_myanmar = not show_myanmar
        print(f"✓ ဘာသာစကား: {'မြန်မာ' if show_myanmar else 'အင်္ဂလိပ်'}")
    elif key == ord('f'):
        show_fps = not show_fps

# Clean up
cap.release()
if recording:
    out.release()
cv2.destroyAllWindows()
print("\n✓ Program ရပ်ဆိုင်းသွားပါပြီ")