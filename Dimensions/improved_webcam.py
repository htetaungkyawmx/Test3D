import cv2
from ultralytics import YOLO

# COCO dataset class names (YOLO သုံးတဲ့အရာဝတ္ထုအမည်များ)
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

print("=" * 60)
print("IMPROVED OBJECT DETECTION")
print("=" * 60)

# Model ကို Load လုပ်မယ်
print("\nLoading YOLO model...")
model = YOLO('yolov8n.pt')
print("✓ Model loaded!")

# Webcam ကိုဖွင့်မယ်
print("\nOpening webcam...")
print("Commands:")
print("  'q' - ရပ်ရန်")
print("  's' - ပုံသိမ်းရန်")
print("  'c' - ယုံကြည်မှုပမာဏပြောင်းရန်")
print("-" * 60)

cap = cv2.VideoCapture(0)
confidence_threshold = 0.5  # စတင်မယ့်ယုံကြည်မှုပမာဏ

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Object detection လုပ်မယ်
    results = model(frame)
    
    # Result တွေကိုဆွဲမယ်
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # ယုံကြည်မှုပမာဏစစ်မယ်
            confidence = float(box.conf[0])
            if confidence < confidence_threshold:
                continue
            
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            class_id = int(box.cls[0])
            class_name = class_names[class_id] if class_id < len(class_names) else f'Class-{class_id}'
            
            # ယုံကြည်မှုအလိုက်အရောင်ပြောင်းမယ်
            if confidence > 0.8:
                color = (0, 255, 0)  # အစိမ်း - ယုံကြည်မှုမြင့်
            elif confidence > 0.5:
                color = (0, 255, 255)  # အဝါ - အလယ်အလတ်
            else:
                color = (0, 165, 255)  # လိမ္မော် - ယုံကြည်မှုနည်း
            
            # အကွက်ဆွဲမယ်
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # စာသားထည့်မယ်
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            print(f"✓ တွေ့ရှိခဲ့သည်: {class_name} (ယုံကြည်မှု: {confidence:.2f})")
    
    # ယုံကြည်မှုပမာဏကိုပြမယ်
    cv2.putText(frame, f'Confidence Threshold: {confidence_threshold:.2f}', 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Frame ကိုပြမယ်
    cv2.imshow('Object Detection - Press "q" to quit', frame)
    
    # Key press ကိုစစ်ဆေးမယ်
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f'detection_{class_name}_{confidence:.2f}.jpg', frame)
        print(f"✓ ပုံကိုသိမ်းပြီးပါပြီ")
    elif key == ord('c'):
        confidence_threshold = (confidence_threshold + 0.1) % 1.0
        print(f"✓ ယုံကြည်မှုပမာဏ {confidence_threshold:.2f} သို့ပြောင်းလဲပြီးပါပြီ")
cap.release()
cv2.destroyAllWindows()
print("\n✓ Program ရပ်ဆိုင်းသွားပါပြီ")