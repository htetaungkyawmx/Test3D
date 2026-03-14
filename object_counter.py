import cv2
from ultralytics import YOLO
from collections import defaultdict

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
print("OBJECT COUNTER")
print("=" * 60)

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

object_counts = defaultdict(int)
frame_count = 0

print("\nStarting object counting...")
print("Press 'q' to quit")
print("-" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    results = model(frame)
    
    # ဒီ frame ထဲကအရာဝတ္ထုတွေကို ရေတွက်မယ်
    frame_counts = defaultdict(int)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            frame_counts[class_name] += 1
    
    # စုစုပေါင်းကို update လုပ်မယ်
    for obj, count in frame_counts.items():
        object_counts[obj] = max(object_counts[obj], count)
    
    # Display မှာပြမယ်
    y_pos = 30
    cv2.putText(frame, f"Frame: {frame_count}", (10, y_pos), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    y_pos += 25
    for obj, count in object_counts.items():
        if count > 0:
            cv2.putText(frame, f"{obj}: {count}", (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_pos += 20
    
    cv2.imshow('Object Counter', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("FINAL COUNTS:")
for obj, count in object_counts.items():
    if count > 0:
        print(f"  {obj}: {count}")
print("=" * 60)