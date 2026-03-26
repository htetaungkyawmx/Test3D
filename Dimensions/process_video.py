import cv2
from ultralytics import YOLO

# Class names
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
print("VIDEO OBJECT DETECTION")
print("=" * 60)

# Model ကို Load လုပ်မယ်
model = YOLO('yolov8n.pt')

# Video ဖိုင်ကိုဖွင့်မယ်
video_path = input("Video ဖိုင်ရဲ့လမ်းကြောင်းထည့်ပါ: ")
cap = cv2.VideoCapture(video_path)

# Video ရဲ့အချက်အလက်တွေယူမယ်
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer ဆောက်မယ်
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\nVideo အချက်အလက်များ:")
print(f"  - Frame အရေအတွက်: {total_frames}")
print(f"  - FPS: {fps}")
print(f"  - Resolution: {width}x{height}")
print(f"\nProcessing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    # Object detection လုပ်မယ်
    results = model(frame)
    
    # Result တွေကိုဆွဲမယ်
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            
            # အကွက်ဆွဲမယ်
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {confidence:.2f}', 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Frame ကို save မယ်
    out.write(frame)
    
    # Progress ပြမယ်
    if frame_count % 30 == 0:  # 30 frame တစ်ခါ
        progress = (frame_count / total_frames) * 100
        print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames})")

cap.release()
out.release()
print(f"\n✓ Video processing complete!")
print(f"✓ Output file: output_video.mp4")