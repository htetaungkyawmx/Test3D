"""
Object Detection Project
ပုံထဲကအရာဝတ္ထုတွေကိုရှာဖွေတဲ့ Program
"""

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

print("=" * 60)
print("OBJECT DETECTION PROJECT")
print("=" * 60)

# ============================================
# 1. YOLO Model ကို Load လုပ်မယ်
# ============================================
print("\n1. Loading YOLO model...")
model = YOLO('yolov8n.pt')  # YOLOv8 nano model (အမြန်ဆုံး)
print("✓ Model loaded successfully!")

# ============================================
# 2. ပုံထဲကအရာဝတ္ထုတွေကိုရှာမယ် (Image)
# ============================================
def detect_objects_in_image(image_path):
    """
    ပုံတစ်ပုံထဲက အရာဝတ္ထုတွေကိုရှာဖွေမယ်
    """
    print(f"\n2. Detecting objects in: {image_path}")
    
    # ပုံကို load လုပ်မယ်
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Object detection လုပ်မယ်
    results = model(image)
    
    # Result တွေကိုပြမယ်
    result_image = image_rgb.copy()
    
    # Class names တွေ
    class_names = [
        'လူ', 'စက်ဘီး', 'ကား', 'မော်တော်ဆိုင်ကယ်', 'လေယာဉ်', 'ဘတ်စ်ကား', 
        'ရထား', 'ထရပ်ကား', 'လှေ', '�ာဉ်အသွားအလာမီး', 'မီးသတ်ရေပိုက်', 
        'ရပ်တန့်အမှတ်အသား', 'မီတာချိန်စက်', 'ခုံတန်း', 'ငှက်', 'ကြောင်', 
        'ခွေး', 'မြင်း', 'သိုး', 'နွား', 'ဆင်', 'ဝက်ဝံ', 'မြင်းကျား', 
        'သစ်ကုလားအုပ်', 'ကျောပိုးအိတ်', 'ထီး', 'ပိုက်ဆံအိတ်', 'လည်စည်း', 
        'ခရီးဆောင်အိတ်', 'ဖရီစဘီ', 'စကိတ်ဘုတ်', 'ဆားဖိန်းဘုတ်', 
        'တင်းနစ်ဘောလုံး', 'ဘောလုံး', 'စွန်လွှတ်', 'ဘေ့စ်ဘောလက်အိတ်', 
        'စကိတ်ဘုတ်', 'ဘေ့စ်ဘောဘတ်', 'လက်ဝှေ့လက်အိတ်', 'စကိတ်ဘုတ်', 
        'တံငါလှံ', 'ခြေအိတ်', 'ဆိုဖာ', 'အိုး', 'စားပွဲ', 'တီဗီ', 
        'လက်တော့ပ်', 'မောက်စ်', 'အဝေးထိန်း', 'ကီးဘုတ်', 'ဆဲလ်ဖုန်း', 
        'မိုက်ခရိုဝေ့', 'မီးဖို', 'မီးဖိုချောင်သုံး', 'ရေခဲသေတ္တာ', 'စာအုပ်', 
        'နာရီ', 'ပန်းအိုး', 'ကတ်ကြေး', 'တက်ဘီ', 'သွားတိုက်တံ'
    ]
    
    # Detection result တွေကို ဆွဲမယ်
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding box ဆွဲမယ်
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Class name နဲ့ confidence ယူမယ်
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            
            if class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = f'Class-{class_id}'
            
            # အကွက်ဆွဲမယ်
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # စာသားထည့်မယ်
            label = f'{class_name}: {confidence:.2f}'
            cv2.putText(result_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            print(f"  - တွေ့ရှိခဲ့သည်: {class_name} (ယုံကြည်မှု: {confidence:.2f})")
    
    return result_image

# ============================================
# 3. Webcam ကနေ တိုက်ရိုက်ရှာဖွေမယ် (Real-time)
# ============================================
def detect_objects_webcam():
    """
    Webcam ကိုဖွင့်ပြီး real-time object detection လုပ်မယ်
    """
    print("\n3. Starting webcam detection...")
    print("   Webcam ကိုဖွင့်နေပါပြီ...")
    print("   'q' ကိုနှိပ်ပြီး ရပ်လိုက်ပါ")
    
    # Webcam ကိုဖွင့်မယ်
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Webcam မဖွင့်နိုင်ပါ")
        return
    
    while True:
        # Frame တစ်ခုချင်းစီဖတ်မယ်
        ret, frame = cap.read()
        if not ret:
            break
        
        # Object detection လုပ်မယ်
        results = model(frame)
        
        # Result တွေကိုဆွဲမယ်
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # အကွက်ဆွဲမယ်
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'ID: {class_id} {confidence:.2f}', 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
        
        # Frame ကိုပြမယ်
        cv2.imshow('Object Detection - Press "q" to quit', frame)
        
        # 'q' နှိပ်ရပ်ရပ်လိုက်မယ်
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("✓ Webcam detection stopped")

# ============================================
# 4. Video File ထဲကရှာဖွေမယ်
# ============================================
def detect_objects_in_video(video_path):
    """
    Video file ထဲက object တွေကိုရှာဖွေမယ်
    """
    print(f"\n4. Detecting objects in video: {video_path}")
    
    # Video ကိုဖွင့်မယ်
    cap = cv2.VideoCapture(video_path)
    
    # Video writer ဆောက်မယ်
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, 
                          (int(cap.get(3)), int(cap.get(4))))
    
    frame_count = 0
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
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{class_id}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Frame ကို save မယ်
        out.write(frame)
        
        frame_count += 1
        if frame_count % 30 == 0:  # 30 frame တစ်ခါပြမယ်
            print(f"   Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    print(f"✓ Video processing complete! Output saved as 'output_video.mp4'")

# ============================================
# 5. Main Program
# ============================================
def main():
    print("\n" + "=" * 60)
    print("OBJECT DETECTION PROGRAM")
    print("=" * 60)
    print("\nဘယ် mode နဲ့အလုပ်လုပ်မလဲ?")
    print("1. ပုံတစ်ပုံထဲကရှာမယ်")
    print("2. Webcam နဲ့ရှာမယ်")
    print("3. Video ဖိုင်ထဲကရှာမယ်")
    print("4. ထွက်မယ်")
    
    choice = input("\nရွေးချယ်ပါ (1-4): ")
    
    if choice == '1':
        # ပုံတစ်ပုံထဲကရှာမယ်
        image_path = input("ပုံဖိုင်ရဲ့လမ်းကြောင်းထည့်ပါ (ဥပမာ: test.jpg): ")
        
        if os.path.exists(image_path):
            result_image = detect_objects_in_image(image_path)
            
            # Result ကိုပြမယ်
            plt.figure(figsize=(12, 8))
            plt.imshow(result_image)
            plt.title('Object Detection Result')
            plt.axis('off')
            plt.show()
            
            # Result ကို save မယ်
            cv2.imwrite('result_image.jpg', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            print("✓ Result ကို 'result_image.jpg' အဖြစ်သိမ်းပြီးပါပြီ")
        else:
            print(f"❌ ပုံဖိုင် {image_path} မရှိပါ")
    
    elif choice == '2':
        # Webcam နဲ့ရှာမယ်
        detect_objects_webcam()
    
    elif choice == '3':
        # Video ဖိုင်ထဲကရှာမယ်
        video_path = input("Video ဖိုင်ရဲ့လမ်းကြောင်းထည့်ပါ (ဥပမာ: test.mp4): ")
        
        if os.path.exists(video_path):
            detect_objects_in_video(video_path)
        else:
            print(f"❌ Video ဖိုင် {video_path} မရှိပါ")
    
    elif choice == '4':
        print("ကျေးဇူးတင်ပါတယ်! နောက်မှပြန်ဆုံကြမယ်။")
        return
    
    else:
        print("❌ မှားယွင်းသောရွေးချယ်မှုပါ")

if __name__ == "__main__":
    main()