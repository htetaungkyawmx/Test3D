from ultralytics import YOLO

print("=" * 60)
print("CUSTOM MODEL TRAINING")
print("=" * 60)

# 1. Pre-trained model ကိုယူမယ်
model = YOLO('yolov8n.pt')  # ဒါမှမဟုတ် 'yolov8s.pt', 'yolov8m.pt'

# 2. ကိုယ်ပိုင် dataset နဲ့ train လုပ်မယ်
# သင့်မှာ dataset ရှိရင် အောက်ပါအတိုင်း train လုပ်လို့ရတယ်
"""
model.train(
    data='dataset.yaml',  # သင့် dataset ရဲ့ yaml ဖိုင်
    epochs=100,           # အကြိမ်ရေ
    imgsz=640,            # ပုံအရွယ်
    batch=16,             # တစ်ခါသွင်းမယ့်ပုံအရေအတွက်
    name='custom_model'   # သိမ်းမယ့်နာမည်
)
"""

print("\nTraining လုပ်ဖို့အတွက်:")
print("1. Dataset လိုအပ်ပါတယ် (ပုံတွေနဲ့ label တွေ)")
print("2. dataset.yaml ဖိုင်လိုအပ်ပါတယ်")
print("3. အနည်းဆုံး ပုံ ၁၀၀ လောက်ရှိရပါမယ်")

# 3. Train လုပ်ပြီးသား model ကိုသုံးမယ်
print("\nTrain လုပ်ပြီးသား model ကိုသုံးချင်ရင်:")
print("model = YOLO('path/to/your/model.pt')")