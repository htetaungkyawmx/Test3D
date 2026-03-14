# download_sample.py ဖိုင်ဆောက်ပြီး အောက်ပါကုဒ်ကိုကူးထည့်ပါ
import urllib.request
import os

print("Downloading sample images...")

# နမူနာပုံတွေကို download လုပ်မယ်
sample_images = {
    'cat.jpg': 'https://images.pexels.com/photos/45201/kitty-cat-kitten-pet-45201.jpeg',
    'dog.jpg': 'https://images.pexels.com/photos/1805164/pexels-photo-1805164.jpeg',
    'car.jpg': 'https://images.pexels.com/photos/170811/pexels-photo-170811.jpeg',
    'people.jpg': 'https://images.pexels.com/photos/697243/pexels-photo-697243.jpeg'
}

for filename, url in sample_images.items():
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"✓ {filename} downloaded!")

print("\nAll sample images downloaded!")