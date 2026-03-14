import urllib.request
import os
import ssl

# SSL certificate error ရှင်းဖို့
ssl._create_default_https_context = ssl._create_unverified_context

print("=" * 60)
print("DOWNLOADING SAMPLE IMAGES")
print("=" * 60)

# Browser လိုယောင်ဆောင်ဖို့ headers
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# နမူနာပုံတွေ (အခြား source ကိုသုံးမယ်)
sample_images = {
    'cat.jpg': 'https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?w=400',
    'dog.jpg': 'https://images.unsplash.com/photo-1517849845537-4d257902454a?w=400',
    'car.jpg': 'https://images.unsplash.com/photo-1494976388531-d1058494cdd8?w=400',
    'people.jpg': 'https://images.unsplash.com/photo-1511988617509-a57c8a288659?w=400'
}

for filename, url in sample_images.items():
    print(f"\nDownloading {filename}...")
    try:
        # Request ကို headers နဲ့ဆောက်မယ်
        req = urllib.request.Request(url, headers=headers)
        
        # Download လုပ်မယ်
        with urllib.request.urlopen(req) as response:
            data = response.read()
            
            # File ကိုသိမ်းမယ်
            with open(filename, 'wb') as f:
                f.write(data)
                
        print(f"✓ {filename} downloaded successfully!")
        
    except Exception as e:
        print(f"❌ {filename} download failed: {str(e)}")

print("\n" + "=" * 60)
print("Download process completed!")
print("=" * 60)