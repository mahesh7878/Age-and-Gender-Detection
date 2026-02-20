import urllib.request
import os

models_dir = os.path.dirname(os.path.abspath(__file__))

# Working URLs for models
models = {
    "age_net.caffemodel": [
        "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/age_net.caffemodel",
    ],
    "opencv_face_detector_uint8.pb": [
        "https://github.com/opencv/opencv_3rdparty/raw/8033c2bc31b3256f0d461c919ecc01c2428ca03b/opencv_face_detector_uint8.pb",
        "https://dl.opencv.org/models/opencv_face_detector_uint8.pb",
    ],
    "gender_net.caffemodel": [
        "https://github.com/smahesh29/Gender-and-Age-Detection/raw/master/gender_net.caffemodel",
        "https://github.com/nagitej/Gender-Detection/raw/master/gender_net.caffemodel",
    ]
}

print("Downloading required model files...")

for filename, urls in models.items():
    filepath = os.path.join(models_dir, filename)
    
    # Skip if already exists and is a file (not a directory)
    if os.path.isfile(filepath):
        size = os.path.getsize(filepath)
        if size > 1000:  # File exists and has content
            print(f"✓ {filename} already exists ({size} bytes)")
            continue
    
    print(f"\nDownloading {filename}...")
    downloaded = False
    
    for url in urls:
        try:
            print(f"  Trying: {url}")
            urllib.request.urlretrieve(url, filepath)
            size = os.path.getsize(filepath)
            print(f"  ✓ Success! Downloaded {size} bytes")
            downloaded = True
            break
        except Exception as e:
            print(f"  ✗ Failed: {str(e)[:50]}")
            continue
    
    if not downloaded:
        print(f"✗ Could not download {filename}")

print("\n" + "="*50)
print("Download complete!")
print("="*50)
