import cv2 as cv
import os

print("Testing model loading...")

# Check files exist
files = {
    "age_deploy.prototxt": os.path.exists("age_deploy.prototxt"),
    "age_net.caffemodel": os.path.exists("age_net.caffemodel"),
    "gender_deploy.prototxt": os.path.exists("gender_deploy.prototxt"),
    "gender_net.caffemodel": os.path.exists("gender_net.caffemodel"),
    "opencv_face_detector.pbtxt": os.path.exists("opencv_face_detector.pbtxt"),
    "opencv_face_detector_uint8.pb": os.path.exists("opencv_face_detector_uint8.pb"),
}

print("\nFile existence check:")
for fname, exists in files.items():
    status = "✓" if exists else "✗"
    print(f"  {status} {fname}")

print("\nFile sizes:")
for fname in files.keys():
    if os.path.exists(fname):
        size = os.path.getsize(fname)
        print(f"  {fname}: {size:,} bytes")

print("\nLoading models...")
try:
    ageNet = cv.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
    print("  ✓ Age model loaded")
except Exception as e:
    print(f"  ✗ Age model failed: {e}")

try:
    genderNet = cv.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")
    print("  ✓ Gender model loaded")
except Exception as e:
    print(f"  ✗ Gender model failed: {e}")

try:
    faceNet = cv.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")
    print("  ✓ Face detector model loaded")
except Exception as e:
    print(f"  ✗ Face detector model failed: {e}")

print("\nAll tests completed!")
