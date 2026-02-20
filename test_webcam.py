#!/usr/bin/env python3
# Quick test - captures one frame from webcam and detects age/gender

import cv2 as cv
import sys

print("Initializing models...")

# Load networks
ageNet = cv.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
genderNet = cv.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")
faceNet = cv.dnn.readNet("opencv_face_detector_uint8.pb", "opencv_face_detector.pbtxt")

ageNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
genderNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
faceNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

print("Opening webcam...")
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam")
    sys.exit(1)

print("Capturing frame...")
ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Cannot read frame from webcam")
    sys.exit(1)

print(f"Frame size: {frame.shape}")

# Detect faces
blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
faceNet.setInput(blob)
detections = faceNet.forward()

face_count = 0
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        face_count += 1
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        x1 = int(detections[0, 0, i, 3] * frameWidth)
        y1 = int(detections[0, 0, i, 4] * frameHeight)
        x2 = int(detections[0, 0, i, 5] * frameWidth)
        y2 = int(detections[0, 0, i, 6] * frameHeight)
        
        padding = 20
        face = frame[max(0,y1-padding):min(y2+padding,frameHeight-1),
                     max(0,x1-padding):min(x2+padding, frameWidth-1)]
        
        if face.shape[0] > 0 and face.shape[1] > 0:
            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            gender_conf = genderPreds[0].max()
            
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            age_conf = agePreds[0].max()
            
            print(f"\nFace {face_count}:")
            print(f"  Confidence: {confidence:.2%}")
            print(f"  Gender: {gender} (confidence: {gender_conf:.3f})")
            print(f"  Age: {age} (confidence: {age_conf:.3f})")

if face_count == 0:
    print("No faces detected in the frame")
else:
    print(f"\nSuccessfully detected {face_count} face(s)")
