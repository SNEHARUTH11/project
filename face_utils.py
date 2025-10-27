import cv2
import os
import numpy as np
from datetime import datetime

# Paths
MY_FACE_DIR = "my_face"      # Put all images of your face here
EVIDENCE_DIR = "evidence"

os.makedirs(MY_FACE_DIR, exist_ok=True)
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Initialize recognizer and detector
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- Train on your face only ---
def train_model():
    faces = []
    labels = []
    for fname in os.listdir(MY_FACE_DIR):
        fpath = os.path.join(MY_FACE_DIR, fname)
        img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        faces.append(img)
        labels.append(0)   # label 0 = you

    if faces:
        recognizer.train(faces, np.array(labels))
        print(f"[INFO] Trained on {len(faces)} images of your face.")
    else:
        print("[WARN] No face images found. Add images to 'my_face/' folder.")

# --- Recognize face ---
def recognize_face(frame, threshold=70):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        try:
            label, confidence = recognizer.predict(roi)
            conf_percent = max(0, 100 - confidence)
            if label == 0 and conf_percent >= threshold:
                return "ME", conf_percent, (x, y, w, h)  # recognized as you
            else:
                return None, conf_percent, (x, y, w, h) # unknown
        except cv2.error as e:
            print(f"[ERROR] Recognizer error: {e}")
            return None, 0, (x, y, w, h)

    return None, 0, None

# --- Save unknown face ---
def save_evidence_image(frame):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(EVIDENCE_DIR, f"evidence_{ts}.jpg")
    cv2.imwrite(path, frame)
    print(f"[EVIDENCE] Saved: {path}")
    return path
