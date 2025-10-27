from flask import Flask, jsonify, Response, request, send_from_directory
import cv2
import serial
import threading
import time
import os
import platform
import subprocess
from flask_cors import CORS
from face_utils import train_model, recognize_face, save_evidence_image

# --- CONFIG ---
ARDUINO_PORT = "COM8"       # Update with your Arduino COM port
BAUD_RATE = 115200
CAMERA_INDEX = 0
CONF_THRESHOLD = 20
ALARM_BLINK_INTERVAL = 0.5
EVIDENCE_DIR = "evidence"

# --- INIT ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

if not os.path.exists(EVIDENCE_DIR):
    os.makedirs(EVIDENCE_DIR)

label_map = train_model()
camera_lock = threading.Lock()
camera = cv2.VideoCapture(CAMERA_INDEX)
arduino = None
alarm_active = False
events = []
last_frame = None
logged_in_user = None

# --- Connect to Arduino ---
try:
    arduino = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
    print(f"[INFO] Connected to Arduino on {ARDUINO_PORT}")
except Exception as e:
    print(f"[WARN] Could not connect to Arduino: {e}")

# --- Utility: Lock system ---
def lock_system():
    os_name = platform.system()
    print("[SECURITY] Locking system now...")
    try:
        if os_name == "Windows":
            subprocess.run("rundll32.exe user32.dll,LockWorkStation")
        elif os_name == "Linux":
            subprocess.run(["gnome-screensaver-command", "-l"])
        elif os_name == "Darwin":
            subprocess.run(["/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession", "-suspend"])
        else:
            print("[ERROR] Unsupported OS for locking")
        return True
    except Exception as e:
        print(f"[ERROR] Lock command failed: {e}")
        return False


# --- Motion Event ---
def handle_motion_event():
    global alarm_active, last_frame
    with camera_lock:
        ret, frame = camera.read()
        if not ret:
            print("[ERROR] Camera not found")
            return

        last_frame = frame.copy()
        name, conf = recognize_face(frame, label_map)

        if name and conf >= CONF_THRESHOLD:
            print(f"[SAFE] Known person detected: {name} ({conf:.1f}%)")
            if arduino:
                arduino.write(b"SAFE\n")
            events.append({
                "id": len(events)+1,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "event_type": "Known",
                "user_id": name,
                "confidence": conf
            })
        else:
            print(f"[ALERT] Unknown person detected! ({conf:.1f}%)")
            img_path = save_evidence_image(frame)

            # 🚨 NEW: Lock system automatically here!
            lock_system()

            events.append({
                "id": len(events)+1,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "event_type": "Unknown",
                "user_id": name or "Unknown",
                "confidence": conf,
                "image": img_path
            })

            if arduino and not alarm_active:
                alarm_active = True
                threading.Thread(target=blink_alarm_continuous, daemon=True).start()


# --- Alarm Blinking ---
def blink_alarm_continuous():
    global alarm_active
    print("[ALARM] Continuous blinking started")
    while alarm_active:
        if arduino:
            arduino.write(b"ALARM_ON\n")
        time.sleep(ALARM_BLINK_INTERVAL)
        if arduino:
            arduino.write(b"ALARM_OFF\n")
        time.sleep(ALARM_BLINK_INTERVAL)
    print("[ALARM] Stopped blinking")


# --- Arduino listener ---
def arduino_listener():
    global arduino
    while True:
        try:
            if arduino and arduino.in_waiting:
                line = arduino.readline().decode('utf-8', errors='ignore').strip()
                if line == "MOTION":
                    print("[EVENT] Motion detected!")
                    handle_motion_event()
                elif line == "TOUCH":
                    print("[EVENT] Touch detected! Triggering alarm.")
                    if arduino and not alarm_active:
                        alarm_active = True
                        threading.Thread(target=blink_alarm_continuous, daemon=True).start()
        except Exception as e:
            print("[ERROR] Arduino listener:", e)
        time.sleep(0.1)

threading.Thread(target=arduino_listener, daemon=True).start()

# --- API Routes ---
@app.route("/")
def home():
    return jsonify({"status": "Smart Security System active"})

# --- Authentication ---
@app.route("/login", methods=["POST"])
def login():
    global logged_in_user
    data = request.json
    username = data.get("username")
    password = data.get("password")
    if username == "admin" and password == "1234":
        logged_in_user = username
        return jsonify({"status": "success", "message": "Login successful"})
    return jsonify({"status": "error", "message": "Invalid credentials"}), 401

@app.route("/logout", methods=["POST"])
def logout():
    global logged_in_user
    logged_in_user = None
    return jsonify({"status": "success", "message": "Logged out"})

@app.route("/events")
def get_events():
    return jsonify(events)

@app.route("/device-status")
def device_status():
    return jsonify({
        "led": alarm_active,
        "buzzer": alarm_active,
        "camera_active": camera.isOpened()
    })

@app.route("/reset-alarm", methods=["POST"])
def reset_alarm():
    global alarm_active
    alarm_active = False
    if arduino:
        arduino.write(b"SAFE\n")
    return jsonify({"status": "alarm reset"})

@app.route("/lock-system", methods=["POST"])
def lock_now():
    if lock_system():
        return jsonify({"status": "success", "message": "System locked"})
    return jsonify({"status": "error", "message": "Failed to lock system"}), 500

# --- Serve Evidence Images ---
@app.route("/evidence/<path:filename>")
def get_evidence_image(filename):
    return send_from_directory(EVIDENCE_DIR, filename)

# --- Live Video Feed ---
def generate_video():
    global last_frame
    while True:
        if last_frame is None:
            time.sleep(0.1)
            continue
        with camera_lock:
            frame = last_frame.copy()
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

@app.route("/video")
def video_feed():
    return Response(generate_video(), mimetype="multipart/x-mixed-replace; boundary=frame")

# --- Run Server ---
if __name__ == "__main__":
    print("[INFO] Smart Security System Running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000)
