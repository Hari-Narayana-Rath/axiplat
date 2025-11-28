import os
import time
import re
import math
import json
from threading import Lock

import cv2
import numpy as np
from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_cors import CORS

import mediapipe as mp

# TensorFlow regression model loads only when the dependency is present
try:
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Allow extension to call API

# Base configuration values
MODEL_DIR = os.path.join(os.getcwd(), "model")
AGE_PROTO = os.path.join(MODEL_DIR, "deploy_age.prototxt")
AGE_CAFFE = os.path.join(MODEL_DIR, "age_net.caffemodel")
AGE_LIST = ['(0-2)','(4-6)','(8-12)','(15-20)','(25-32)','(38-43)','(48-53)','(60-100)']
# Numeric midpoint for each group (used to convert soft outputs to numeric age)
AGE_MIDPOINTS = [ (int(a.split('(')[1].split('-')[0]) + int(a.split('-')[1].split(')')[0]))/2.0 for a in AGE_LIST ]

# Sampling cadence and smoothing behavior
SAMPLE_INTERVAL = 0.5
MAX_SECONDS = 10
STABLE_SECONDS = 3
STABILITY_THRESH = 1.0
EMA_ALPHA = float(os.environ.get("AGE_EMA_ALPHA", "0.3"))

# Weighting for each model contribution
DNN_WEIGHT = 0.6
CNN_WEIGHT = 0.4
CAMERA_INDEX = 0
DNN_CONFIDENCE_LOCK = 0.55  # if the top class is confident enough, snap to that age bucket
CALIBRATION_SCALE = float(os.environ.get("AGE_CALIBRATION_SCALE", "0.9"))
CALIBRATION_OFFSET = float(os.environ.get("AGE_CALIBRATION_OFFSET", "-2.5"))

# Adaptive normalization / brightness compensation
DEFAULT_BLOB_MEAN = (
    78.4263377603,
    87.7689143744,
    114.895847746
)
DYNAMIC_MEAN_BLEND = float(os.environ.get("AGE_DYNAMIC_MEAN_BLEND", "0.35"))
BRIGHTNESS_REFERENCE = float(os.environ.get("AGE_BRIGHTNESS_REFERENCE", "132.0"))
BRIGHTNESS_CORRECTION = float(os.environ.get("AGE_BRIGHTNESS_CORRECTION", "0.018"))
BRIGHTNESS_CORRECTION_CLAMP = float(os.environ.get("AGE_BRIGHTNESS_CLAMP", "6.0"))

# Model loading
USE_DNN = False
age_net = None
if os.path.isfile(AGE_PROTO) and os.path.isfile(AGE_CAFFE):
    try:
        age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_CAFFE)
        USE_DNN = True
        print("Loaded OpenCV DNN age model.")
    except Exception as e:
        print("Failed to load DNN age model:", e)

age_cnn_model = None
if TF_AVAILABLE:
    cnn_path = os.path.join(MODEL_DIR, "cnn_age_model.h5")   # 🔥 corrected filename
    if os.path.isfile(cnn_path):
        try:
            age_cnn_model = load_model(cnn_path)
            print("Loaded CNN regression model:", cnn_path)
        except Exception as e:
            print("Failed to load CNN:", e)
    else:
        print("No CNN model found; continuing without regression CNN.")
else:
    print("TensorFlow not available; CNN regression disabled.")

# Face landmark detector
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Shared state between streaming and endpoints
lock = Lock()

latest_ema_age = None
sampled_ages = []
final_median_age = None
final_age_group = None
access_allowed = False
stream_active = False
camera_error = None

# Verification tokens stored in memory (persisted to json between runs)
verification_tokens = {}  # token -> timestamp
VERIFICATION_TOKEN_FILE = "verification_tokens.json"
VERIFICATION_EXPIRY_HOURS = 24

# Default fallback passcode (override via env)
PASSCODE = os.environ.get("AGE_GATE_PASSCODE", "admin123")
REMOTE_SHUTDOWN_ENABLED = os.environ.get("AXIPLAT_ENABLE_REMOTE_SHUTDOWN", "0").lower() in ("1","true","yes")
SHUTDOWN_SECRET = os.environ.get("AXIPLAT_SHUTDOWN_SECRET", PASSCODE)

# Load existing tokens
if os.path.isfile(VERIFICATION_TOKEN_FILE):
    try:
        with open(VERIFICATION_TOKEN_FILE, 'r') as f:
            verification_tokens = json.load(f)
    except:
        verification_tokens = {}

# Helper routines
def adaptive_blob_mean(face_img):
    """Blend default mean subtraction with current frame stats."""
    if face_img is None or DYNAMIC_MEAN_BLEND <= 0:
        return DEFAULT_BLOB_MEAN
    b_mean, g_mean, r_mean, _ = cv2.mean(face_img)
    blended = []
    for default_val, current_val in zip(DEFAULT_BLOB_MEAN, (b_mean, g_mean, r_mean)):
        blended.append((1.0 - DYNAMIC_MEAN_BLEND) * default_val + DYNAMIC_MEAN_BLEND * current_val)
    return tuple(blended)


def brightness_age_adjust(face_img):
    """Convert overall frame brightness into a gentle age correction."""
    if face_img is None or BRIGHTNESS_CORRECTION == 0:
        return 0.0
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    delta = BRIGHTNESS_REFERENCE - float(np.mean(gray))
    adjustment = delta * BRIGHTNESS_CORRECTION
    return float(np.clip(adjustment, -BRIGHTNESS_CORRECTION_CLAMP, BRIGHTNESS_CORRECTION_CLAMP))


def soft_expected_age_from_dnn(face_img):
    if age_net is None:
        return None
    try:
        mean_vals = adaptive_blob_mean(face_img)
        blob = cv2.dnn.blobFromImage(
            face_img, 1.0, (227,227),
            mean_vals,
            swapRB=False
        )
        age_net.setInput(blob)
        preds = age_net.forward()[0]

        e = np.exp(preds - np.max(preds))
        probs = e / e.sum()

        top_idx = int(np.argmax(probs))
        top_prob = float(probs[top_idx])
        discrete_age = AGE_MIDPOINTS[top_idx]
        expected = float(np.dot(probs, AGE_MIDPOINTS))

        raw_age = discrete_age if top_prob >= DNN_CONFIDENCE_LOCK else expected
        calibrated = CALIBRATION_SCALE * raw_age + CALIBRATION_OFFSET
        calibrated += brightness_age_adjust(face_img)
        return float(calibrated)
    except:
        return None

def cnn_regression_age(face_img):
    if age_cnn_model is None:
        return None
    try:
        img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (64,64))
        arr = img.astype("float32")/255.0
        arr = np.expand_dims(arr, axis=0)
        pred = age_cnn_model.predict(arr)
        return float(pred[0][0])
    except:
        return None

def crop_face_from_landmarks(frame, landmarks):
    h, w, _ = frame.shape
    xs = [int(lm.x * w) for lm in landmarks.landmark]
    ys = [int(lm.y * h) for lm in landmarks.landmark]
    x_min, x_max = max(0,min(xs)), min(w-1,max(xs))
    y_min, y_max = max(0,min(ys)), min(h-1,max(ys))

    mw = int((x_max-x_min)*0.2)
    mh = int((y_max-y_min)*0.2)

    x1=max(0,x_min-mw); y1=max(0,y_min-mh)
    x2=min(w-1,x_max+mw); y2=min(h-1,y_max+mh)

    if x2-x1<10 or y2-y1<10:
        return None

    return frame[y1:y2, x1:x2].copy()

def normalize_face(face_img):
    try:
        lab=cv2.cvtColor(face_img,cv2.COLOR_BGR2LAB)
        l,a,b=cv2.split(lab)
        clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
        l2=clahe.apply(l)
        lab2=cv2.merge((l2,a,b))
        return cv2.cvtColor(lab2,cv2.COLOR_LAB2BGR)
    except:
        return face_img

def ema_update(prev,val,alpha=EMA_ALPHA):
    if prev is None:
        return val
    return prev*(1-alpha)+val*alpha

def group_from_median(m):
    diffs=[abs(m-mp) for mp in AGE_MIDPOINTS]
    idx=int(np.argmin(diffs))
    return AGE_LIST[idx]

def save_verification_tokens():
    """Save tokens to file"""
    try:
        with open(VERIFICATION_TOKEN_FILE, 'w') as f:
            json.dump(verification_tokens, f)
    except:
        pass

def generate_verification_token():
    """Generate a unique verification token"""
    import secrets
    return secrets.token_urlsafe(32)

def is_token_valid(token):
    """Check if token exists and hasn't expired"""
    if token not in verification_tokens:
        return False
    token_time = verification_tokens[token]
    age_hours = (time.time() - token_time) / 3600
    if age_hours > VERIFICATION_EXPIRY_HOURS:
        del verification_tokens[token]
        save_verification_tokens()
        return False
    return True

# Frame generator for the MJPEG stream
def gen_frames():
    global latest_ema_age, sampled_ages, final_median_age, final_age_group, access_allowed, stream_active, camera_error

    with lock:
        sampled_ages=[]
        latest_ema_age=None
        final_median_age=None
        final_age_group=None
        access_allowed=False
        stream_active=True
        camera_error=None

    cap=cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        with lock:
            camera_error="Unable to access camera index {}".format(CAMERA_INDEX)
            stream_active=False
        return

    start=time.time()
    last_sample=start
    stable_since=None
    last_ema=None

    while True:
        now=time.time()
        if now-start > MAX_SECONDS:
            break

        success,frame=cap.read()
        if not success:
            camera_error="Camera read failure"
            break

        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=face_mesh.process(rgb)

        frame_display=frame.copy()
        if res and res.multi_face_landmarks:
            lm=res.multi_face_landmarks[0]

            h,w,_=frame.shape
            for p in lm.landmark:
                x=int(p.x*w); y=int(p.y*h)
                cv2.circle(frame_display,(x,y),1,(0,255,255),-1)

            face=crop_face_from_landmarks(frame,lm)
            if face is not None:
                face=normalize_face(face)

                dnn_age=soft_expected_age_from_dnn(face) if USE_DNN else None
                cnn_age=cnn_regression_age(face) if age_cnn_model else None

                vals=[]; wts=[]
                if dnn_age is not None:
                    vals.append(dnn_age); wts.append(DNN_WEIGHT)
                if cnn_age is not None:
                    vals.append(cnn_age); wts.append(CNN_WEIGHT)

                if vals:
                    age=float(np.average(vals,weights=wts))
                else:
                    age=None
            else:
                age=None
        else:
            age=None

        if now-last_sample>=SAMPLE_INTERVAL:
            sampled_ages.append(age if age else 0)
            last_sample=now

            latest_ema_age=ema_update(latest_ema_age, age if age else 0)

            if last_ema is None:
                last_ema=latest_ema_age
                stable_since=now
            else:
                if abs(latest_ema_age-last_ema)<=STABILITY_THRESH:
                    if now-stable_since>=STABLE_SECONDS:
                        break
                else:
                    stable_since=now
                    last_ema=latest_ema_age

        display="Detecting..."
        if latest_ema_age and latest_ema_age>0:
            display=f"Age (EMA): {latest_ema_age:.1f}"
        cv2.putText(frame_display,display,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)

        ret,buf=cv2.imencode('.jpg',frame_display)
        if ret:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+buf.tobytes()+b'\r\n')

    numeric=[s for s in sampled_ages if s and s>0]
    if numeric:
        final_median_age=float(np.median(numeric))
        final_age_group=group_from_median(final_median_age)
        access_allowed=(final_median_age>=14)
    else:
        final_median_age=None
        final_age_group=None
        access_allowed=False

    try:
        cap.release()
    except Exception:
        pass

    stream_active=False
    return

# HTTP routes
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({
        "stream_active": stream_active,
        "ema": None if latest_ema_age is None else float(latest_ema_age),
        "final_median_age": None if final_median_age is None else float(final_median_age),
        "final_group": final_age_group,
        "access_allowed": access_allowed,
        "camera_error": camera_error
    })

@app.route('/insta')
def insta():
    # Generate verification token if access was granted
    if access_allowed:
        token = generate_verification_token()
        verification_tokens[token] = time.time()
        save_verification_tokens()
        return render_template("insta.html", token=token)
    return render_template("insta.html")

@app.route('/passcode', methods=['GET', 'POST'])
def passcode():
    """Passcode entry page for denied access"""
    if request.method == 'POST':
        entered = request.form.get('passcode', '')
        if entered == PASSCODE:
            # Grant access with passcode
            token = generate_verification_token()
            verification_tokens[token] = time.time()
            save_verification_tokens()
            return render_template("insta.html", token=token)
        else:
            return render_template("passcode.html", error="Incorrect passcode. Please try again.")
    return render_template("passcode.html")

@app.route('/api/shutdown', methods=['POST'])
def shutdown_server():
    """API endpoint to shutdown the server"""
    import threading

    if not REMOTE_SHUTDOWN_ENABLED:
        return jsonify({"status": "disabled"}), 403

    payload = request.get_json(silent=True) or {}
    provided_secret = payload.get("secret") or request.headers.get("X-AXIPLAT-SHUTDOWN")
    if SHUTDOWN_SECRET and provided_secret != SHUTDOWN_SECRET:
        return jsonify({"status": "unauthorized"}), 401

    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        def hard_exit():
            time.sleep(1)
            os._exit(0)
        threading.Thread(target=hard_exit, daemon=True).start()
        return jsonify({"status": "Server shutting down... (forced)"}), 202

    def shutdown(target):
        time.sleep(1)  # Give time for response to be sent
        target()

    threading.Thread(target=shutdown, args=(func,), daemon=True).start()
    return jsonify({"status": "Server shutting down..."})

@app.route('/api/verify_token', methods=['POST'])
def verify_token():
    """API endpoint for extension to check if token is valid"""
    data = request.get_json()
    token = data.get('token', '') if data else ''
    if token and is_token_valid(token):
        return jsonify({"valid": True})
    return jsonify({"valid": False}), 401

@app.route('/api/check_verification', methods=['GET'])
def check_verification():
    """API endpoint for extension to check verification status"""
    token = request.args.get('token', '')
    if token and is_token_valid(token):
        return jsonify({"verified": True})
    return jsonify({"verified": False})

@app.route('/control')
def control_panel():
    """Web app control panel for server management"""
    return render_template("control_panel.html")

@app.route('/download/<filename>')
def download_file(filename):
    """Download endpoint for server launcher files"""
    allowed_files = {
        'start_server_hidden.bat': 'start_server_hidden.bat',
        'start_server_portable.bat': 'start_server_portable.bat',
        'start_server_hidden_portable.bat': 'start_server_hidden_portable.bat',
        'start_server.sh': 'start_server.sh',
        'setup_autostart.bat': 'setup_autostart.bat'
    }
    
    if filename in allowed_files:
        file_path = os.path.join(os.getcwd(), allowed_files[filename])
        if os.path.isfile(file_path):
            return send_file(file_path, as_attachment=True, download_name=filename)
    
    return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
