import cv2
import numpy as np
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# ── Load model ───────────────────────────────────────────────────
print("Loading COGNITO model...")
model = joblib.load('models/cognito_model.pkl')
print("Model loaded!")

# ── Setup Face Landmarker ────────────────────────────────────────
MODEL_PATH = 'face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# ── Feature functions ────────────────────────────────────────────
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_EAR(lm, w, h):
    L = [(lm[i].x * w, lm[i].y * h) for i in [33, 160, 158, 133, 153, 144]]
    R = [(lm[i].x * w, lm[i].y * h) for i in [362, 385, 387, 263, 373, 380]]
    def ear(e):
        return (euclidean(e[1], e[5]) + euclidean(e[2], e[4])) / (2.0 * euclidean(e[0], e[3]))
    return (ear(L) + ear(R)) / 2.0

def get_MAR(lm, w, h):
    mouth = [(lm[i].x * w, lm[i].y * h) for i in [61, 291, 13, 14, 78, 308]]
    return (euclidean(mouth[2], mouth[3]) + euclidean(mouth[4], mouth[5])) / (2.0 * euclidean(mouth[0], mouth[1]))

def get_gaze(lm, w, h):
    gaze_x = (lm[468].x + lm[473].x) / 2.0
    gaze_y = (lm[468].y + lm[473].y) / 2.0
    return gaze_x, gaze_y

def get_head_pose(lm, w, h):
    pitch = (lm[1].y - lm[152].y)
    yaw   = (lm[263].x - lm[33].x)
    return pitch, yaw

# ── Color per focus level ────────────────────────────────────────
COLORS = {
    'High':   (0, 255, 0),    # Green
    'Medium': (0, 165, 255),  # Orange
    'Low':    (0, 0, 255)     # Red
}

# ── Webcam loop ──────────────────────────────────────────────────
print("Starting webcam... Press Q to quit")
cap = cv2.VideoCapture(0)

focus_label  = "Detecting..."
focus_color  = (255, 255, 255)
frame_count  = 0
low_start    = None
session_start = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run detection every 5 frames for performance
    if frame_count % 5 == 0:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            try:
                ear            = get_EAR(lm, w, h)
                mar            = get_MAR(lm, w, h)
                gaze_x, gaze_y = get_gaze(lm, w, h)
                pitch, yaw     = get_head_pose(lm, w, h)

                features    = np.array([[ear, mar, gaze_x, gaze_y, pitch, yaw]])
                focus_label = model.predict(features)[0]
                focus_color = COLORS.get(focus_label, (255, 255, 255))

                # Track low focus duration
                if focus_label == 'Low':
                    if low_start is None:
                        low_start = time.time()
                else:
                    low_start = None

            except:
                pass
        else:
            focus_label = "No Face Detected"
            focus_color = (255, 255, 255)

    # ── Draw UI ──────────────────────────────────────────────────
    # Background bar at top
    cv2.rectangle(frame, (0, 0), (w, 60), (30, 30, 30), -1)

    # Focus label
    cv2.putText(frame, f"Focus: {focus_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, focus_color, 2)

    # Session timer
    elapsed = int(time.time() - session_start)
    mins, secs = divmod(elapsed, 60)
    cv2.putText(frame, f"Session: {mins:02d}:{secs:02d}", (w - 200, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Low focus warning
    if low_start and (time.time() - low_start) > 300:  # 5 minutes
        cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 180), -1)
        cv2.putText(frame, "Take a break! You've been unfocused for 5 mins",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Yawn detection
    try:
        if get_MAR(lm, w, h) > 0.6:
            cv2.putText(frame, "Yawning detected - Eye exercise recommended!",
                        (10, h - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
    except:
        pass

    cv2.imshow('COGNITO - Focus Monitor', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()

elapsed = int(time.time() - session_start)
mins, secs = divmod(elapsed, 60)
print(f"\nSession ended. Total time: {mins:02d}:{secs:02d}")