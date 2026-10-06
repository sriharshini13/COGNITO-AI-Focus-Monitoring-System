import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request

# ── Download the face landmarker model ──────────────────────────
MODEL_PATH = 'face_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        MODEL_PATH
    )
    print("Model downloaded!")

# ── Setup Face Landmarker ────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# ── Feature calculation functions ───────────────────────────────

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def get_EAR(landmarks, w, h):
    L = [(landmarks[i].x * w, landmarks[i].y * h) for i in [33, 160, 158, 133, 153, 144]]
    R = [(landmarks[i].x * w, landmarks[i].y * h) for i in [362, 385, 387, 263, 373, 380]]
    def ear(e):
        return (euclidean(e[1], e[5]) + euclidean(e[2], e[4])) / (2.0 * euclidean(e[0], e[3]))
    return (ear(L) + ear(R)) / 2.0

def get_MAR(landmarks, w, h):
    mouth = [(landmarks[i].x * w, landmarks[i].y * h) for i in [61, 291, 13, 14, 78, 308]]
    return (euclidean(mouth[2], mouth[3]) + euclidean(mouth[4], mouth[5])) / (2.0 * euclidean(mouth[0], mouth[1]))

def get_gaze(landmarks, w, h):
    left_iris  = landmarks[468]
    right_iris = landmarks[473]
    gaze_x = (left_iris.x + right_iris.x) / 2.0
    gaze_y = (left_iris.y + right_iris.y) / 2.0
    return gaze_x, gaze_y

def get_head_pose(landmarks, w, h):
    nose  = landmarks[1]
    chin  = landmarks[152]
    l_eye = landmarks[33]
    r_eye = landmarks[263]
    pitch = (nose.y - chin.y)
    yaw   = (r_eye.x - l_eye.x)
    return pitch, yaw

# ── Main extraction loop ─────────────────────────────────────────

frames_df = pd.read_csv('features/frames_index.csv')
print(f"Total frames to process: {len(frames_df)}")

results = []
skipped = 0

for _, row in tqdm(frames_df.iterrows(), total=len(frames_df), desc="Extracting features"):
    img_path    = row['frame_path']
    focus_label = row['focus_level']

    img = cv2.imread(img_path)
    if img is None:
        skipped += 1
        continue

    h, w = img.shape[:2]
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = detector.detect(mp_image)

    if not result.face_landmarks:
        skipped += 1
        continue

    lm = result.face_landmarks[0]

    try:
        ear            = get_EAR(lm, w, h)
        mar            = get_MAR(lm, w, h)
        gaze_x, gaze_y = get_gaze(lm, w, h)
        pitch, yaw     = get_head_pose(lm, w, h)

        results.append({
            'EAR':         round(ear,    4),
            'MAR':         round(mar,    4),
            'Gaze_X':      round(gaze_x, 4),
            'Gaze_Y':      round(gaze_y, 4),
            'Head_Pitch':  round(pitch,  4),
            'Head_Yaw':    round(yaw,    4),
            'focus_level': focus_label
        })
    except:
        skipped += 1
        continue

detector.close()

# Save results
features_df = pd.DataFrame(results)
features_df.to_csv('features/features.csv', index=False)

print(f"\nDone!")
print(f"Features extracted : {len(features_df)}")
print(f"Frames skipped     : {skipped}")
print(f"\nSample output:")
print(features_df.head())
print(f"\nFocus distribution:")
print(features_df['focus_level'].value_counts())