import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
from modules.multi_person import MultiPersonTracker

# Load models
focus_model   = joblib.load('models/cognito_model.pkl')
emotion_model = joblib.load('models/emotion_model.pkl')
tracker       = MultiPersonTracker()

# Setup face landmarker for multiple faces
options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path='face_landmarker.task'),
    num_faces=4,  # detect up to 4 people
    min_face_detection_confidence=0.5
)
detector = vision.FaceLandmarker.create_from_options(options)

# Colors per person
PERSON_COLORS = [
    (0,255,0), (0,165,255),
    (255,0,255), (0,255,255)
]
FOCUS_COLORS = {
    'High':   (0,255,0),
    'Medium': (0,165,255),
    'Low':    (0,0,255)
}

print("Multi-person mode started... Press Q to quit")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    h, w  = frame.shape[:2]
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    if result.face_landmarks:
        people = tracker.update(
            result.face_landmarks,
            focus_model, emotion_model, w, h
        )

        for i, person in enumerate(people):
            color = PERSON_COLORS[i % len(PERSON_COLORS)]
            x1, y1, x2, y2 = person['bbox']

            # Draw bounding box
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)

            # Label above box
            label = (f"{person['id']} | "
                     f"{person['focus']} | "
                     f"{person['emotion']}")
            cv2.rectangle(frame,
                          (x1, y1-30), (x1+len(label)*9, y1),
                          color, -1)
            cv2.putText(frame, label,
                        (x1+2, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0,0,0), 2)

        # Classroom overview at bottom
        cv2.rectangle(frame, (0,h-40), (w,h), (20,20,30), -1)
        cv2.putText(frame,
                    f"Classroom Mode | "
                    f"People detected: {len(people)}",
                    (10, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (150,150,255), 2)

    cv2.imshow('COGNITO - Classroom Mode', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()

# Print classroom summary
print("\n=== Classroom Summary ===")
summary = tracker.get_classroom_summary()
for pid, data in summary.items():
    print(f"{pid}: Focus={data['dominant_focus']} | "
          f"Emotion={data['dominant_emotion']} | "
          f"High Focus={data['high_focus_pct']}% | "
          f"Engaged={data['engaged_pct']}%")